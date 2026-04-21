/**
 * @file ws_server_http.cpp
 * @philosophical_role Peer TU of ws_server.cpp carrying the HTTP-side of the Nexus network
 *         surface: request parsing, static-file serving, the WebSocket upgrade handshake,
 *         and the small SHA-1 / base64 / mime_type helpers only those paths use. Split out
 *         under R1 because the HTTP methods together are ~167 lines and kept ws_server.cpp
 *         over the 500-line hard cap; the runtime loop (epoll, accept, read, flush, frame
 *         decode, send, broadcast) stays in the core TU where its cohesion earns its keep.
 * @serves WsServer callers (Actus::awaken via nexus/ws_server.h); no external surface changed.
 */
#include "nexus/ws_server.h"
#include "communis/log.h"

#include <sys/epoll.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <climits>
#include <algorithm>
#include <vector>
#include <string>

namespace deusridet {

// ============================================================================
// SHA-1 (RFC 3174) — needed for Sec-WebSocket-Accept computation.
// Minimal public-domain implementation (no dependency on OpenSSL).
// ============================================================================

static void sha1(const uint8_t* data, size_t len, uint8_t out[20]) {
    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE,
             h3 = 0x10325476, h4 = 0xC3D2E1F0;

    // Pre-processing: pad to 64-byte blocks.
    size_t ml = len * 8;
    size_t padded = ((len + 8) / 64 + 1) * 64;
    std::vector<uint8_t> msg(padded, 0);
    memcpy(msg.data(), data, len);
    msg[len] = 0x80;
    for (int i = 0; i < 8; i++)
        msg[padded - 1 - i] = (uint8_t)(ml >> (i * 8));

    auto rol = [](uint32_t v, int n) { return (v << n) | (v >> (32 - n)); };

    for (size_t off = 0; off < padded; off += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++)
            w[i] = (uint32_t)msg[off + i*4+0] << 24 | (uint32_t)msg[off + i*4+1] << 16 |
                   (uint32_t)msg[off + i*4+2] <<  8 | (uint32_t)msg[off + i*4+3];
        for (int i = 16; i < 80; i++)
            w[i] = rol(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);

        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if      (i < 20) { f = (b & c) | (~b & d);       k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d;                k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else              { f = b ^ c ^ d;                k = 0xCA62C1D6; }
            uint32_t t = rol(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rol(b, 30); b = a; a = t;
        }
        h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
    }
    auto put32 = [&](uint8_t* p, uint32_t v) {
        p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v;
    };
    put32(out, h0); put32(out+4, h1); put32(out+8, h2);
    put32(out+12, h3); put32(out+16, h4);
}

static std::string base64_encode(const uint8_t* data, size_t len) {
    static const char t[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string r;
    r.reserve((len + 2) / 3 * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = (uint32_t)data[i] << 16;
        if (i+1 < len) n |= (uint32_t)data[i+1] << 8;
        if (i+2 < len) n |= (uint32_t)data[i+2];
        r += t[(n >> 18) & 0x3F];
        r += t[(n >> 12) & 0x3F];
        r += (i+1 < len) ? t[(n >> 6) & 0x3F] : '=';
        r += (i+2 < len) ? t[n & 0x3F] : '=';
    }
    return r;
}

// ============================================================================
// Static file helpers
// ============================================================================

static const char* mime_type(const std::string& path) {
    auto ends = [&](const char* s) {
        size_t sl = strlen(s);
        return path.size() >= sl && path.compare(path.size()-sl, sl, s) == 0;
    };
    if (ends(".html")) return "text/html; charset=utf-8";
    if (ends(".css"))  return "text/css; charset=utf-8";
    if (ends(".js"))   return "application/javascript; charset=utf-8";
    if (ends(".json")) return "application/json";
    if (ends(".svg"))  return "image/svg+xml";
    if (ends(".png"))  return "image/png";
    if (ends(".ico"))  return "image/x-icon";
    if (ends(".woff2"))return "font/woff2";
    return "application/octet-stream";
}

// ============================================================================
// HTTP request parsing and dispatch
// ============================================================================

void WsServer::process_http(Client& c) {
    // Wait until we have a complete header block (\r\n\r\n).
    auto hdr_end = c.recv_buf.find("\r\n\r\n");
    if (hdr_end == std::string::npos) return;

    std::string header = c.recv_buf.substr(0, hdr_end);
    c.recv_buf.erase(0, hdr_end + 4);

    // Parse request line.
    auto first_nl = header.find("\r\n");
    std::string req_line = header.substr(0, first_nl);

    // Extract method and path.
    size_t sp1 = req_line.find(' ');
    size_t sp2 = req_line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
        close_client(c.fd);
        return;
    }
    std::string method = req_line.substr(0, sp1);
    std::string path   = req_line.substr(sp1 + 1, sp2 - sp1 - 1);

    // Extract headers into a simple map (lowercased keys).
    auto get_hdr = [&](const std::string& name) -> std::string {
        std::string search = "\r\n" + name + ":";
        // Case-insensitive search.
        std::string lower_hdr = header;
        std::transform(lower_hdr.begin(), lower_hdr.end(),
                       lower_hdr.begin(), ::tolower);
        std::string lower_name = search;
        std::transform(lower_name.begin(), lower_name.end(),
                       lower_name.begin(), ::tolower);
        auto pos = lower_hdr.find(lower_name);
        if (pos == std::string::npos) return "";
        pos += search.size();
        auto end = header.find("\r\n", pos);
        std::string val = header.substr(pos, end - pos);
        // Trim leading whitespace.
        size_t start = val.find_first_not_of(" \t");
        return (start != std::string::npos) ? val.substr(start) : "";
    };

    // Check for WebSocket upgrade.
    std::string upgrade = get_hdr("upgrade");
    std::string ws_key  = get_hdr("sec-websocket-key");
    std::string lower_upgrade = upgrade;
    std::transform(lower_upgrade.begin(), lower_upgrade.end(),
                   lower_upgrade.begin(), ::tolower);

    if (method == "GET" && lower_upgrade == "websocket" && !ws_key.empty()) {
        ws_handshake(c, ws_key);
        return;
    }

    // Static file serving for GET requests.
    if (method == "GET") {
        serve_static(c, path);
        return;
    }

    // Unsupported method.
    std::string resp = "HTTP/1.1 405 Method Not Allowed\r\n"
                       "Content-Length: 0\r\n\r\n";
    c.send_buf += resp;
    flush_send(c.fd);
}

// ============================================================================
// Static file serving
// ============================================================================

void WsServer::serve_static(Client& c, const std::string& url_path) {
    // Security: reject path traversal.
    if (url_path.find("..") != std::string::npos) {
        c.send_buf += "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n";
        flush_send(c.fd);
        return;
    }

    std::string rel = (url_path == "/" || url_path.empty()) ? "/index.html" : url_path;
    std::string full = config_.static_dir + rel;

    // Resolve and verify under static_dir.
    char resolved[PATH_MAX];
    if (!realpath(full.c_str(), resolved)) {
        c.send_buf += "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
        flush_send(c.fd);
        return;
    }
    char root_resolved[PATH_MAX];
    if (!realpath(config_.static_dir.c_str(), root_resolved) ||
        strncmp(resolved, root_resolved, strlen(root_resolved)) != 0) {
        c.send_buf += "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n";
        flush_send(c.fd);
        return;
    }

    struct stat st;
    if (stat(resolved, &st) < 0 || !S_ISREG(st.st_mode)) {
        c.send_buf += "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
        flush_send(c.fd);
        return;
    }

    FILE* fp = fopen(resolved, "rb");
    if (!fp) {
        c.send_buf += "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n";
        flush_send(c.fd);
        return;
    }
    std::vector<char> body(st.st_size);
    fread(body.data(), 1, st.st_size, fp);
    fclose(fp);

    char hdr[512];
    snprintf(hdr, sizeof(hdr),
             "HTTP/1.1 200 OK\r\n"
             "Content-Type: %s\r\n"
             "Content-Length: %ld\r\n"
             "Cache-Control: no-cache\r\n"
             "\r\n", mime_type(resolved), (long)st.st_size);

    c.send_buf += hdr;
    c.send_buf.append(body.data(), body.size());

    // Enable EPOLLOUT to flush.
    struct epoll_event ev{};
    ev.events  = EPOLLIN | EPOLLOUT;
    ev.data.fd = c.fd;
    epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, c.fd, &ev);
}

// ============================================================================
// WebSocket handshake
// ============================================================================

void WsServer::ws_handshake(Client& c, const std::string& ws_key) {
    const char* magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string input = ws_key + magic;
    uint8_t hash[20];
    sha1((const uint8_t*)input.data(), input.size(), hash);
    std::string accept = base64_encode(hash, 20);

    char resp[256];
    snprintf(resp, sizeof(resp),
             "HTTP/1.1 101 Switching Protocols\r\n"
             "Upgrade: websocket\r\n"
             "Connection: Upgrade\r\n"
             "Sec-WebSocket-Accept: %s\r\n"
             "\r\n", accept.c_str());

    c.send_buf += resp;
    c.is_ws = true;

    struct epoll_event ev{};
    ev.events  = EPOLLIN | EPOLLOUT;
    ev.data.fd = c.fd;
    epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, c.fd, &ev);

    if (on_connect_) on_connect_(c.fd);
    LOG_INFO("WS", "WebSocket upgraded (fd=%d)", c.fd);
}

} // namespace deusridet
