// ws_server.cpp — Self-contained WebSocket + static-file server.
// Handles WS upgrade, binary/text frames, and serves the WebUI.
// Uses epoll (level-triggered) for non-blocking I/O.

#include "nexus/ws_server.h"
#include "communis/log.h"

#include <sys/socket.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <climits>
#include <algorithm>

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

static bool set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

// ============================================================================
// WsServer lifecycle
// ============================================================================

WsServer::~WsServer() { stop(); }

bool WsServer::start(const WsServerConfig& cfg) {
    config_ = cfg;

    listen_fd_ = socket(AF_INET6, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        LOG_ERROR("WS", "socket(): %s", strerror(errno));
        return false;
    }
    int on = 1, off = 0;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    setsockopt(listen_fd_, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof(off));
    set_nonblocking(listen_fd_);

    struct sockaddr_in6 addr{};
    addr.sin6_family = AF_INET6;
    addr.sin6_port   = htons(cfg.port);
    addr.sin6_addr   = in6addr_any;

    if (bind(listen_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        LOG_ERROR("WS", "bind(:%d): %s", cfg.port, strerror(errno));
        close(listen_fd_); listen_fd_ = -1;
        return false;
    }
    if (listen(listen_fd_, 64) < 0) {
        LOG_ERROR("WS", "listen(): %s", strerror(errno));
        close(listen_fd_); listen_fd_ = -1;
        return false;
    }

    epoll_fd_ = epoll_create1(0);
    event_fd_ = eventfd(0, EFD_NONBLOCK);

    struct epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.fd = listen_fd_;
    epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, listen_fd_, &ev);
    ev.data.fd = event_fd_;
    epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, event_fd_, &ev);

    running_.store(true, std::memory_order_release);
    thread_ = std::thread(&WsServer::run, this);

    LOG_INFO("WS", "Listening on port %d  (webui: %s)",
             cfg.port, cfg.static_dir.c_str());
    return true;
}

void WsServer::stop() {
    if (!running_.load(std::memory_order_acquire)) return;
    running_.store(false, std::memory_order_release);
    uint64_t v = 1;
    (void)write(event_fd_, &v, sizeof(v));   // wake epoll
    if (thread_.joinable()) thread_.join();

    std::lock_guard<std::recursive_mutex> lk(clients_mu_);
    for (auto& [fd, _] : clients_) close(fd);
    clients_.clear();

    if (listen_fd_ >= 0) { close(listen_fd_); listen_fd_ = -1; }
    if (epoll_fd_  >= 0) { close(epoll_fd_);  epoll_fd_  = -1; }
    if (event_fd_  >= 0) { close(event_fd_);  event_fd_  = -1; }
    LOG_INFO("WS", "Server stopped");
}

// ============================================================================
// Epoll event loop (runs in server thread)
// ============================================================================

void WsServer::run() {
    constexpr int MAX_EVENTS = 64;
    struct epoll_event events[MAX_EVENTS];

    while (running_.load(std::memory_order_relaxed)) {
        int n = epoll_wait(epoll_fd_, events, MAX_EVENTS, 200);
        for (int i = 0; i < n; i++) {
            int fd = events[i].data.fd;
            if (fd == listen_fd_) {
                accept_client();
            } else if (fd == event_fd_) {
                uint64_t v; (void)read(event_fd_, &v, sizeof(v));
            } else {
                if (events[i].events & (EPOLLERR | EPOLLHUP)) {
                    close_client(fd);
                } else {
                    if (events[i].events & EPOLLIN)  handle_read(fd);
                    if (events[i].events & EPOLLOUT) flush_send(fd);
                }
            }
        }
    }
}

void WsServer::accept_client() {
    while (true) {
        struct sockaddr_in6 addr{};
        socklen_t alen = sizeof(addr);
        int fd = accept4(listen_fd_, (struct sockaddr*)&addr, &alen, SOCK_NONBLOCK);
        if (fd < 0) break;

        int on = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));

        struct epoll_event ev{};
        ev.events  = EPOLLIN;
        ev.data.fd = fd;
        epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev);

        std::lock_guard<std::recursive_mutex> lk(clients_mu_);
        if ((int)clients_.size() >= config_.max_clients) {
            close(fd);
            continue;
        }
        Client& c = clients_[fd];
        c.fd = fd;
        c.is_ws = false;
        LOG_INFO("WS", "Client connected (fd=%d)", fd);
    }
}

void WsServer::handle_read(int fd) {
    char buf[8192];
    while (true) {
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n > 0) {
            std::lock_guard<std::recursive_mutex> lk(clients_mu_);
            auto it = clients_.find(fd);
            if (it == clients_.end()) return;
            it->second.recv_buf.append(buf, n);
        } else if (n == 0) {
            close_client(fd);
            return;
        } else {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                close_client(fd);
                return;
            }
            break; // EAGAIN — done reading for now
        }
    }

    // Process accumulated data.
    std::lock_guard<std::recursive_mutex> lk(clients_mu_);
    auto it = clients_.find(fd);
    if (it == clients_.end()) return;
    Client& c = it->second;

    if (c.is_ws)
        process_ws(c);
    else
        process_http(c);
}

void WsServer::flush_send(int fd) {
    std::lock_guard<std::recursive_mutex> lk(clients_mu_);
    auto it = clients_.find(fd);
    if (it == clients_.end()) return;
    Client& c = it->second;

    while (!c.send_buf.empty()) {
        ssize_t n = ::send(fd, c.send_buf.data(), c.send_buf.size(), MSG_NOSIGNAL);
        if (n <= 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            close_client(fd);
            return;
        }
        c.send_buf.erase(0, n);
    }

    if (c.send_buf.empty()) {
        // No more pending data — stop watching EPOLLOUT.
        struct epoll_event ev{};
        ev.events  = EPOLLIN;
        ev.data.fd = fd;
        epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &ev);
    }
}

void WsServer::close_client(int fd) {
    epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
    close(fd);

    bool was_ws = false;
    {
        std::lock_guard<std::recursive_mutex> lk(clients_mu_);
        auto it = clients_.find(fd);
        if (it != clients_.end()) {
            was_ws = it->second.is_ws;
            clients_.erase(it);
        }
    }
    if (was_ws && on_disconnect_) on_disconnect_(fd);
    LOG_INFO("WS", "Client disconnected (fd=%d)", fd);
}

// ============================================================================
// HTTP request handling (pre-upgrade or static files)
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

// ============================================================================
// WebSocket frame processing
// ============================================================================

void WsServer::process_ws(Client& c) {
    while (true) {
        const uint8_t* d = (const uint8_t*)c.recv_buf.data();
        size_t avail = c.recv_buf.size();
        if (avail < 2) return;

        bool fin    = d[0] & 0x80;
        uint8_t op  = d[0] & 0x0F;
        bool masked = d[1] & 0x80;
        uint64_t plen = d[1] & 0x7F;
        size_t hdr_len = 2 + (masked ? 4 : 0);

        if (plen == 126) {
            if (avail < 4) return;
            plen = ((uint64_t)d[2] << 8) | d[3];
            hdr_len = 4 + (masked ? 4 : 0);
        } else if (plen == 127) {
            if (avail < 10) return;
            plen = 0;
            for (int i = 0; i < 8; i++)
                plen = (plen << 8) | d[2 + i];
            hdr_len = 10 + (masked ? 4 : 0);
        }

        if (plen > config_.max_frame_payload) {
            close_client(c.fd);
            return;
        }
        size_t frame_len = hdr_len + plen;
        if (avail < frame_len) return;

        // Unmask payload (client→server frames are always masked).
        const uint8_t* mask_key = masked ? d + (hdr_len - 4) : nullptr;
        const uint8_t* payload = d + hdr_len;
        std::vector<uint8_t> data(plen);
        if (masked) {
            for (size_t i = 0; i < plen; i++)
                data[i] = payload[i] ^ mask_key[i & 3];
        } else {
            memcpy(data.data(), payload, plen);
        }
        c.recv_buf.erase(0, frame_len);

        if (!fin) continue; // Ignore fragmented frames for now.

        switch (op) {
        case 0x1: // Text
            if (on_text_)
                on_text_(c.fd, std::string(data.begin(), data.end()));
            break;
        case 0x2: // Binary
            if (on_binary_)
                on_binary_(c.fd, data.data(), data.size());
            break;
        case 0x8: // Close
            ws_send_frame(c, 0x8, data.data(), std::min(data.size(), (size_t)2));
            close_client(c.fd);
            return;
        case 0x9: // Ping → Pong
            ws_send_frame(c, 0xA, data.data(), data.size());
            break;
        case 0xA: // Pong — ignore
            break;
        }
    }
}

void WsServer::ws_send_frame(Client& c, uint8_t opcode,
                              const uint8_t* data, size_t len) {
    // Server→client frames are NOT masked.
    std::string frame;
    frame += (char)(0x80 | opcode);
    if (len < 126) {
        frame += (char)len;
    } else if (len < 65536) {
        frame += (char)126;
        frame += (char)(len >> 8);
        frame += (char)(len & 0xFF);
    } else {
        frame += (char)127;
        for (int i = 7; i >= 0; i--)
            frame += (char)((len >> (i * 8)) & 0xFF);
    }
    frame.append((const char*)data, len);
    c.send_buf += frame;

    struct epoll_event ev{};
    ev.events  = EPOLLIN | EPOLLOUT;
    ev.data.fd = c.fd;
    epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, c.fd, &ev);
}

// ============================================================================
// Public send / broadcast
// ============================================================================

void WsServer::send_text(int client_fd, const std::string& msg) {
    std::lock_guard<std::recursive_mutex> lk(clients_mu_);
    auto it = clients_.find(client_fd);
    if (it == clients_.end() || !it->second.is_ws) return;
    ws_send_frame(it->second, 0x1, (const uint8_t*)msg.data(), msg.size());
}

void WsServer::send_binary(int client_fd, const uint8_t* data, size_t len) {
    std::lock_guard<std::recursive_mutex> lk(clients_mu_);
    auto it = clients_.find(client_fd);
    if (it == clients_.end() || !it->second.is_ws) return;
    ws_send_frame(it->second, 0x2, data, len);
}

void WsServer::broadcast_text(const std::string& msg) {
    std::lock_guard<std::recursive_mutex> lk(clients_mu_);
    for (auto& [fd, c] : clients_) {
        if (c.is_ws)
            ws_send_frame(c, 0x1, (const uint8_t*)msg.data(), msg.size());
    }
}

} // namespace deusridet
