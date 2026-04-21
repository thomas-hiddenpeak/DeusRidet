/**
 * @file ws_server.cpp
 * @philosophical_role Sole external network surface — WebSocket + HTTP. Nexus translates between the outside world and internal subsystems; no subsystem exposes itself to the network directly.
 * @serves Actus::awaken binds it; Auditus, Conscientia, Vox facades pipe through it.
 */
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

// HTTP request parsing, static-file serving, WebSocket handshake, and their
// supporting helpers (sha1 / base64_encode / mime_type) live in ws_server_http.cpp
// as a peer TU to keep this file under the R1 500-line hard cap. process_http,
// serve_static and ws_handshake are declared as members of WsServer in
// nexus/ws_server.h; the split is invisible to callers.

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
