/**
 * @file ws_server.h
 * @philosophical_role Declaration of the WS/HTTP server and the callback shape each subsystem must adhere to when publishing state.
 * @serves Actus::awaken, all subsystem facades that publish to the WebUI.
 */
#pragma once
// ws_server.h — Self-contained WebSocket + static file server (epoll-based).
// Serves the WebUI and handles bidirectional binary/text WebSocket streams.
// Single-threaded event loop in a dedicated thread; communicates with the
// engine via registered callbacks.  No external library dependency.

#include <cstdint>
#include <cstddef>
#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>
#include <unordered_map>

namespace deusridet {

struct WsServerConfig {
    uint16_t port       = 8080;
    std::string static_dir;           // root for serving webui files
    int max_clients     = 32;
    size_t max_frame_payload = 65536; // max single WS frame payload bytes
};

class WsServer {
public:
    // Callback signatures: client_fd used as client identifier.
    using OnConnect    = std::function<void(int)>;
    using OnDisconnect = std::function<void(int)>;
    using OnText       = std::function<void(int, const std::string&)>;
    using OnBinary     = std::function<void(int, const uint8_t*, size_t)>;

    WsServer() = default;
    ~WsServer();

    WsServer(const WsServer&) = delete;
    WsServer& operator=(const WsServer&) = delete;

    bool start(const WsServerConfig& cfg);
    void stop();
    bool running() const { return running_.load(std::memory_order_relaxed); }

    // Send to a single client.
    void send_text(int client_fd, const std::string& msg);
    void send_binary(int client_fd, const uint8_t* data, size_t len);

    // Broadcast to all WebSocket clients.
    void broadcast_text(const std::string& msg);

    // Register callbacks (set before start()).
    void set_on_connect(OnConnect cb)       { on_connect_ = std::move(cb); }
    void set_on_disconnect(OnDisconnect cb) { on_disconnect_ = std::move(cb); }
    void set_on_text(OnText cb)             { on_text_ = std::move(cb); }
    void set_on_binary(OnBinary cb)         { on_binary_ = std::move(cb); }

private:
    // Per-connection state.
    struct Client {
        int fd = -1;
        bool is_ws = false;           // true after successful WS upgrade
        std::string recv_buf;         // accumulated inbound bytes
        std::string send_buf;         // pending outbound bytes
    };

    void run();                       // epoll event loop (runs in thread_)
    void accept_client();
    void handle_read(int fd);
    void flush_send(int fd);
    void close_client(int fd);

    // HTTP handling (pre-upgrade or static files).
    void process_http(Client& c);
    void serve_static(Client& c, const std::string& url_path);

    // WebSocket protocol.
    void ws_handshake(Client& c, const std::string& ws_key);
    void process_ws(Client& c);
    void ws_send_frame(Client& c, uint8_t opcode,
                       const uint8_t* data, size_t len);

    // State.
    std::atomic<bool> running_{false};
    std::thread thread_;
    int listen_fd_ = -1;
    int epoll_fd_  = -1;
    int event_fd_  = -1;              // for waking epoll on stop()
    WsServerConfig config_;

    // Client tracking (accessed only from epoll thread except send paths).
    std::recursive_mutex clients_mu_;
    std::unordered_map<int, Client> clients_;

    // Callbacks.
    OnConnect    on_connect_;
    OnDisconnect on_disconnect_;
    OnText       on_text_;
    OnBinary     on_binary_;
};

} // namespace deusridet
