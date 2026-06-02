/**
 * @file awaken_serve.cpp
 * @philosophical_role Implements the per-front-door ingress wiring declared in
 *         awaken_serve.h. Pure orchestration: it binds the shared pipeline's
 *         entry points to one socket's callbacks and nothing more.
 * @serves awaken.cpp.
 */
#include "actus/awaken_serve.h"

#include <cstdio>

#include "actus/awaken_consciousness.h"   // ConscientiaBootstrap
#include "actus/awaken_hello.h"           // send_consciousness_hello
#include "actus/awaken_router.h"          // handle_ws_text_command
#include "nexus/ws_server.h"
#include "sensus/auditus/auditus_facade.h"  // install_ws_binary_callback

namespace deusridet {
namespace actus {

void wire_server_ingress(WsServer& server,
                         AudioPipeline& audio,
                         ConscientiaBootstrap& cb,
                         std::atomic<bool>& loopback,
                         std::atomic<uint64_t>& total_frames,
                         std::atomic<uint64_t>& total_bytes,
                         orator::DiarizenPeriodicWorker* worker,
                         orator::DiarizenPipeline* native) {
    WsServer* sp = &server;
    server.set_on_connect([sp, &cb](int fd) {
        send_consciousness_hello(fd, *sp, cb.stream, cb.cache, cb.persona_cfg, cb.loaded);
    });
    server.set_on_disconnect([](int fd) {
        printf("[awaken] WS client disconnected (fd=%d)\n", fd);
    });
    server.set_on_text([sp, &audio, &cb, &loopback, worker, native](int fd, const std::string& msg) {
        handle_ws_text_command(fd, msg, audio, *sp, cb.stream, loopback, cb.loaded, worker, native);
    });
    auditus::install_ws_binary_callback(server, audio, total_frames, total_bytes, loopback);
}

}  // namespace actus
}  // namespace deusridet
