/**
 * @file awaken_serve.h
 * @philosophical_role One entity, two front doors. The user UI (9527) and the
 *         debug console (3721) are distinct WsServer instances, yet they feed
 *         the same audio + consciousness pipeline and must answer commands on
 *         their own socket. This helper wires a single server's ingress
 *         (connect / disconnect / text command / binary PCM) so `awaken()`
 *         can call it once per front door instead of duplicating the block.
 * @serves awaken.cpp (the sole caller).
 */
#pragma once

#include <atomic>
#include <cstdint>

namespace deusridet {

class AudioPipeline;
class WsServer;
struct ConscientiaBootstrap;
namespace auditus { class TranscriptHoldback; }
namespace orator {
class DiarizenPeriodicWorker;
class DiarizenPipeline;
}

namespace actus {

// Wire one WsServer's ingress callbacks. Every reference is owned by the
// caller (awaken's scope) and outlives the server, so the installed lambdas
// hold them safely. Entity-side broadcasts are NOT wired here — they
// originate on the primary server and reach mirrors via broadcast fan-out.
void wire_server_ingress(WsServer& server,
                         AudioPipeline& audio,
                         ConscientiaBootstrap& cb,
                         std::atomic<bool>& loopback,
                         std::atomic<uint64_t>& total_frames,
                         std::atomic<uint64_t>& total_bytes,
                         auditus::TranscriptHoldback*& holdback,
                         orator::DiarizenPeriodicWorker*& worker,
                         orator::DiarizenPipeline*& native);

}  // namespace actus
}  // namespace deusridet
