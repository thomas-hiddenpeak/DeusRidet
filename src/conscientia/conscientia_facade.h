/**
 * @file conscientia_facade.h
 * @philosophical_role The bridge between Conscientia (the continuous Prefill engine)
 *         and Nexus (the voice to the outer world). Owns every consciousness-derived
 *         broadcast — decode output, per-token speech streaming, wakefulness/metric
 *         heartbeats. Mirror of auditus_facade: facade installs, does not own.
 *         ConscientiStream and WsServer lifetimes remain with the caller.
 * @serves cmd_test_ws and any future Actus verb that needs to expose the
 *         consciousness stream over a WS channel.
 */
#pragma once

namespace deusridet {

class WsServer;
class ConscientiStream;

namespace conscientia {

// Decode output → ws "consciousness_decode" envelope + stdout trace.
// Wires `consciousness.set_on_decode(...)`.
void install_decode_callback(ConscientiStream& consciousness,
                             WsServer& server);

// Per-token speech streaming → ws "speech_token" envelope.
// token_id == -1 signals start of new speech (frontend resets its accumulator).
// Wires `consciousness.set_on_speech_token(...)`.
void install_speech_token_callback(ConscientiStream& consciousness,
                                   WsServer& server);

// Wakefulness / metrics / system-memory heartbeat → ws "consciousness_state"
// envelope. Samples CUDA free/total, MemAvailable, VmRSS at broadcast time.
// Wires `consciousness.set_on_state(...)`.
void install_state_callback(ConscientiStream& consciousness,
                            WsServer& server);

}  // namespace conscientia
}  // namespace deusridet
