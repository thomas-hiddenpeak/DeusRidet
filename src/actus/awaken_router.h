/**
 * @file awaken_router.h
 * @philosophical_role The control surface of awaken. Every runtime knob the
 *         WebUI can turn — VAD thresholds, ASR parameters, speaker-DB edits,
 *         consciousness enables/prompts/sampling — flows through this single
 *         function. Kept as a peer Actus TU (not a subsystem facade) because
 *         the router is by construction a cross-subsystem integration point:
 *         it speaks to Auditus *and* Conscientia in one place, and that is
 *         precisely the charter of an Actus verb.
 * @serves awaken (the sole caller; invoked from WsServer::set_on_text).
 */
#pragma once

#include <atomic>
#include <string>

namespace deusridet {

class AudioPipeline;
class WsServer;
class ConscientiStream;

namespace actus {

// Dispatches a single text-frame command received by the awaken WS server.
// Called from the WsServer text-callback thread. Performs no allocations
// beyond what each command's JSON reply already does. Unknown commands are
// logged to stdout.
void handle_ws_text_command(int fd,
                            const std::string& msg,
                            AudioPipeline& audio,
                            WsServer& server,
                            ConscientiStream& consciousness,
                            std::atomic<bool>& loopback,
                            bool llm_loaded);

// Peer routing helper (R1 split): handles the four consciousness_* prefixes
// plus text_input. Returns true if the message matched one of those prefixes
// and was consumed; false otherwise so the caller can fall through. Guard
// llm_loaded at the call site — this helper assumes the LLM is ready.
bool handle_ws_consciousness_command(int fd,
                                     const std::string& msg,
                                     WsServer& server,
                                     ConscientiStream& consciousness);

}  // namespace actus
}  // namespace deusridet
