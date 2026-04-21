/**
 * @file awaken_hello.h
 * @philosophical_role The greeting a client hears on WS connect — a
 *         snapshot of consciousness state (wakefulness, KV cache, persona,
 *         sampling configs) and the active prompts. Extracted from
 *         awaken.cpp (Step 7e) so that the hello envelope lives next
 *         to the text-command router it answers.
 * @serves awaken_hello.cpp.
 */
#pragma once

namespace deusridet {

class WsServer;
class ConscientiStream;
class CacheManager;
struct PersonaConfig;

namespace actus {

void send_consciousness_hello(int fd,
                              WsServer& server,
                              ConscientiStream& consciousness,
                              CacheManager& llm_cache,
                              const PersonaConfig& persona_cfg,
                              bool llm_loaded);

}  // namespace actus
}  // namespace deusridet
