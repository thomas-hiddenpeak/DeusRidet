/**
 * @file config.h
 * @philosophical_role Declarative shape of every runtime knob the operator may turn. A header that forbids implicit defaults from spreading into subsystem code.
 * @serves All subsystems that include runtime configuration.
 */
// config.h — Unified configuration parser
//
// Parses key=value .conf files. Supports comments (#), whitespace trimming.
// Each DeusRidet config file (machina.conf, conscientia.conf, persona.conf)
// is loaded into a flat string map; typed accessors provide defaults.

#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace deusridet {

class Config {
public:
    Config() = default;

    // Load from .conf file (key=value format, # comments)
    bool load(const std::string& path);

    // Typed accessors with defaults
    std::string get_string(const std::string& key, const std::string& def = "") const;
    int         get_int(const std::string& key, int def = 0) const;
    double      get_double(const std::string& key, double def = 0.0) const;
    bool        get_bool(const std::string& key, bool def = false) const;

    // Check existence
    bool has(const std::string& key) const;

    // Set a value (for CLI override)
    void set(const std::string& key, const std::string& value);

    // Print all entries (for debug)
    void print() const;

private:
    std::unordered_map<std::string, std::string> entries_;
    std::string path_;
};

// Structured config for the machina inference engine
struct MachinaConfig {
    std::string llm_model_dir;
    std::string asr_model_dir;
    std::string tts_model_dir;

    double kv_cache_gb      = 14.0;
    double ssm_conv_gb      = 2.0;
    double scratch_gb       = 4.0;

    int    max_chunk_size    = 2048;
    int    max_ssm_slots     = 64;

    bool   mtp_enabled       = true;
    int    mtp_num_drafts    = 1;

    bool   cache_enabled     = true;
    std::string cache_dir    = "/tmp/deusridet_cache";
    double cache_max_gb      = 20.0;
    int    cache_chunk_size   = 256;

    // Construct from Config
    static MachinaConfig from_config(const Config& cfg);

    void print() const;
};

// Structured config for persona and response behavior
struct PersonaConfig {
    std::string name;                    // entity name (e.g. "黑娃")
    std::vector<std::string> aliases;    // name variants / wake words

    int  speech_max_tokens   = 80;       // short, concise speech output
    int  thinking_max_tokens = 256;      // internal analysis budget
    int  decode_interleave_tokens = 4;   // check input every N decode tokens

    static PersonaConfig from_config(const Config& cfg);
    void print() const;
};

} // namespace deusridet
