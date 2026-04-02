// config.h — Unified configuration parser
//
// Parses key=value .conf files. Supports comments (#), whitespace trimming.
// Each DeusRidet config file (machina.conf, conscientia.conf, persona.conf)
// is loaded into a flat string map; typed accessors provide defaults.

#pragma once

#include <string>
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

} // namespace deusridet
