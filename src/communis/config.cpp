// config.cpp — Configuration parser implementation

#include "config.h"
#include "log.h"
#include <fstream>
#include <algorithm>

namespace deusridet {

bool Config::load(const std::string& path) {
    path_ = path;
    std::ifstream ifs(path);
    if (!ifs) {
        LOG_ERROR("Config", "Cannot open config file: %s", path.c_str());
        return false;
    }

    std::string line;
    int line_num = 0;
    while (std::getline(ifs, line)) {
        line_num++;
        // Strip comments
        auto hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);

        // Trim whitespace
        auto ltrim = line.find_first_not_of(" \t");
        if (ltrim == std::string::npos) continue;
        auto rtrim = line.find_last_not_of(" \t\r\n");
        line = line.substr(ltrim, rtrim - ltrim + 1);
        if (line.empty()) continue;

        // Split on first '='
        auto eq = line.find('=');
        if (eq == std::string::npos) {
            LOG_WARN("Config", "%s:%d: no '=' found, skipping", path.c_str(), line_num);
            continue;
        }

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);

        // Trim key and value
        auto kt = key.find_last_not_of(" \t");
        if (kt != std::string::npos) key.resize(kt + 1);
        auto vt = val.find_first_not_of(" \t");
        if (vt != std::string::npos) val = val.substr(vt);
        else val.clear();

        entries_[key] = val;
    }

    LOG_INFO("Config", "Loaded %zu entries from %s", entries_.size(), path.c_str());
    return true;
}

std::string Config::get_string(const std::string& key, const std::string& def) const {
    auto it = entries_.find(key);
    return (it != entries_.end()) ? it->second : def;
}

int Config::get_int(const std::string& key, int def) const {
    auto it = entries_.find(key);
    if (it == entries_.end()) return def;
    try { return std::stoi(it->second); }
    catch (...) { return def; }
}

double Config::get_double(const std::string& key, double def) const {
    auto it = entries_.find(key);
    if (it == entries_.end()) return def;
    try { return std::stod(it->second); }
    catch (...) { return def; }
}

bool Config::get_bool(const std::string& key, bool def) const {
    auto it = entries_.find(key);
    if (it == entries_.end()) return def;
    return (it->second == "true" || it->second == "1" || it->second == "yes");
}

bool Config::has(const std::string& key) const {
    return entries_.find(key) != entries_.end();
}

void Config::set(const std::string& key, const std::string& value) {
    entries_[key] = value;
}

void Config::print() const {
    fprintf(stderr, "[Config] %s (%zu entries):\n", path_.c_str(), entries_.size());
    for (const auto& [k, v] : entries_) {
        fprintf(stderr, "  %s = %s\n", k.c_str(), v.c_str());
    }
}

// ============================================================================
// MachinaConfig
// ============================================================================

MachinaConfig MachinaConfig::from_config(const Config& cfg) {
    MachinaConfig mc;
    mc.llm_model_dir    = cfg.get_string("llm_model_dir");
    mc.asr_model_dir    = cfg.get_string("asr_model_dir");
    mc.tts_model_dir    = cfg.get_string("tts_model_dir");

    mc.kv_cache_gb      = cfg.get_double("kv_cache_gb",      14.0);
    mc.ssm_conv_gb      = cfg.get_double("ssm_conv_gb",      2.0);
    mc.scratch_gb       = cfg.get_double("scratch_gb",        4.0);

    mc.max_chunk_size   = cfg.get_int("max_chunk_size",       2048);
    mc.max_ssm_slots    = cfg.get_int("max_ssm_slots",        64);

    mc.mtp_enabled      = cfg.get_bool("mtp_enabled",         true);
    mc.mtp_num_drafts   = cfg.get_int("mtp_num_drafts",       1);

    mc.cache_enabled    = cfg.get_bool("cache_enabled",        true);
    mc.cache_dir        = cfg.get_string("cache_dir",          "/tmp/deusridet_cache");
    mc.cache_max_gb     = cfg.get_double("cache_max_gb",       20.0);
    mc.cache_chunk_size = cfg.get_int("cache_chunk_size",      256);

    return mc;
}

void MachinaConfig::print() const {
    fprintf(stderr, "[MachinaConfig]\n");
    fprintf(stderr, "  llm_model_dir  = %s\n", llm_model_dir.c_str());
    fprintf(stderr, "  asr_model_dir  = %s\n", asr_model_dir.c_str());
    fprintf(stderr, "  tts_model_dir  = %s\n", tts_model_dir.c_str());
    fprintf(stderr, "  kv_cache_gb    = %.1f\n", kv_cache_gb);
    fprintf(stderr, "  ssm_conv_gb    = %.1f\n", ssm_conv_gb);
    fprintf(stderr, "  scratch_gb     = %.1f\n", scratch_gb);
    fprintf(stderr, "  max_chunk_size = %d\n", max_chunk_size);
    fprintf(stderr, "  mtp_enabled    = %s\n", mtp_enabled ? "true" : "false");
    fprintf(stderr, "  mtp_num_drafts = %d\n", mtp_num_drafts);
    fprintf(stderr, "  cache_enabled  = %s\n", cache_enabled ? "true" : "false");
    fprintf(stderr, "  cache_dir      = %s\n", cache_dir.c_str());
}

} // namespace deusridet
