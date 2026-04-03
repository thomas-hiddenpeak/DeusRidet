// safetensors.cpp — Safetensors loader implementation
//
// Format: [8-byte header_len (LE uint64)] [JSON header] [raw tensor data]
// JSON header maps tensor names to {dtype, shape, data_offsets: [start, end]}
//
// Multi-shard: model.safetensors.index.json contains:
//   { "weight_map": { "tensor_name": "shard_file.safetensors", ... } }
//
// Adapted from qwen35-orin (src/engine/safetensors.cpp): JSON parsing logic
// and safetensors format handling.
// Original: https://github.com/thomas-hiddenpeak/qwen35-orin

#include "safetensors.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <sys/mman.h>
#include <filesystem>

namespace deusridet {

// ============================================================================
// Minimal JSON helpers (safetensors header only — no external dependency)
// ============================================================================

namespace {

std::string extract_json_value(const std::string& json, const std::string& key) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    pos = json.find(":", pos);
    if (pos == std::string::npos) return "";

    size_t start = json.find_first_not_of(" \t\n\r", pos + 1);
    if (start == std::string::npos) return "";

    if (json[start] == '"') {
        size_t end = json.find("\"", start + 1);
        return json.substr(start + 1, end - start - 1);
    } else if (json[start] == '[') {
        size_t end = json.find("]", start + 1);
        return json.substr(start, end - start + 1);
    }
    return "";
}

std::vector<int64_t> parse_json_array(const std::string& s) {
    std::vector<int64_t> result;
    size_t start = 1;
    while (start < s.length() - 1) {
        size_t end = s.find_first_of(",]", start);
        if (end == std::string::npos) break;
        std::string num = s.substr(start, end - start);
        num.erase(std::remove_if(num.begin(), num.end(), ::isspace), num.end());
        if (!num.empty()) {
            result.push_back(std::stoll(num));
        }
        start = end + 1;
    }
    return result;
}

// Parse model.safetensors.index.json → weight_map
// Returns: tensor_name → shard_filename
std::unordered_map<std::string, std::string> parse_index_json(const std::string& path) {
    std::unordered_map<std::string, std::string> weight_map;

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return weight_map;

    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());

    // Find "weight_map" object
    size_t wm_pos = content.find("\"weight_map\"");
    if (wm_pos == std::string::npos) return weight_map;

    size_t obj_start = content.find("{", wm_pos);
    if (obj_start == std::string::npos) return weight_map;

    // Find matching closing brace
    int depth = 1;
    size_t pos = obj_start + 1;
    while (pos < content.size() && depth > 0) {
        if (content[pos] == '{') depth++;
        else if (content[pos] == '}') depth--;
        pos++;
    }
    std::string wm_json = content.substr(obj_start, pos - obj_start);

    // Parse key-value pairs: "tensor_name": "shard_file"
    size_t p = 1;
    while (p < wm_json.size() - 1) {
        size_t ks = wm_json.find("\"", p);
        if (ks == std::string::npos) break;
        size_t ke = wm_json.find("\"", ks + 1);
        if (ke == std::string::npos) break;
        std::string key = wm_json.substr(ks + 1, ke - ks - 1);

        size_t vs = wm_json.find("\"", ke + 1);
        if (vs == std::string::npos) break;
        size_t ve = wm_json.find("\"", vs + 1);
        if (ve == std::string::npos) break;
        std::string val = wm_json.substr(vs + 1, ve - vs - 1);

        weight_map[key] = val;
        p = ve + 1;
    }
    return weight_map;
}

} // anonymous namespace

// ============================================================================
// SafetensorsFile — single shard loader
// ============================================================================

SafetensorsFile::SafetensorsFile(const std::string& path)
    : path_(path), header_size_(0)
{
    mmap_ = std::make_unique<MmapAllocator>(path_);
    parse_header();
}

void SafetensorsFile::parse_header() {
    void* base = mmap_->base_ptr();
    if (base == MAP_FAILED) {
        throw std::runtime_error("Cannot parse header: mmap failed");
    }

    // First 8 bytes: header length (uint64_t LE)
    uint64_t header_len = 0;
    std::memcpy(&header_len, base, sizeof(uint64_t));
    header_size_ = sizeof(uint64_t) + header_len;

    const char* json_start = static_cast<const char*>(base) + sizeof(uint64_t);
    std::string header_json(json_start, header_len);

    // Parse top-level keys
    size_t pos = 1;
    while (pos < header_json.length() - 1) {
        size_t key_start = header_json.find("\"", pos);
        if (key_start == std::string::npos) break;
        size_t key_end = header_json.find("\"", key_start + 1);
        std::string key = header_json.substr(key_start + 1, key_end - key_start - 1);

        if (key == "__metadata__") {
            size_t obj_start = header_json.find("{", key_end);
            int depth = 1;
            size_t p = obj_start + 1;
            while (p < header_json.size() && depth > 0) {
                if (header_json[p] == '{') depth++;
                else if (header_json[p] == '}') depth--;
                p++;
            }
            pos = p;
            continue;
        }

        size_t obj_start = header_json.find("{", key_end);
        size_t obj_end = header_json.find("}", obj_start);
        std::string obj_json = header_json.substr(obj_start, obj_end - obj_start + 1);

        TensorMeta meta;
        meta.dtype_str = extract_json_value(obj_json, "dtype");
        meta.dtype = parse_dtype(meta.dtype_str);

        std::string shape_str = extract_json_value(obj_json, "shape");
        meta.shape = parse_json_array(shape_str);

        std::string offsets_str = extract_json_value(obj_json, "data_offsets");
        auto offsets = parse_json_array(offsets_str);
        if (offsets.size() == 2) {
            meta.offset_start = offsets[0];
            meta.offset_end   = offsets[1];
        }

        meta_[key] = meta;
        pos = obj_end + 1;
    }
}

DataType SafetensorsFile::parse_dtype(const std::string& s) const {
    if (s == "F32")  return DataType::FP32;
    if (s == "F16")  return DataType::FP16;
    if (s == "BF16") return DataType::BF16;
    if (s == "I8")   return DataType::INT8;
    if (s == "U8")   return DataType::U8;
    if (s == "I32")  return DataType::INT32;
    return DataType::UNKNOWN;
}

std::unique_ptr<Tensor> SafetensorsFile::get_tensor(const std::string& name) const {
    auto it = meta_.find(name);
    if (it == meta_.end()) {
        throw std::runtime_error("Tensor not found: " + name + " in " + path_);
    }

    const TensorMeta& meta = it->second;
    char* data_ptr = static_cast<char*>(mmap_->base_ptr()) +
                     header_size_ + meta.offset_start;
    size_t nbytes = meta.offset_end - meta.offset_start;

    return std::make_unique<Tensor>(meta.shape, meta.dtype,
                                    static_cast<void*>(data_ptr), nbytes);
}

bool SafetensorsFile::has_tensor(const std::string& name) const {
    return meta_.find(name) != meta_.end();
}

std::vector<std::string> SafetensorsFile::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(meta_.size());
    for (const auto& [k, v] : meta_) {
        names.push_back(k);
    }
    return names;
}

// ============================================================================
// SafetensorsLoader — multi-shard loader
// ============================================================================

SafetensorsLoader::SafetensorsLoader(const std::string& model_dir)
    : model_dir_(model_dir)
{
    namespace fs = std::filesystem;

    std::string index_path = model_dir + "/model.safetensors.index.json";

    if (fs::exists(index_path)) {
        // Sharded model: parse index to discover shard files
        auto weight_map = parse_index_json(index_path);

        // Collect unique shard filenames
        std::unordered_map<std::string, size_t> shard_name_to_idx;
        for (const auto& [tensor_name, shard_file] : weight_map) {
            if (shard_name_to_idx.find(shard_file) == shard_name_to_idx.end()) {
                size_t idx = shards_.size();
                shard_name_to_idx[shard_file] = idx;
                std::string shard_path = model_dir + "/" + shard_file;
                shards_.push_back(std::make_unique<SafetensorsFile>(shard_path));
            }
            tensor_to_shard_[tensor_name] = shard_name_to_idx[shard_file];
        }

        fprintf(stderr, "[SafetensorsLoader] Loaded %zu shards, %zu tensors from %s\n",
                shards_.size(), tensor_to_shard_.size(), model_dir.c_str());
    } else {
        // Single file
        std::string single_path = model_dir + "/model.safetensors";
        if (!fs::exists(single_path)) {
            throw std::runtime_error("No safetensors files found in " + model_dir);
        }
        shards_.push_back(std::make_unique<SafetensorsFile>(single_path));
        for (const auto& name : shards_[0]->tensor_names()) {
            tensor_to_shard_[name] = 0;
        }

        fprintf(stderr, "[SafetensorsLoader] Loaded single shard, %zu tensors from %s\n",
                tensor_to_shard_.size(), model_dir.c_str());
    }
}

std::unique_ptr<Tensor> SafetensorsLoader::get_tensor(const std::string& name) const {
    auto it = tensor_to_shard_.find(name);
    if (it == tensor_to_shard_.end()) {
        throw std::runtime_error("Tensor not found in any shard: " + name);
    }
    return shards_[it->second]->get_tensor(name);
}

bool SafetensorsLoader::has_tensor(const std::string& name) const {
    return tensor_to_shard_.find(name) != tensor_to_shard_.end();
}

std::vector<std::string> SafetensorsLoader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensor_to_shard_.size());
    for (const auto& [k, v] : tensor_to_shard_) {
        names.push_back(k);
    }
    return names;
}

void SafetensorsLoader::for_each_shard(ShardCallback cb) {
    for (size_t i = 0; i < shards_.size(); i++) {
        if (shards_[i]) {
            cb(i, *shards_[i]);
        }
    }
}

void SafetensorsLoader::release_shard(size_t shard_idx) {
    if (shard_idx < shards_.size()) {
        shards_[shard_idx].reset();  // triggers ~MmapAllocator → munmap + FADV_DONTNEED
    }
}

// Static: stream one shard at a time, release mmap between shards
void SafetensorsLoader::stream_load(const std::string& model_dir, ShardCallback cb) {
    namespace fs = std::filesystem;

    std::string index_path = model_dir + "/model.safetensors.index.json";

    if (fs::exists(index_path)) {
        auto weight_map = parse_index_json(index_path);

        // Collect unique shard filenames in discovery order
        std::vector<std::string> shard_files;
        std::unordered_map<std::string, bool> seen;
        for (const auto& [tensor_name, shard_file] : weight_map) {
            if (!seen.count(shard_file)) {
                seen[shard_file] = true;
                shard_files.push_back(shard_file);
            }
        }

        for (size_t i = 0; i < shard_files.size(); i++) {
            std::string shard_path = model_dir + "/" + shard_files[i];
            auto shard = std::make_unique<SafetensorsFile>(shard_path);
            cb(i, *shard);
            // shard destroyed here → mmap released → FADV_DONTNEED
        }
    } else {
        std::string single_path = model_dir + "/model.safetensors";
        if (!fs::exists(single_path)) {
            throw std::runtime_error("No safetensors files found in " + model_dir);
        }
        auto shard = std::make_unique<SafetensorsFile>(single_path);
        cb(0, *shard);
    }
}

} // namespace deusridet
