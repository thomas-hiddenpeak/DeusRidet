// safetensors.h — Zero-copy safetensors weight loader
//
// Parses safetensors file header (JSON metadata), provides direct mmap'd
// pointers to tensor data. Supports multi-shard loading via index file.
//
// Adapted from qwen35-orin (src/engine/safetensors.h): safetensors format
// parsing and mmap-based zero-copy loading architecture.
// Original: https://github.com/thomas-hiddenpeak/qwen35-orin

#pragma once

#include "tensor.h"
#include "allocator.h"
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>
#include <functional>

namespace deusridet {

// Metadata for a single tensor within a safetensors file
struct TensorMeta {
    std::string dtype_str;
    DataType    dtype;
    std::vector<int64_t> shape;
    size_t      offset_start;
    size_t      offset_end;
};

// Loads a single safetensors shard via mmap
class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::string& path);
    ~SafetensorsFile() = default;

    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    // Get a non-owning Tensor view into mmap'd data
    std::unique_ptr<Tensor> get_tensor(const std::string& name) const;

    bool has_tensor(const std::string& name) const;
    std::vector<std::string> tensor_names() const;
    const std::unordered_map<std::string, TensorMeta>& metadata() const { return meta_; }

private:
    void parse_header();
    DataType parse_dtype(const std::string& s) const;

    std::string path_;
    std::unique_ptr<MmapAllocator> mmap_;
    std::unordered_map<std::string, TensorMeta> meta_;
    size_t header_size_;
};

// Multi-shard loader: reads model.safetensors.index.json to locate tensors
// across multiple shard files, then provides unified tensor access.
class SafetensorsLoader {
public:
    // Load from model directory (auto-detects single file or sharded)
    explicit SafetensorsLoader(const std::string& model_dir);
    ~SafetensorsLoader() = default;

    SafetensorsLoader(const SafetensorsLoader&) = delete;
    SafetensorsLoader& operator=(const SafetensorsLoader&) = delete;

    // Get tensor by name (searches across all shards)
    std::unique_ptr<Tensor> get_tensor(const std::string& name) const;

    bool has_tensor(const std::string& name) const;
    std::vector<std::string> tensor_names() const;
    size_t shard_count() const { return shards_.size(); }

    // Per-shard streaming: iterate over one shard at a time.
    // Callback receives (shard_index, SafetensorsFile&). After callback returns,
    // the shard can be released to free mmap pages.
    // Use release_shard() inside or after callback to free mmap immediately.
    using ShardCallback = std::function<void(size_t shard_idx, SafetensorsFile& shard)>;
    void for_each_shard(ShardCallback cb);

    // Release a specific shard's mmap (frees physical pages for GPU use).
    // After release, get_tensor() for tensors in this shard will throw.
    void release_shard(size_t shard_idx);

    // Static helper: stream-load from model_dir, processing one shard at a time.
    // Each shard is mmap'd, callback processes its tensors, then mmap is released.
    // Peak memory: single shard mmap + accumulated cudaMalloc'd weights.
    static void stream_load(const std::string& model_dir, ShardCallback cb);

private:
    std::string model_dir_;
    // Shard files, each loaded via mmap
    std::vector<std::unique_ptr<SafetensorsFile>> shards_;
    // Tensor name → shard index for fast lookup
    std::unordered_map<std::string, size_t> tensor_to_shard_;
};

} // namespace deusridet
