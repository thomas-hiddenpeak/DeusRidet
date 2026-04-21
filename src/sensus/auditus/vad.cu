/**
 * @file src/sensus/auditus/vad.cu
 * @philosophical_role
 *   Energy VAD — the simplest voice-activity detector, GPU-accelerated only at the energy computation. Keeps the state machine on CPU where sequential hangover logic is natural.
 * @serves
 *   Auditus pipeline default VAD path when silero/fsmn are disabled; diagnostic baseline.
 */
// vad.cu — Energy-based VAD with GPU batch energy computation.
//
// Simple but effective for pipeline testing. The GPU kernel computes
// mean log-mel energy per frame; the state machine runs on CPU (cheap,
// and needs sequential state tracking for hangover logic).

#include "vad.h"
#include "../../communis/log.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace deusridet {

// ============================================================================
// GPU kernel: mean energy per frame
// ============================================================================

__global__ void frame_energy_kernel(
    const float* __restrict__ mel,     // [n_frames x n_mels]
    float*       __restrict__ energy,  // [n_frames]
    int n_mels)
{
    int frame = blockIdx.x;
    int tid   = threadIdx.x;

    extern __shared__ float smem[];

    // Each thread accumulates a partial sum.
    float sum = 0.0f;
    const float* row = mel + frame * n_mels;
    for (int i = tid; i < n_mels; i += blockDim.x) {
        sum += row[i];
    }
    smem[tid] = sum;
    __syncthreads();

    // Warp reduction then block reduction.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        energy[frame] = smem[0] / (float)n_mels;
    }
}

void launch_frame_energy(
    const float* d_mel,
    float* d_energy,
    int n_frames,
    int n_mels,
    cudaStream_t stream)
{
    if (n_frames <= 0) return;
    int block = 128; // n_mels = 128, one thread per mel bin.
    int smem = block * sizeof(float);
    frame_energy_kernel<<<n_frames, block, smem, stream>>>(
        d_mel, d_energy, n_mels);
}

// ============================================================================
// Single-frame VAD (CPU state machine)
// ============================================================================

VadResult VoiceActivityDetector::process_frame(const float* mel_frame, int n_mels) {
    // Compute mean energy on CPU for single frame.
    float sum = 0.0f;
    for (int i = 0; i < n_mels; i++) {
        sum += mel_frame[i];
    }
    float energy = sum / (float)n_mels;

    bool frame_is_speech = energy > cfg_.energy_threshold;

    VadResult r{};
    r.energy = energy;

    if (frame_is_speech) {
        silence_count_ = 0;
        speech_count_++;

        if (!in_speech_ && speech_count_ >= cfg_.min_speech_frames) {
            in_speech_ = true;
            r.segment_start = true;
        }
    } else {
        // Update noise floor estimate from non-speech frames.
        noise_floor_ = noise_alpha_ * energy + (1.0f - noise_alpha_) * noise_floor_;

        if (in_speech_) {
            silence_count_++;
            if (silence_count_ >= cfg_.min_silence_frames) {
                in_speech_ = false;
                r.segment_end = true;
                speech_count_ = 0;
            }
        } else {
            speech_count_ = 0;
        }
    }

    r.is_speech = in_speech_;
    r.speech_frames = speech_count_;
    return r;
}

// ============================================================================
// Batch processing: GPU energy + CPU state machine
// ============================================================================

void VoiceActivityDetector::process_batch(
    const float* d_mel_frames, int n_frames, int n_mels,
    VadResult* results)
{
    if (n_frames <= 0) return;

    // Allocate device buffer for per-frame energy.
    float* d_energy = nullptr;
    cudaMalloc(&d_energy, n_frames * sizeof(float));

    launch_frame_energy(d_mel_frames, d_energy, n_frames, n_mels);
    cudaDeviceSynchronize();

    // Copy energy to host.
    std::vector<float> h_energy(n_frames);
    cudaMemcpy(h_energy.data(), d_energy, n_frames * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_energy);

    // Run state machine on each frame sequentially.
    // Build temporary mel frame on host for the state machine (just energy).
    for (int i = 0; i < n_frames; i++) {
        // We already have the mean energy; create a fake mel frame.
        float fake_mel = h_energy[i];
        bool frame_is_speech = fake_mel > cfg_.energy_threshold;

        VadResult& r = results[i];
        r.energy = fake_mel;
        r.segment_start = false;
        r.segment_end = false;

        if (frame_is_speech) {
            silence_count_ = 0;
            speech_count_++;
            if (!in_speech_ && speech_count_ >= cfg_.min_speech_frames) {
                in_speech_ = true;
                r.segment_start = true;
            }
        } else {
            noise_floor_ = noise_alpha_ * fake_mel + (1.0f - noise_alpha_) * noise_floor_;
            if (in_speech_) {
                silence_count_++;
                if (silence_count_ >= cfg_.min_silence_frames) {
                    in_speech_ = false;
                    r.segment_end = true;
                    speech_count_ = 0;
                }
            } else {
                speech_count_ = 0;
            }
        }
        r.is_speech = in_speech_;
        r.speech_frames = speech_count_;
    }
}

void VoiceActivityDetector::reset() {
    speech_count_ = 0;
    silence_count_ = 0;
    in_speech_ = false;
    noise_floor_ = -10.0f;
}

} // namespace deusridet
