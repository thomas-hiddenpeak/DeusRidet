/**
 * @file ring_buffer.h
 * @philosophical_role Lock-free SPSC ring buffer — the canonical seam between producer and consumer threads. Decouples rhythm: the perceiver's pace must not dictate the thinker's pace.
 * @serves Auditus (PCM ingress), Conscientia (inter-thread handoff), Vox (PCM egress) and anywhere a bounded queue is needed without a mutex.
 */
// ring_buffer.h — Lock-free SPSC (single-producer single-consumer) ring buffer.
//
// Designed for the audio pipeline: WS callback thread produces PCM chunks,
// GPU processing thread consumes them.  Wait-free push/pop with eventfd
// notification so the consumer can block efficiently.
//
// Memory layout: contiguous byte buffer with head (write) and tail (read)
// indices.  Power-of-two capacity for fast modular arithmetic (mask).

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sys/eventfd.h>
#include <unistd.h>

namespace deusridet {

class RingBuffer {
public:
    // capacity_bytes must be a power of two.
    explicit RingBuffer(size_t capacity_bytes)
        : capacity_(capacity_bytes)
        , mask_(capacity_bytes - 1)
        , head_(0)
        , tail_(0)
    {
        // Enforce power-of-two.
        if (capacity_bytes == 0 || (capacity_bytes & (capacity_bytes - 1)) != 0) {
            capacity_ = 1;
            while (capacity_ < capacity_bytes) capacity_ <<= 1;
            mask_ = capacity_ - 1;
        }
        buf_ = new uint8_t[capacity_];
        event_fd_ = eventfd(0, EFD_NONBLOCK | EFD_SEMAPHORE);
    }

    ~RingBuffer() {
        delete[] buf_;
        if (event_fd_ >= 0) close(event_fd_);
    }

    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    // Producer: write data into the ring. Returns bytes actually written
    // (may be less than len if buffer is full — non-blocking).
    size_t push(const uint8_t* data, size_t len) {
        size_t h = head_.load(std::memory_order_relaxed);
        size_t t = tail_.load(std::memory_order_acquire);
        size_t avail = capacity_ - (h - t);
        size_t to_write = len < avail ? len : avail;
        if (to_write == 0) return 0;

        // Write in up to two segments (wrap-around).
        size_t pos = h & mask_;
        size_t first = capacity_ - pos;
        if (first >= to_write) {
            memcpy(buf_ + pos, data, to_write);
        } else {
            memcpy(buf_ + pos, data, first);
            memcpy(buf_, data + first, to_write - first);
        }

        head_.store(h + to_write, std::memory_order_release);

        // Signal consumer.
        if (event_fd_ >= 0) {
            uint64_t one = 1;
            (void)write(event_fd_, &one, sizeof(one));
        }

        return to_write;
    }

    // Consumer: read up to max_len bytes. Returns bytes actually read.
    size_t pop(uint8_t* out, size_t max_len) {
        size_t t = tail_.load(std::memory_order_relaxed);
        size_t h = head_.load(std::memory_order_acquire);
        size_t avail = h - t;
        size_t to_read = max_len < avail ? max_len : avail;
        if (to_read == 0) return 0;

        size_t pos = t & mask_;
        size_t first = capacity_ - pos;
        if (first >= to_read) {
            memcpy(out, buf_ + pos, to_read);
        } else {
            memcpy(out, buf_ + pos, first);
            memcpy(out + first, buf_, to_read - first);
        }

        tail_.store(t + to_read, std::memory_order_release);

        // Drain one event.
        if (event_fd_ >= 0) {
            uint64_t v;
            (void)read(event_fd_, &v, sizeof(v));
        }

        return to_read;
    }

    // Consumer: peek at available data without consuming.
    size_t peek(uint8_t* out, size_t max_len) const {
        size_t t = tail_.load(std::memory_order_relaxed);
        size_t h = head_.load(std::memory_order_acquire);
        size_t avail = h - t;
        size_t to_read = max_len < avail ? max_len : avail;
        if (to_read == 0) return 0;

        size_t pos = t & mask_;
        size_t first = capacity_ - pos;
        if (first >= to_read) {
            memcpy(out, buf_ + pos, to_read);
        } else {
            memcpy(out, buf_ + pos, first);
            memcpy(out + first, buf_, to_read - first);
        }
        return to_read;
    }

    // Bytes available for reading.
    size_t available() const {
        size_t h = head_.load(std::memory_order_acquire);
        size_t t = tail_.load(std::memory_order_relaxed);
        return h - t;
    }

    // Free space for writing.
    size_t free_space() const {
        return capacity_ - available();
    }

    size_t capacity() const { return capacity_; }

    // eventfd for epoll/poll-based consumer wakeup.
    int event_fd() const { return event_fd_; }

    void reset() {
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

private:
    uint8_t* buf_;
    size_t capacity_;
    size_t mask_;

    // head_ and tail_ on separate cache lines to avoid false sharing.
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;

    int event_fd_ = -1;
};

} // namespace deusridet
