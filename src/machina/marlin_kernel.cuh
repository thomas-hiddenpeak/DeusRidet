/**
 * @file src/machina/marlin_kernel.cuh
 * @philosophical_role
 *   The Marlin INT4 GEMM kernel template body — pulled out of marlin.cu so
 *   dispatcher + repack sit under the R1 800-line cap without crossing TU
 *   boundaries that would hide the template from its callers.
 * @serves
 *   marlin.cu only. Included exactly once, mid-namespace, AFTER Section 2
 *   (PTX wrappers) and BEFORE Section 4 (dispatch).
 */
// marlin_kernel.cuh — private textual include, not part of the Machina
// public surface. NO namespace wrapping here; caller controls nesting.
//
// Adapted from IST-DASLab/marlin (Apache 2.0 License).
// Original: https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu
// Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)

#pragma once


template <
    const int threads,          // threads per block (256)
    const int thread_m_blocks,  // 16-row blocks in M dimension per threadblock
    const int thread_n_blocks,  // 16-col blocks in N dimension per threadblock
    const int thread_k_blocks,  // 16-element blocks in K dimension per threadblock
    const int stages,           // async pipeline stages (4)
    const int group_blocks = -1 // consecutive 16x16 blocks per scale group (-1 = per-column)
>
__global__ void Marlin(
    const int4* __restrict__ A,   // FP16 input [M, K]
    const int4* __restrict__ B,   // INT4 quantized weights [K/16, N*16/8] in Marlin format
          int4* __restrict__ C,   // FP16 output [M, N]
    const int4* __restrict__ s,   // FP16 scales [K/groupsize, N] Marlin-permuted
    int prob_m, int prob_n, int prob_k,
    int* locks,
    const int4* residual          // fused add: C[i] = residual[i] + result (nullptr = disabled)
) {
    // Local copy for parallel-batch offset tracking
    const int4* res_ptr = residual;

    // Striped partitioning: each threadblock processes one "stripe" of B.
    // Stripes ensure good utilization across all SMs with minimal global reductions.
    int parallel = 1;
    if (prob_m > 16 * thread_m_blocks) {
        parallel = prob_m / (16 * thread_m_blocks);
        prob_m = 16 * thread_m_blocks;
    }

    int k_tiles = prob_k / 16 / thread_k_blocks;
    int n_tiles = prob_n / 16 / thread_n_blocks;
    int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);

    // Ensure stripe boundaries align with group boundaries
    if (group_blocks != -1)
        iters = (group_blocks / thread_k_blocks) *
                ceildiv(iters, (group_blocks / thread_k_blocks));

    int slice_row = (iters * blockIdx.x) % k_tiles;
    int slice_col_par = (iters * blockIdx.x) / k_tiles;
    int slice_col = slice_col_par;
    int slice_iters;
    int slice_count = 0;
    int slice_idx;

    // Handle parallel batch problem instances
    if (slice_col_par >= n_tiles) {
        int par_idx = slice_col_par / n_tiles;
        A += par_idx * 16 * thread_m_blocks * prob_k / 8;
        C += par_idx * 16 * thread_m_blocks * prob_n / 8;
        if (res_ptr) res_ptr += par_idx * 16 * thread_m_blocks * prob_n / 8;
        locks += par_idx * n_tiles;
        slice_col = slice_col_par % n_tiles;
    }

    // Compute slice metadata for synchronization
    auto init_slice = [&]() {
        slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
        if (slice_iters < 0 || slice_col_par >= n_tiles * parallel)
            slice_iters = 0;
        if (slice_iters == 0) return;
        if (slice_row + slice_iters > k_tiles)
            slice_iters = k_tiles - slice_row;
        slice_count = 1;
        slice_idx = 0;
        int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
        if (col_first <= k_tiles * (slice_col_par + 1)) {
            int col_off = col_first - k_tiles * slice_col_par;
            slice_count = ceildiv(k_tiles - col_off, iters);
            if (col_off > 0) slice_count++;
            int delta_first = iters * blockIdx.x - col_first;
            if (delta_first < 0 || (col_off == 0 && delta_first == 0))
                slice_idx = slice_count - 1;
            else {
                slice_idx = slice_count - 1 - delta_first / iters;
                if (col_off > 0) slice_idx--;
            }
        }
        if (slice_col == n_tiles) {
            A += 16 * thread_m_blocks * prob_k / 8;
            C += 16 * thread_m_blocks * prob_n / 8;
            if (res_ptr) res_ptr += 16 * thread_m_blocks * prob_n / 8;
            locks += n_tiles;
            slice_col = 0;
        }
    };
    init_slice();

    // Stride calculations (all in int4 = 16-byte units)
    int a_gl_stride = prob_k / 8;
    constexpr int a_sh_stride = 16 * thread_k_blocks / 8;
    constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;
    int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
    constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
    constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));
    constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
    constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);
    constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta);

    int b_gl_stride = 16 * prob_n / 32;
    constexpr int b_sh_stride = 32 * thread_n_blocks / 4;
    int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
    int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
    constexpr int b_sh_wr_delta = threads;
    constexpr int b_sh_rd_delta = threads;
    constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
    constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

    int s_gl_stride = prob_n / 8;
    constexpr int s_sh_stride = 16 * thread_n_blocks / 8;
    constexpr int s_sh_stage = s_sh_stride;
    int s_gl_rd_delta = s_gl_stride;

    // Thread-level read/write indices
    int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o);
    a_gl_rd += a_gl_rd_delta_o * slice_row;
    int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o);
    int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16) +
                  (threadIdx.x % 32) / 16;
    a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

    int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) +
                  (threadIdx.x % b_sh_stride);
    b_gl_rd += b_sh_stride * slice_col;
    b_gl_rd += b_gl_rd_delta_o * slice_row;
    int b_sh_wr = threadIdx.x;
    int b_sh_rd = threadIdx.x;

    int s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                  s_sh_stride * slice_col + threadIdx.x;
    int s_sh_wr = threadIdx.x;
    int s_sh_rd;
    if (group_blocks != -1)
        s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
                  (threadIdx.x % 32) / 4;
    else
        s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
                  (threadIdx.x % 32) % 4;

    // Predication for boundary handling
    bool a_sh_wr_pred[a_sh_wr_iters];
    #pragma unroll
    for (int i = 0; i < a_sh_wr_iters; i++)
        a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
    bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

    // XOR-based SMEM layout for fully bank-conflict-free A tile access
    auto transform_a = [&](int i) {
        int row = i / a_gl_rd_delta_o;
        return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
    };

    // Precompute transformed SMEM indices (all accesses are static after unrolling)
    int a_sh_wr_trans[a_sh_wr_iters];
    #pragma unroll
    for (int i = 0; i < a_sh_wr_iters; i++)
        a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
    int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
    #pragma unroll
    for (int i = 0; i < b_sh_wr_iters; i++) {
        #pragma unroll
        for (int j = 0; j < thread_m_blocks; j++)
            a_sh_rd_trans[i][j] = transform_a(
                a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
    }

    // Pre-split B pointers to break dependency chains between iterations
    const int4* B_ptr[b_sh_wr_iters];
    #pragma unroll
    for (int i = 0; i < b_sh_wr_iters; i++)
        B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

    // Dynamic shared memory: A tiles | B tiles | S tiles
    extern __shared__ int4 sh[];
    int4* sh_a = sh;
    int4* sh_b = sh_a + (stages * a_sh_stage);
    int4* sh_s = sh_b + (stages * b_sh_stage);

    // Register double-buffer for SMEM→register loads
    FragA frag_a[2][thread_m_blocks];
    I4 frag_b_quant[2];
    FragC frag_c[thread_m_blocks][4][2];
    FragS frag_s[2][4];

    // Zero accumulators
    auto zero_accums = [&]() {
        #pragma unroll
        for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
            reinterpret_cast<float*>(frag_c)[i] = 0;
    };

    // Async fetch next A, B, s tiles from global to shared memory pipeline stage
    auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
        if (pred) {
            int4* sh_a_stage = sh_a + a_sh_stage * pipe;
            #pragma unroll
            for (int i = 0; i < a_sh_wr_iters; i++) {
                cp_async4_pred(
                    &sh_a_stage[a_sh_wr_trans[i]],
                    &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
                    a_sh_wr_pred[i]);
            }
            int4* sh_b_stage = sh_b + b_sh_stage * pipe;
            #pragma unroll
            for (int i = 0; i < b_sh_wr_iters; i++) {
                cp_async4_stream(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
                B_ptr[i] += b_gl_rd_delta_o;
            }
            if (group_blocks != -1 &&
                pipe % (group_blocks / thread_k_blocks) == 0)
            {
                int4* sh_s_stage = sh_s + s_sh_stage * pipe;
                if (s_sh_wr_pred)
                    cp_async4_stream(&sh_s_stage[s_sh_wr], &s[s_gl_rd]);
                s_gl_rd += s_gl_rd_delta;
            }
        }
        cp_async_fence();
    };

    // Wait for next pipeline stage to be ready in shared memory
    auto wait_for_stage = [&]() {
        cp_async_wait<stages - 2>();
        __syncthreads();
    };

    // Load sub-tile from shared memory into register double-buffer
    auto fetch_to_registers = [&](int k, int pipe) {
        if (group_blocks != -1) {
            int4* sh_s_stage = sh_s + s_sh_stage *
                ((group_blocks / thread_k_blocks) *
                 (pipe / (group_blocks / thread_k_blocks)));
            reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
        }
        int4* sh_a_stage = sh_a + a_sh_stage * pipe;
        #pragma unroll
        for (int i = 0; i < thread_m_blocks; i++)
            ldsm4(frag_a[k % 2][i],
                   &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
        int4* sh_b_stage = sh_b + b_sh_stage * pipe;
        frag_b_quant[k % 2] = *reinterpret_cast<I4*>(
            &sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
    };

    // Execute tensor core matmul for one sub-tile
    auto matmul = [&](int k) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int b_quant = frag_b_quant[k % 2][j];
            int b_quant_shift = b_quant >> 8;
            FragB frag_b0 = dequant(b_quant);
            if (group_blocks != -1)
                scale(frag_b0, frag_s[k % 2][j], 0);
            FragB frag_b1 = dequant(b_quant_shift);
            if (group_blocks != -1)
                scale(frag_b1, frag_s[k % 2][j], 1);
            #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
                mma(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
                mma(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
            }
        }
    };

    // Thread-block level reduction (multiple warps → single result per output position)
    auto thread_block_reduce = [&]() {
        constexpr int red_off = threads / b_sh_stride / 2;
        if (red_off >= 1) {
            int red_idx = threadIdx.x / b_sh_stride;
            constexpr int red_sh_stride = b_sh_stride * 4 * 2;
            constexpr int red_sh_delta = b_sh_stride;
            int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) +
                            (threadIdx.x % b_sh_stride);

            #pragma unroll
            for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
                #pragma unroll
                for (int i = red_off; i > 0; i /= 2) {
                    if (i <= red_idx && red_idx < 2 * i) {
                        #pragma unroll
                        for (int j = 0; j < 4 * 2; j++) {
                            int red_sh_wr = red_sh_delta * j +
                                            (red_sh_rd - red_sh_stride * i);
                            if (i < red_off) {
                                float* c_rd = reinterpret_cast<float*>(
                                    &sh[red_sh_delta * j + red_sh_rd]);
                                float* c_wr = reinterpret_cast<float*>(
                                    &sh[red_sh_wr]);
                                #pragma unroll
                                for (int k = 0; k < 4; k++)
                                    reinterpret_cast<FragC*>(frag_c)
                                        [4 * 2 * m_block + j][k] +=
                                        c_rd[k] + c_wr[k];
                            }
                            sh[red_sh_wr] = reinterpret_cast<int4*>(
                                &frag_c)[4 * 2 * m_block + j];
                        }
                    }
                    __syncthreads();
                }
                if (red_idx == 0) {
                    #pragma unroll
                    for (int i = 0; i < 4 * 2; i++) {
                        float* c_rd = reinterpret_cast<float*>(
                            &sh[red_sh_delta * i + red_sh_rd]);
                        #pragma unroll
                        for (int j = 0; j < 4; j++)
                            reinterpret_cast<FragC*>(frag_c)
                                [4 * 2 * m_block + i][j] += c_rd[j];
                    }
                }
                __syncthreads();
            }
        }
    };

    // Global reduction across threadblocks in the same column slice (via L2 cache)
    auto global_reduce = [&](bool first = false, bool last = false) {
        constexpr int active_threads = 32 * thread_n_blocks / 4;
        if (threadIdx.x < active_threads) {
            int c_gl_stride = prob_n / 8;
            int c_gl_wr_delta_o = 8 * c_gl_stride;
            int c_gl_wr_delta_i = 4 * (active_threads / 32);
            int c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) +
                          4 * (threadIdx.x / 32) + threadIdx.x % 4;
            c_gl_wr += (2 * thread_n_blocks) * slice_col;
            constexpr int c_sh_wr_delta = active_threads;
            int c_sh_wr = threadIdx.x;
            int row = (threadIdx.x % 32) / 4;

            if (!first) {
                #pragma unroll
                for (int i = 0; i < thread_m_blocks * 4; i++) {
                    cp_async4_pred(
                        &sh[c_sh_wr + c_sh_wr_delta * i],
                        &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                           c_gl_wr_delta_i * (i % 2)],
                        i < (thread_m_blocks - 1) * 4 ||
                            8 * (i / 2) + row < prob_m);
                }
                cp_async_fence();
                cp_async_wait<0>();
            }

            #pragma unroll
            for (int i = 0; i < thread_m_blocks * 4; i++) {
                if (i < (thread_m_blocks - 1) * 4 ||
                    8 * (i / 2) + row < prob_m)
                {
                    if (!first) {
                        int4 c_red = sh[c_sh_wr + i * c_sh_wr_delta];
                        #pragma unroll
                        for (int j = 0; j < 2 * 4; j++) {
                            reinterpret_cast<float*>(frag_c)
                                [4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)] +=
                                __half2float(
                                    reinterpret_cast<__half*>(&c_red)[j]);
                        }
                    }
                    if (!last) {
                        int4 c;
                        #pragma unroll
                        for (int j = 0; j < 2 * 4; j++) {
                            reinterpret_cast<__half*>(&c)[j] = __float2half(
                                reinterpret_cast<float*>(frag_c)
                                    [4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)]);
                        }
                        C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                          c_gl_wr_delta_i * (i % 2)] = c;
                    }
                }
            }
        }
    };

    // Write final result, reshuffling from fragment layout to row-major
    auto write_result = [&]() {
        int c_gl_stride = prob_n / 8;
        constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
        int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
        constexpr int c_sh_rd_delta =
            c_sh_stride * (threads / (2 * thread_n_blocks));

        int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                      (threadIdx.x % (2 * thread_n_blocks));
        c_gl_wr += (2 * thread_n_blocks) * slice_col;
        int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) +
                      (threadIdx.x % 32) % 4;
        c_sh_wr += 32 * (threadIdx.x / 32);
        int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                      (threadIdx.x % (2 * thread_n_blocks));

        int c_gl_wr_end = c_gl_stride * prob_m;

        auto write = [&](int idx, float c0, float c1, FragS& s) {
            half2 res = __halves2half2(__float2half(c0), __float2half(c1));
            if (group_blocks == -1)  // per-column scale applied here
                res = __hmul2(res, s[0]);
            ((half2*)sh)[idx] = res;
        };

        if (threadIdx.x / 32 < thread_n_blocks / 4) {
            #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int wr = c_sh_wr + 8 * j;
                    write(wr + (4 * c_sh_stride) * 0 + 0,
                          frag_c[i][j][0][0], frag_c[i][j][0][1],
                          frag_s[j / 2][2 * (j % 2) + 0]);
                    write(wr + (4 * c_sh_stride) * 8 + 0,
                          frag_c[i][j][0][2], frag_c[i][j][0][3],
                          frag_s[j / 2][2 * (j % 2) + 0]);
                    write(wr + (4 * c_sh_stride) * 0 + 4,
                          frag_c[i][j][1][0], frag_c[i][j][1][1],
                          frag_s[j / 2][2 * (j % 2) + 1]);
                    write(wr + (4 * c_sh_stride) * 8 + 4,
                          frag_c[i][j][1][2], frag_c[i][j][1][3],
                          frag_s[j / 2][2 * (j % 2) + 1]);
                }
                c_sh_wr += 16 * (4 * c_sh_stride);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0;
             i < ceildiv(16 * thread_m_blocks,
                         threads / (2 * thread_n_blocks));
             i++)
        {
            if (c_gl_wr < c_gl_wr_end) {
                if (res_ptr) {
                    // Fused residual add: C[i] = residual[i] + gemm_result[i]
                    int4 r = sh[c_sh_rd];
                    int4 o = res_ptr[c_gl_wr];
                    half2* rh = reinterpret_cast<half2*>(&r);
                    const half2* oh = reinterpret_cast<const half2*>(&o);
                    rh[0] = __hadd2(rh[0], oh[0]);
                    rh[1] = __hadd2(rh[1], oh[1]);
                    rh[2] = __hadd2(rh[2], oh[2]);
                    rh[3] = __hadd2(rh[3], oh[3]);
                    C[c_gl_wr] = r;
                } else {
                    C[c_gl_wr] = sh[c_sh_rd];
                }
                c_gl_wr += c_gl_wr_delta;
                c_sh_rd += c_sh_rd_delta;
            }
        }
    };

    // ---- Main execution pipeline ----

    auto start_pipes = [&]() {
        #pragma unroll
        for (int i = 0; i < stages - 1; i++)
            fetch_to_shared(i, i, i < slice_iters);
        zero_accums();
        wait_for_stage();
        fetch_to_registers(0, 0);
        a_gl_rd += a_gl_rd_delta_o * (stages - 1);
    };
    start_pipes();

    // Main loop: interleaved fetch + compute
    while (slice_iters) {
        #pragma unroll
        for (int pipe = 0; pipe < stages;) {
            #pragma unroll
            for (int k = 0; k < b_sh_wr_iters; k++) {
                fetch_to_registers(k + 1, pipe % stages);
                if (k == b_sh_wr_iters - 2) {
                    fetch_to_shared((pipe + stages - 1) % stages,
                                    pipe, slice_iters >= stages);
                    pipe++;
                    wait_for_stage();
                }
                matmul(k);
            }
            slice_iters--;
            if (slice_iters == 0) break;
        }
        a_gl_rd += a_gl_rd_delta_o * stages;

        // End-of-slice processing
        if (slice_iters == 0) {
            cp_async_wait<0>();
            bool last = slice_idx == slice_count - 1;

            if (group_blocks == -1 && last) {
                if (s_sh_wr_pred)
                    cp_async4_stream(&sh_s[s_sh_wr], &s[s_gl_rd]);
                cp_async_fence();
            }
            thread_block_reduce();
            if (group_blocks == -1 && last) {
                cp_async_wait<0>();
                __syncthreads();
                if (threadIdx.x / 32 < thread_n_blocks / 4) {
                    reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
                    reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
                }
            }

            if (slice_count > 1) {
                barrier_acquire(&locks[slice_col], slice_idx);
                global_reduce(slice_idx == 0, last);
                barrier_release(&locks[slice_col], last);
            }
            if (last)
                write_result();

            slice_row = 0;
            slice_col_par++;
            slice_col++;
            init_slice();
            if (slice_iters) {
                a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                          (threadIdx.x % a_gl_rd_delta_o);
                #pragma unroll
                for (int i = 0; i < b_sh_wr_iters; i++)
                    B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
                if (slice_col == 0) {
                    #pragma unroll
                    for (int i = 0; i < b_sh_wr_iters; i++)
                        B_ptr[i] -= b_gl_stride;
                }
                s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
                start_pipes();
            }
        }
    }
}
