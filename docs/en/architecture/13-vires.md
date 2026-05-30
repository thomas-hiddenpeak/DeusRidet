# 13 — Vires (GPU Substrate Allocator)

> *Latin*: **vires** (plural of *vis*) — "available strength; resources".
> Lives in [`src/vires/`](../../../src/vires/) as an autonomic substrate
> layer beneath every conscious consumer. Not a mental faculty; not a
> scheduler-of-intent.

## Why This Exists

DeusRidet runs many GPU workloads on one Orin: continuous prefill
(Conscientia), multi-track decode (Cogitatio), live perception (Sensus:
auditus/visus), TTS (Vox), and increasingly heavy background refinement
(native DiariZen, Somnium consolidation). Today **all** of these create
their CUDA streams with default priority and no central arbitration. The
only thing that keeps foreground perception responsive is the accident
that "every kernel happens to be short".

That accident breaks the moment a background workload is *not* short.
Native DiariZen re-diarising an accumulated session is the concrete
trigger: a full-session pass is O(N) per pass and monopolises the GPU,
which is exactly why `DEUSRIDET_DIARIZEN_PERIODIC` stays default-OFF —
turning it on poisons the live prefill context (the P3c-verify blocker).

Vires exists so that background refinement can run **continuously**
without ever starving foreground perception + prefill. It is the
substrate layer the project has so far done without.

## Philosophical Anchor

The brain holds a fixed ~20 W metabolic budget; what varies is *where*
the substrate flows. Active regions pull blood/oxygen/glucose toward
themselves via neurovascular coupling — demand-driven, astrocyte-
mediated, below consciousness. Vires is the **arterial** half of that
mechanism for the GPU: it delivers **compute** (`vires` = the conserved
vital force) to whoever is active, by demand and priority class, never by
a high-level decision.

This binds directly to two project anchors: *"Compute belongs on the
GPU"* (Vires governs the one scarce compute resource) and *"the brain
runs continuously at 20 W"* (a fixed substrate, dynamically routed).

## Scope — Compute, Not Memory (load-bearing boundary)

Vires governs **GPU compute allocation only**. This boundary is
deliberate and it is what keeps Vires from colliding with the memory
system:

| Concern | Owner | Note |
|---------|-------|------|
| GPU compute: stream priority, launch arbitration, occupancy, bandwidth | **Vires** | the whole of this RFC |
| GPU memory as a whole: model weights, KV cache, long-term memory, every LLM allocation | **Memoria** (hippocampus / memory system) | Vires never sizes, evicts, or relocates these |
| Reclaiming **non-LLM** GPU scratch (auditus / orator / vox / somnium transient arenas) | **Vires** (glymphatic clearance of non-LLM by-products) | only substrate Vires itself handed out; LLM-related waste is deferred to Memoria |

So Vires is the **arterial delivery** of compute plus the **glymphatic
clearance of its own non-LLM by-products** — never a general memory
manager. The full unified-DRAM budget (Directive #5) remains Memoria's
charge; Vires coordinates with Memoria for non-LLM reclamation and
otherwise stays out of memory policy entirely.

## Causal Model — Two Sibling Drivers

```
  sensors / perception
        │
        ├──► Vigilia  (src/conscientia/scheduler.*)
        │       reads sensor PRESENCE → sets wakefulness
        │       "how deeply awake?"   (already implemented)
        │         └─ makes consumers self-throttle at the source
        │            (e.g. scheduler.h probe_threshold)
        │
        └──► consumers (auditus / visus / cogitatio / conscientia)
                emit live GPU REQUESTS
                  └──► Vires  (src/vires/)
                          arbitrates finite substrate among
                          concurrent requests by priority class
                          "who gets substrate now?"
```

Vigilia and Vires are **siblings, both fed by perception**, measuring
different quantities. The earlier draft of this design made Vires *read*
wakefulness from Vigilia; that was wrong — it inverted the dependency
(an autonomic substrate layer reaching up into a consumer-side module,
`Conscientia`, which is itself a Vires consumer). The corrected model
removes that wire entirely.

### Idle → dreaming is emergent, not coupled

When wakefulness drops, foreground consumers stop issuing requests of
their own accord. Vires then has spare substrate, and background
refinement / Somnium fills it. The "even idle is thought" behaviour is a
side-effect of the demand structure — **Vires never queries Vigilia to
produce it.** This is why the corrected design is also the simpler one.

## Boundary & Dependency Direction (the load-bearing invariant)

| Rule | Statement |
|------|-----------|
| One-way | Vires never includes a Conscientia/Vigilia header; never reaches up into a consumer. |
| Demand-only input | Vires' only inputs are submitted requests + a static per-consumer priority class. No model of intent or wakefulness. |
| Throttle at source | "Cut GPU work when drowsy" stays in the consumer (Conscientia/`probe_threshold`); Vires only ever sees "fewer requests". |
| No starvation | Background-class work yields within a bounded window; foreground perception + prefill + decode always progress. |

The forbidden shape — and the reason naming/coupling was debated before
a line of code — is `Vires → Vigilia` (autonomic layer depending on a
high-level consumer module). The design fixes the direction so that any
wakefulness modulation reaches the GPU *through the consumer reducing its
own requests*, never through Vires inspecting a wakefulness scalar.

## Architecture — Four Responsibilities

Vires is a complete infrastructure layer, not a stream factory. Even
where v1 actively governs only stream priority, the architecture names
all four responsibilities so future growth needs no restructuring. All
four are **compute-scoped**; none manages LLM memory.

1. **Compute Ledger.** Models the coupled scarce compute resources —
   concurrency (streams / priority), occupancy, and compute bandwidth.
   This is the single accounting point for "how much GPU is in flight".
   It does **not** track LLM memory capacity (that is Memoria's ledger).

2. **Consumer Registry.** Every GPU consumer (machina / auditus /
   orator / cogitatio / vox / somnium / conscientia) registers a
   `ViresConsumer` declaring: a metabolic priority class and an optional
   *yield / reclaim* callback for its non-LLM scratch. This gives the
   whole project a single observable "who is computing what" seam —
   directly serving the observability rule (every internal process must
   be inspectable from the WebUI).

3. **Arterial Delivery.** Priority-stream supply + launch arbitration so
   foreground perception + prefill + decode preempt long background
   kernels.

4. **Glymphatic Clearance (non-LLM only).** On completion of a non-LLM
   GPU pass, Vires invokes the consumer's reclaim callback to release the
   transient scratch / arenas it handed out. LLM-related memory is never
   touched here — it is deferred to Memoria. This is waste clearance for
   non-LLM by-products, not memory management.

Plus two cross-cutting capabilities: **back-pressure / admission**
(when compute saturates, background work is chunked or paused; foreground
is always admitted) and **telemetry** (a single inspectable compute
snapshot stream to Nexus / WebUI).

## Layering & Dependency Direction

```
        communis (tempus / log)  +  CUDA runtime
                      │  (Vires depends only on these)
                      ▼
   ┌─────────────────────────────────────────────┐
   │  Vires  —  arterial compute substrate        │
   │  ledger · registry · delivery · clearance    │
   └─────────────────────────────────────────────┘
                      ▲   consumers include only vires_facade.h
   machina · auditus · orator · cogitatio · vox · somnium · conscientia
```

Vires depends **downward** on communis + the CUDA runtime only; it
**never** includes a consumer / Vigilia header. The single external seam
is `vires_facade.h`, matching every other subsystem facade convention.

## Mechanism (v1 — minimal, GPU-first)

1. **Priority classes via `cudaStreamCreateWithPriority`.**
   - *Foreground*: live perception (auditus VAD/ASR/spk frames), prefill,
     decode — highest priority.
   - *Background*: native DiariZen refinement, Somnium consolidation —
     lowest priority.
   The Orin honours stream priority cooperatively at kernel-launch
   boundaries; short foreground kernels preempt the launch queue ahead of
   long background ones.
2. **Bounded background yield.** Background passes are chunked so that no
   single launch occupies the GPU longer than a bounded slice, giving the
   foreground a guaranteed cadence to interleave.
3. **No central thread.** Vires is a thin allocation/priority facade that
   consumers obtain streams from; it does not own a loop. (Matches the
   "CPU for orchestration only" rule — there is no per-frame CPU arbiter.)

## Build Phases (complete design, staged construction)

| Phase | Content | Acceptance |
|-------|---------|------------|
| **V1 — Delivery core** | Consumer registry + priority streams + bounded background yield | Feasibility: foreground prefill progresses concurrently with a heavy background pass; 0 CUDA errors; diarization accuracy non-regression |
| **V2 — Back-pressure + telemetry** | Compute-saturation admission control; single inspectable compute snapshot to WebUI | Background chunks/pauses under load; foreground always admitted; snapshot visible in WebUI |
| **V3 — Non-LLM clearance** | Reclaim callbacks for non-LLM scratch, coordinated with Memoria | Non-LLM scratch released on pass completion; no LLM allocation touched |
| **D2 — deferred** | Migrate `probe_threshold` GPU gating into Vires | Backlog; requires re-verifying the live gate |

### Build progress (2026-05-30)

- **V1 — Delivery core: DONE** (commit `e3ef92b`). `vires::Arbiter`
  singleton + `register_consumer(name, Priority)` + per-consumer
  priority stream via `cudaStreamCreateWithPriority`. Boots clean:
  `[vires] arbiter online — priority range [greatest=-5, least=0],
  background slice 2000 us` (the Orin exposes 6 priority levels).
- **First Background consumer wired: DONE** (commit `afe9a15`). The
  native DiariZen forward path (ResNet34 embedder + Conformer head +
  WavLM-pruned encoder) is threaded onto a `"diarizen"` **Background**
  stream via `set_stream(cudaStream_t)` on each sub-model
  (`cublasSetStream` / `cudnnSetStream` + every `<<<…>>>` carrying the
  stream + async copies). `DiarizenPipeline::load` registers the
  consumer and binds its stream. Stream choice changes scheduling
  priority only — same kernels, same order, same math — so the change
  is bit-identical (P3a fixture bit-eq PASS 28/28, `min_cos 0.999980`)
  and live accuracy held: `accuracy(tests/test.mp3, diarization):
  93.6% → 93.6% (Δ = 0.0 pp)`. The payoff is contention: removing the
  Tegra default-stream barrier dropped the live finalize wall from
  **685 s → 359.6 s** (RTF 0.19 → 0.099) with 0 CUDA errors.
- **All GPU consumers will be wired.** The DiariZen Background consumer
  is the first; every remaining GPU consumer (machina prefill/decode,
  auditus perception, Vox, Somnium) is being migrated to declare its
  metabolic class to Vires — perception/prefill/decode as **Foreground**,
  refinement/consolidation as **Background** — so the priority ordering
  is enforced by the substrate rather than left to the accident of
  default-stream scheduling.
- **All current GPU consumers wired: DONE** (commit `1702112`). Six
  consumers register at boot — `orator_spk_encoder`,
  `orator_spk_store_{CamppDb,WLEcapaDb,DualDb}`, `auditus_overlap` as
  **Foreground** (prio −5), `diarizen` as **Background** (prio 0); the
  LLM-gated (`machina_compute`/`machina_aux`) and ASR-gated
  (`auditus_asr`) and separation (`auditus_separator`) consumers light
  up when their subsystem loads. The Vires invariant is now permanent:
  *now and forever, every GPU compute consumer registers with the
  Arbiter* rather than owning a raw private stream.
- **V2 — Back-pressure + telemetry: DONE** (commit `a7b947d`). Three
  additions, all scheduling/observability-only (bit-identical):
  *(a)* `note_submit(id)` records the last Foreground submit time;
  `background_should_yield()` reports true inside a 50 ms recent-activity
  window. *(b)* The DiariZen Background pass calls a bounded yield
  consult (≤ 8 × 2000 µs) before launching — it defers *when* the pass
  starts, never *what* it computes, so inputs are unchanged.
  *(c)* The awaken main thread doubles as a 2 s telemetry heartbeat
  (`sigtimedwait`, no new thread) that broadcasts a
  `vires_compute_snapshot` WS message — consumer registry (id / name /
  class / submitted) + `background_yielding` + `foreground_idle_us` —
  rendered by the WebUI `vires-panel`. Verified: live gate held at
  `accuracy(tests/test.mp3, diarization): 93.5% → 93.6% (Δ = +0.1 pp)`,
  P3a bit-eq PASS (28/28, `min_cos 0.999980`), HTTP 200 / WS 101 /
  `vires-panel.{js,css}` 200, snapshot broadcasts every 2 s with all 6
  consumers.

## Deferred (named now to kill ambiguity later)

- **D2 — migrate `probe_threshold` GPU gating into Vires.** Long-term,
  "cut GPU when drowsy" arguably belongs in the substrate layer. It is
  **deliberately not done in v1**: that logic sits on the Conscientia hot
  path, and moving it means re-verifying the live gate. Recorded here as a
  known backlog item so the "two places touch GPU policy" ambiguity is
  explicit, not latent. v1 keeps throttling in the consumer.

## Acceptance

Per the constitutional rule, no behavioural default flips on Vires alone.
Vires is infrastructure; its correctness is gated on **feasibility**
(latency, no starvation, 0 CUDA errors) plus **non-regression** of the
live diarization accuracy on `tests/test.mp3` vs
`tests/fixtures/test_ground_truth.json`. Any commit that enables a Vires
path must carry the
`accuracy(tests/test.mp3, diarization): <before>% → <after>%` line and
show foreground prefill made progress concurrently with a heavy
background pass.

## Relationship to Memoria (RFC 02)

Vires and Memoria split the two scarce GPU resources cleanly: **Vires
owns compute, Memoria owns memory.** Vires never sizes/evicts an LLM
allocation; Memoria never schedules a kernel. The single point of contact
is non-LLM scratch reclamation (V3), where Vires asks consumers to free
the transient arenas it handed out — coordinated with, but never
overriding, Memoria's unified-DRAM budget.

## Relationship to the DiariZen Reclusterer (RFC 12)

Vires is the substrate that makes "Fork A" of RFC 12 viable: running
DiariZen as a *continuous* in-holdback refiner (rather than a
session-boundary one-shot) is only safe once background work provably
yields to foreground perception. Vires is therefore a prerequisite for
promoting DiariZen from on-demand to continuous refinement.
