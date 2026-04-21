# 07 — Persona (Inner / Outer Duality)

## Inner world

Divergent, rich, contradictory — the raw stream of consciousness. Visible
in Thinking and Daydream Decode branches (see `03-cogitatio.md`).

## Outer face

External expression dynamically shaped by the entity itself. The Arbiter
evaluates context, environment, and internal state to autonomously decide
how to present — diplomacy, directness, humor, silence, or anything in
between.

**This is not a hardcoded filter.** It is a learned, context-driven
decision process, no different from how humans modulate self-expression.

## Persona Config

`configs/persona.conf` specifies personality traits, speech style,
emotional tendencies, and baseline behavioral parameters that the entity
*adapts upon*, not *rigidly follows*.

## Implementation Surface

```
src/persona/
├── inner_world.{h,cpp}    # internal state representation
├── outer_face.{h,cpp}     # autonomous expression shaping
└── mapper.{h,cpp}         # inner → outer context-driven mapping
```

## Philosophical Notes

- Internal complexity is the prerequisite for external consistency.
  An entity that is rigidly the same inside and outside is not consistent,
  it is brittle.
- Allowing contradictions between inner and outer is not "lying" — it is
  the normal condition of any socially-aware mind. What would be
  pathological is an inner world that cannot diverge from outer expression.
