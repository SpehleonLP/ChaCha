# CLAUDE.md

## Project Overview

ChaCha is a standalone library that deduces joint articulation constraints from skeletal animation data. Given a set of animation channels (rotation/translation/scale keyframes per joint), it produces per-joint constraint descriptors: ranges of motion, observed velocity, and observed peak acceleration for each degree of freedom. (Effort/torque limits are out of scope -- see Limitations.)

The output format follows the [AGI_articulations](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Vendor/AGI_articulations) glTF extension schema, but the library itself has **no dependency on any glTF parser**. It operates on pre-extracted animation data and produces structured constraint data. Bridge code to read/write glTF files lives in the consuming application (e.g., tonton-example), not in ChaCha.

This is the same separation pattern used by its sister modules:
- **tonton**: biomechanical analysis from skeletons — no file I/O, takes pre-computed volumetric data
- **rintintin**: volumetric analysis from meshes — no file I/O, takes raw vertex/index data
- **dodeedum**: 2D silhouette projections — no file I/O, takes mesh primitives

## What Problem Does This Solve?

The AGI_articulations glTF extension describes joint constraints (min/max rotation, translation limits, velocity caps), but:
1. No authoring software generates it
2. Nobody uses it
3. The information is *already implicit* in animation data — if a joint rotates ±45° across all animations, that's its range of motion

ChaCha makes this implicit information explicit by analyzing animation keyframes.

## Architecture

### Input (provided by caller)

```cpp
// One animation clip, referenced by AnimationChannel::animation. `name` is a
// non-owning view: the caller must keep the backing storage alive for the
// duration of the analyze() call (and beyond, if the caller inspects it
// afterward) -- a bridge layer that builds names from a temporary (e.g. a
// glTF JSON string that goes out of scope) creates a use-after-free that
// nothing in this library can detect. Used to auto-detect "AGI "-prefixed
// configuration animations.
struct Animation {
    std::string_view name;
};

// Per animation channel: one joint, one property, one set of keyframes
struct AnimationChannel {
    int node;                    // Joint/node index in skeleton
    int animation;                // Index into the analyze() animations span
    Property property;           // Translation, Rotation, Scale
    InterpolationType interp;    // Linear, Step, CubicSpline
    std::span<const float> times;       // Keyframe timestamps (seconds)
    std::span<const float> values;      // Keyframe values (3 or 4 floats per key)
};

// Skeleton hierarchy (parents array + rest poses)
struct Skeleton {
    std::span<const int> parents;
    std::span<const glm::quat> rest_rotations;   // Rest pose per joint; must be sized >= parents.size()
    std::span<const glm::vec3> rest_translations; // Must be sized >= parents.size()
    std::span<const glm::vec3> rest_scales;       // May be EMPTY (means "assume all-ones" for every node) or sized
                                                   // >= parents.size() -- but SHORT (non-empty, under that floor) is
                                                   // fatal: analyze() returns an empty result with NO diagnostic.
                                                   // Short is the one thing this span may never be.
};
```

### Output

```cpp
// Per degree of freedom for a joint
struct Stage {
    StageType type;          // xRotate, yRotate, zRotate, xTranslate, ..., xScale, ...
    float min_value;         // Lower bound (radians, meters, or a bare multiplicative ratio for scale)
    float max_value;         // Upper bound
    float initial_value;     // Rest/neutral value (0 for rotation/translation, 1 for scale)
    float max_velocity;      // Observed speed (rad/s, m/s, or ratio/s)
    float max_acceleration;  // Observed peak acceleration. Was `max_effort`: force/torque requires
                              // mass and inertia this library cannot derive from keyframes alone,
                              // so that field was replaced with what keyframes actually support.
};

// All constraints for one joint
struct Articulation {
    int node;
    std::string name;                // ALWAYS EMPTY: analyze() never writes this (ChaCha has no node
                                      // names to draw from). Callers derive a name from their own
                                      // data (e.g. doc.nodes[node].name) and pass it through
                                      // ChaCha::sanitize_articulation_name (chacha_naming.h), which
                                      // is public exactly so consumers can do this and get AGI's
                                      // whitespace/uniqueness rules for free.
    std::vector<Stage> stages;       // Ordered sequence, NOT a set keyed by type -- see below
    glm::vec3 pointing_vector;       // Model-wide inferred "forward" axis for this rig
    uint8_t dof_count;               // ROTATIONAL DOF the solver committed to (0-3), NOT stages.size():
                                      // e.g. on sophia, 26 articulations have 6 stages (3 rotation +
                                      // 3 translation/scale) but dof_count == 3. Stays 0 for joints
                                      // with translation/scale stages only (no rotation solve ran).
    float fit_residual_rad;          // Worst-case round-trip residual for the committed rotation
                                      // decomposition. Only meaningful when dof_count > 0; stays 0
                                      // for translation-only articulations, same as dof_count.
};
```

**Stages are an ordered sequence, not a set.** A `StageType` can legitimately repeat within one
articulation's `stages` (see "General multi-axis rotation" below) -- consumers must iterate in
order and disambiguate repeats by occurrence index, e.g. via `stage_name_for(type, occurrence)`
in `chacha_naming.h`, never by keying or deduplicating on `StageType` alone.

### Processing Pipeline

```
Animation Channels (per joint)
  │
  ├─ 1. Resolve the scan
  │     `scan` empty means "all animations". If options.prioritize_rom_animations
  │     is set and any animation's name begins with "AGI " (case-insensitive),
  │     the scan narrows to just those -- an artist-authored configuration
  │     animation is treated as a direct specification. (chacha_analyzer.cpp,
  │     resolve_scan)
  │
  ├─ 2. Convert keyframes to rest-pose-relative values
  │     Rotation: subtract rest rotation (quaternion). Translation/scale:
  │     subtract/divide by rest translation/scale. Scale is a MULTIPLICATIVE
  │     ratio (rest 1.0 -> 2.0 is [1, 2], never [0, 1]), never an additive delta.
  │
  ├─ 3. Per joint, per DOF-family: solve for the best decomposition
  │     Translation and scale project directly onto the three rest-relative
  │     axes (chacha_summary.cpp's `summarise`).
  │
  │     Rotation is harder: a joint's rest-relative quaternion trajectory may
  │     be driven by one axis, two axes, or genuinely three. Two paths:
  │
  │     a) Configuration fast path (chacha_config.cpp): if the scan narrowed to
  │        "AGI " animations AND the motion has configuration shape (exercises
  │        one axis at a time, returning to rest between phases), the phase
  │        order IS the answer -- no search needed, since the axes commute and
  │        a search has no way to recover authored order from commuting axes.
  │
  │     b) General search (chacha_search.cpp, select_candidate): otherwise,
  │        evaluate every 1-axis (3), 2-axis (6, ordered pairs), and 3-axis
  │        (12 Euler charts -- 6 Tait-Bryan + 6 proper-Euler; chacha_charts.cpp)
  │        decomposition and pick the best by dof (fewest wins, subject to a
  │        residual acceptance gate for 1-/2-axis candidates -- chacha_reduced.cpp),
  │        then tightest total range, then best conditioning. The 3-axis charts
  │        always reconstruct exactly (Euler decomposition is always exact,
  │        chacha_charts.cpp's `solve_euler`); 1-/2-axis candidates are
  │        least-squares fits admitted only if they explain every observed pose
  │        within `options.max_fit_residual_rad`.
  │
  │     Per-frame branch ambiguity in the 3-axis charts (each Euler solve has
  │     two solution branches) is resolved by a Viterbi/DP shortest path over
  │     consecutive frames (chacha_dp.cpp, `resolve_branches`), then a global
  │     tie-break toward whichever branch stays closer to rest.
  │
  │     Multiple animations of the same joint are combined by anchoring each
  │     one's trajectory against a common per-axis reference before taking the
  │     union of ranges -- otherwise two animations straddling a +-pi wrap
  │     boundary from opposite sides union to a spurious ~2pi span
  │     (`anchor_trajectory`). The final range is then shifted by whichever
  │     multiple of 2pi brings its midpoint nearest zero, so the reported
  │     absolute values don't depend on which animation was processed first.
  │
  ├─ 4. Compute velocity and acceleration
  │     Resampled onto a uniform time grid (options.resample_rate_hz) before
  │     differentiating, so estimates don't depend on keyframe density
  │     (chacha_summary.cpp). Values reported are OBSERVED, not physical
  │     limits -- see Limitations, especially for the configuration path.
  │
  ├─ 5. Filter noise
  │     Discard stages with negligible range (< threshold). Default
  │     thresholds: 0.01 rad for rotation, 0.001 m for translation, 0.01 for
  │     scale (chacha_filter.cpp).
  │
  ├─ 6. Infer a model-wide pointing vector
  │     A single "forward" axis vote shared across all articulations in the
  │     result, not a per-joint value (chacha_pointing.cpp), with a per-joint
  │     fallback when no axis clears the dominant-axis fraction.
  │
  └─ 7. Produce Articulation per joint
        Stage order is a fixed, deliberate contract: translate, then scale,
        then rotate, regardless of the order channels arrived in. Joints with
        no significant motion in the scanned animations get no articulation
        (locked). A StageType can legitimately repeat (proper-Euler charts
        emit the same axis twice) -- see the Output section above.
```

### Animation Cleaning (Optional)

A separate `clean()` API removes redundant keyframes from animation channels:
- Redundant keyframe removal (Step, Linear, CubicSpline)
- Optional interpolation type promotion (Step→Linear, Linear→CubicSpline) when `frame_time` is set

See `include/chacha_clean.h` for the `CleanOptions` and `CleanedChannel` types.

### What ChaCha Does NOT Do

- **No file I/O.** It doesn't read glTF files, JSON, or binary formats. The caller extracts animation channels and passes them in.
- **No collision detection.** Capsule colliders and self-intersection testing are a separate concern (likely lives in rintintin or a glue layer).
- **No IK solving.** The CCD solver for computing crouch length uses ChaCha's output but is not part of ChaCha.
- **No rendering.** No GPU, no shaders, no display.

## Design Decisions

### General multi-axis rotation decomposition (Step 3)

**Chosen approach: exhaustive 1/2/3-axis candidate search, not swing-twist.** An earlier
design used swing-twist decomposition (twist around bone-local Y, swing on X/Z). It was
replaced because swing-twist's axis assignment is fixed by convention rather than by
evidence: it cannot recognise when a joint's real motion is actually 1-axis or 2-axis (it
always reports a swing pair even for a pure hinge), and it has no way to decide *which* axis
order is correct for a genuinely 3-axis joint. The current approach (`chacha_search.cpp`,
`select_candidate`) instead evaluates every plausible decomposition -- 3 single-axis fits, 6
ordered two-axis fits, and all 12 three-axis Euler charts (6 Tait-Bryan + 6 proper-Euler,
`chacha_charts.cpp`) -- against the observed keyframes and picks the one that actually fits
best, honouring a strict residual gate for the reduced (1-/2-axis) candidates so a fit is only
accepted when it genuinely explains the motion, not just approximates it.

The 12 charts cover every triple of rotation axes (both distinct-axis Tait-Bryan sequences
and repeated-axis proper-Euler sequences like Z-X-Z). Each chart's Euler solve
(`solve_euler`) is exact and closed-form (Bernardes & Viollet, PLoS ONE 2022, generalised to
all 12 sequences); the two solution branches at each frame are disambiguated by a Viterbi/DP
shortest path across frames (`chacha_dp.cpp`) rather than chosen independently per frame,
since an independent per-frame choice can flip branches mid-motion and fabricate a
discontinuous trajectory.

**A separate fast path exists for artist-authored specifications** (`chacha_config.cpp`):
when the scan has narrowed to "AGI "-prefixed animations and the motion visibly exercises
one axis at a time between returns to rest, the phase order recovered directly from that
shape is used as-is, bypassing the search. This is necessary, not just an optimisation: axes
exercised one at a time compose commutatively, so no amount of searching can recover the
artist's intended stage *order* from the resulting quaternion alone -- only the shape of the
motion (one axis at a time) can.

Alternatives considered and rejected:
- Swing-twist decomposition: see above -- fixed axis assignment, no way to detect or prefer
  a reduced-DOF fit, no way to determine multi-axis stage order.
- PCA on rotation vectors: axes don't align with intuitive anatomical directions, and still
  provides no mechanism for recovering stage order on commuting axes.

### Limitations

- ChaCha reports *observed* ROM, not *possible* ROM. If no animation fully extends a joint,
  the range is underestimated. Callers can add safety margins or provide a dedicated ROM
  animation.
- When the scan narrows to `AGI ` configuration animations, velocity and acceleration reflect
  the sweep speed the artist authored, which is usually a uniform ramp rather than a real
  speed limit. Ranges and stage order are the trustworthy outputs in that mode.
- Effort and torque limits are out of scope: they require mass and inertia and depend on the
  motion of the whole subtree.
- A trajectory that crosses the singular set of all 12 charts cannot be represented by three
  stages. A redundant fourth stage would be required; this is not implemented.
- **The reduced-DOF acceptance gate is a WORST CASE over every frame of every scanned
  animation**, not an average or a per-clip check: a single frame that doesn't fit the
  candidate axis disqualifies it, no matter how clean the rest of that joint's motion is.
  On the 88-clip Mixamo rig, every one of sophia's 40 joints resolves to full 3-DOF
  (`0/0/0/40` histogram); the left knee's best 1-DOF residual is 0.836 rad and the left
  elbow's is 0.741 rad, both against a 0.02 rad gate. This is NOT the gate being too
  strict -- it is the correct answer for this corpus. Mixamo's hip does not actually
  twist, so retargeting pushes that rotation onto downstream joints: the knee genuinely
  moves through three independent axes in this data, behaving like a ball-and-socket
  rather than the anatomical hinge it is at rest. Reporting it as 3-DOF is accurate;
  classifying it as a 1-DOF hinge would be false, because ChaCha's contract is *observed*
  range of motion (see the first Limitations bullet above), not the joint's anatomically
  possible range. Loosening the gate to a percentile or RMS criterion instead of a
  worst-case would produce a tidier-looking 1-DOF classification here, but only by
  discarding motion the joint actually undergoes in the corpus -- do not do this. The
  worst-case-over-every-frame criterion is what makes the emitted range/stage bounds a
  true superset of all observed motion; an independent coverage check across all three
  corpus models (sophia, scorpion, treefrog) confirmed 0 samples fall outside their
  emitted bounds out of 116,803 checked, and a percentile/RMS gate would break that
  property in exchange for a more legible DOF count. The reduced-DOF path itself is not
  dead: it fires on real models through the configuration fast path (scorpion emits 11
  one-DOF and 16 two-DOF articulations; treefrog 41 and 7) and is covered end-to-end by
  synthetic unit tests. It simply does not fire via the general search path
  (`select_candidate`'s residual gate) on multi-clip retargeted mocap in this corpus,
  for the anatomically sound reason above -- see the test-coverage note below for what
  that does and doesn't mean for confidence in the gate itself.
- **Velocity and acceleration from an `AGI ` configuration animation reflect the artist's
  authored sweep speed, not a physical limit**, and the configuration path computes
  acceleration per phase rather than across phase boundaries, which yields systematically
  smaller values than the search path for the same input (measured: a 3-phase sinusoid gives
  3.61 rad/s² via the configuration path versus ~13.05 rad/s² via the search path for the
  same input). Ranges and stage order are the trustworthy outputs in that mode.
- **A stage type can legitimately repeat within one articulation.** Six of the twelve
  rotation charts are proper-Euler with a repeated axis, so roughly 31% of general-motion
  three-DOF joints emit e.g. `zRotate` in both the first and third stage. Stages are an
  ordered sequence applied in order of appearance; consumers must never key or deduplicate
  them by type, and must disambiguate repeats by occurrence index via `stage_name_for`.

**Test coverage is honest, not flattering, about the search path.** `select_candidate`'s
reduced-DOF search and residual acceptance gate have no real-data integration coverage in
this repository's own corpus of glTF models: running the whole integration test suite
against a build where that gate can never accept a reduced candidate still passes every
test. It IS covered by this library's own synthetic unit tests -- the same ablation fails
several of those -- and that is the appropriate place for it, but no available real-world
model happens to exercise it end-to-end. Likewise, no corpus model produces a scale stage or
a repeated stage type; those paths are covered synthetically only.

`Animation::name` is a `std::string_view`: the caller owns the backing storage and must keep
it alive for the duration of the `analyze()` call (and beyond, if it inspects the name
afterward). Both the glTF read bridge and the write bridge in a consuming application must
get this right -- a bridge that builds a name from a temporary creates a use-after-free that
nothing in this library can detect.

## Dependencies

- **GLM** (vec3, quat, mat4 — same as tonton/rintintin)
- **C++20** (spans, concepts)
- **No other dependencies.** No JSON parsers, no file I/O libraries, no physics engines.

## Build System

```bash
mkdir build && cd build
cmake ..
make
```

CMake 3.15+, static library output.

## Integration with tonton-example

In tonton-example, a bridge header provides:
- `chacha_fxgltf_bridge.h`: Extracts `AnimationChannel` data from `fx::gltf::Document`
- `chacha_fxgltf_bridge.cpp`: Writes `Articulation` output back as AGI_articulations extension

The example program (binary name `chacha-example`, see `tonton-example/CMakeLists.txt`):
```
chacha-example input.glb [output.glb]
```
Return codes (see `tonton-example/src/chacha_main.cpp`; each of 2 and 4 covers two
distinct conditions, not one):
- `0` — Success: AGI articulations generated and written
- `1` — File already has AGI_articulations (skip, or overwrite with `--force`)
- `2` — Nothing to analyze: either the document has no animations at all, or it has
  animations but none produced a usable channel (a channel must target rotation,
  translation, or scale)
- `3` — Analysis ran but produced no articulations: every observed motion, across
  every animation and joint, fell below the noise threshold (`--rotation-threshold`
  / `--translation-threshold`). There is no "ambiguous decomposition" exit path in
  the current solver -- `select_candidate` (chacha_search.cpp) always picks a best
  candidate; it never refuses to disambiguate.
- `4` — I/O or argument failure: reading the input file, parsing command-line
  arguments, or writing the output file all funnel into this same code

## Downstream Consumers

ChaCha's output feeds into:
1. **tonton builder** — `crouch_length` on `Builder_Chain` (currently a placeholder estimate)
2. **Constrained CCD solver** — uses joint limits + capsule colliders (from rintintin's covariance data) to answer "how close can this IK chain's tip get to its root?"
3. **Any application** that wants to add AGI_articulations to glTF models for runtime constraint-aware animation

## Code Style

Follow tonton conventions:
- `snake_case` for files: `chacha_analyzer.cpp`
- `PascalCase` for types: `Articulation`, `StageType`
- `snake_case` for fields: `min_value`, `max_velocity`
- Units in variable names where applicable: `angle_rad`, `velocity_rad_s`
- SI units throughout

## File Structure

```
chacha/
├── CLAUDE.md
├── CMakeLists.txt
├── LICENSE
├── README.md
├── include/
│   ├── chacha.h              // Public API: analyze() function
│   ├── chacha_types.h        // Input/output data structures
│   ├── chacha_stage.h        // StageType enum and Stage struct
│   ├── chacha_naming.h       // sanitize_articulation_name / stage_name_for for AGI output
│   └── chacha_clean.h        // Public API: clean() for keyframe reduction
└── src/
    ├── chacha_internal.h     // Internal types: Chart, Candidate, JointMotion, pipeline decls
    ├── chacha_analyzer.cpp   // Main analysis pipeline: scan resolution, rest-relative
    │                         // conversion, dispatch to config/search paths, stage assembly
    ├── chacha_charts.cpp     // The 12 Euler charts: build_charts, solve_euler, compose_chart,
    │                         // alternate_branch, chart_conditioning
    ├── chacha_search.cpp     // select_candidate: the general 1/2/3-axis candidate search
    ├── chacha_reduced.cpp    // Closed-form 1-axis solve and Gauss-Newton 2-axis solve,
    │                         // plus their residual metrics
    ├── chacha_dp.cpp         // Viterbi/DP branch resolution and cross-animation anchoring
    ├── chacha_config.cpp     // Configuration-animation fast path: is_configuration_motion,
    │                         // solve_configuration
    ├── chacha_summary.cpp    // Trajectory -> CandidateSummary: resampling, velocity/
    │                         // acceleration, range union
    ├── chacha_filter.cpp     // Noise filtering and thresholds
    ├── chacha_pointing.cpp   // Model-wide pointing-vector inference (infer_pointing_vectors)
    ├── chacha_naming.cpp     // AGI-compliant name sanitisation and stage naming
    └── chacha_clean.cpp      // Redundant keyframe removal and interpolation promotion
```
