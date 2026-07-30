# Articulation Solver Rework — Design

**Date:** 2026-07-30
**Status:** Approved, pending implementation plan
**Scope:** ChaCha library + both consumers (gltfRepackager, tonton-example)

## Summary

Replace ChaCha's rotation decomposition with an exact solver, add a dynamic-programming
pass that resolves per-frame solution-branch ambiguity, and add a search over candidate
stage sets that finds the fewest degrees of freedom explaining a joint's motion. Fix the
range, scale, and AGI-conformance defects found in review. Honour artist-authored
`AGI ` configuration animations as authoritative when present.

Animation cleaning (`chacha_clean.cpp`) is explicitly **out of scope** — it is covered by
a separate spec that interns the more mature implementation from
`gltfRepackager/src/steps/clean_animations.cpp`.

## Motivation

ChaCha's stated goal is to deduce the *most restrictive* AGI articulations that explain a
model's animations. It currently fails at this in three independent ways, each verified
against the existing sources.

### The decomposition is not a decomposition

`swing_to_angles` (`src/chacha_decompose.cpp:31`) reads angles off quaternion components:

```cpp
float x_angle_rad = 2.0f * std::atan2(s.x, s.w);
float z_angle_rad = 2.0f * std::atan2(s.z, s.w);
```

This is only valid when one of the two angles is near zero. Measured against the current
build:

| Input rotation | Reported | Best reconstruction error |
|---|---|---|
| 90° swing about (1,0,1) | x=+70.53°, z=+70.53° | 38.94° |
| X45 then Z45 | x=+36.87°, y=−19.47°, z=+50.48° | 15.50° |

Two consequences. Ranges are inflated, which is the opposite of "most restrictive". And
**phantom degrees of freedom are invented**: a joint performing a pure two-axis swing with
zero axial rotation (fixed X=45°, Z sweeping 0→90°) currently reports

```
xRotate  min=+0.00  max=+45.00
zRotate  min=+0.00  max=+90.00
yRotate  min=-45.00 max=+0.00     <- not present in the input
```

`optimize_stage_order` (`src/chacha_analyzer.cpp:214`) exists to brute-force the least-bad
axis ordering, compensating for this defect, and discards the residual silently.

### Ranges are seeded to zero

`segment_and_merge` (`src/chacha_segment.cpp:20-22`) initialises min and max to `0.0f`
before folding in samples, forcing every range to contain the rest pose. A joint whose
animation lives entirely between 30° and 60° reports `min=0.00, max=60.00` — 50% too wide,
with an `initialValue` the joint never assumes.

### The output is not valid AGI

Verified against the published schema
(`KhronosGroup/glTF/extensions/2.0/Vendor/AGI_articulations/schema/articulation.stage.schema.json`):

- Stage `name` is required; the tonton bridge never writes it.
- Articulation `name` must match `^[^\s]+$` and be unique per model; both consumers pass
  raw node names through, and rigs routinely contain `"Left Arm"` or duplicates.
- `maximumSpeed` / `maximumEffort` are not fields in the stage schema.
- `gltfRepackager/src/steps/infer_articulations.cpp` emits **radians** into fields the
  spec defines as degrees.
- Scale stages store `value − 1.0` (additive) where AGI defines a multiplicative factor,
  and emit `initialValue = 0`, which instructs a conformant viewer to collapse the node.

### Dead computation

`chacha_pointing.cpp` infers a pointing vector that the tonton bridge overwrites with a
hardcoded `{0, 0, 1}` (`chacha_fxgltf_bridge.cpp:279`).

## Design

### Pipeline

The existing pipeline commits to per-axis angles in `extract_dof_tracks` *before* anything
knows what the stage set will be, then tries to repair the choice in
`optimize_stage_order`. The rework inverts this. Per joint:

```
rest-relative quaternion trajectories (per animation)
  │
  ├─ 1. Candidate enumeration:  12 three-stage charts, plus reduced 1- and 2-stage sets
  │
  ├─ 2. For EVERY candidate × EVERY scanned animation:
  │        exact solve       → finite branch set per frame
  │        Viterbi DP        → continuous angle trajectory
  │        anchor            → canonicalise the free 2pi offset per axis
  │        summarise         → discard trajectory, keep fixed-size CandidateSummary
  │
  ├─ 3. Per candidate: union summaries across animations
  │
  ├─ 4. Score the unioned candidates:  (dof_count, sum of range, conditioning)
  │        lexicographic; accept the best
  │
  └─ 5. Emit Articulation
```

**The chart is chosen once per joint, over the unioned result — never per animation.**
The score depends on total range, and range only exists after the union, so every candidate
must be solved against every scanned animation before anything is scored. Allowing
different animations to select different charts would be incoherent: the resulting angles
would not be commensurable and could not be unioned at all.

This costs no additional compute — the full candidate × animation cross product was always
required — provided solved trajectories are not retained. Per `(candidate, animation)` the
pipeline keeps only:

```cpp
struct CandidateSummary {
    float min_value[3], max_value[3];   // per stage axis, after anchoring
    float max_velocity[3];
    float max_acceleration[3];
    float max_residual_rad;             // reduced-DOF acceptance test
    float worst_conditioning;           // min over frames
};
```

Roughly ten floats. Retaining full trajectories instead would cost about 100 MB on the
sophia corpus; summaries cost a few hundred kilobytes. The DP needs its trajectory only
transiently and releases it once the summary is extracted.

File layout:

| File | Change |
|---|---|
| `src/chacha_decompose.cpp` | Rewritten as the exact solver. Swing-twist removed. |
| `src/chacha_dp.cpp` | **New.** Viterbi branch resolution. |
| `src/chacha_search.cpp` | **New.** Candidate enumeration and scoring. Absorbs `optimize_stage_order`. |
| `src/chacha_config.cpp` | **New.** `AGI ` configuration-animation fast path. |
| `src/chacha_segment.cpp` | Reduced to range/velocity/acceleration accumulation over a solved trajectory. |
| `src/chacha_filter.cpp` | Unchanged except scale semantics. |
| `src/chacha_analyzer.cpp` | Reduced to orchestration. |
| `src/chacha_pointing.cpp` | Unchanged; now actually consumed. |
| `src/chacha_clean.cpp` | Untouched (separate spec). |

### The exact solver

One generic routine covers all 12 charts rather than 12 hand-coded extractions. The
preferred basis is the Bernardes & Viollet direct quaternion-to-Euler method
(*PLoS ONE*, 2022), which handles proper-Euler and Tait-Bryan sequences uniformly,
operates on the quaternion without a matrix round-trip, and reports its own singularity
indicator. If that reference proves unsuitable on inspection, Shoemake's 24-convention
matrix extraction is an equivalent fallback; the requirement the design actually depends on
is a *single* parameterised routine covering all 12 orders with explicit singularity
detection, not any particular published formulation.

The 12 charts are the 6 Tait-Bryan orders (XYZ, XZY, YXZ, YZX, ZXY, ZYX) and the 6
proper-Euler orders (XYX, XZX, YXY, YZY, ZXZ, ZYZ). Proper-Euler orders are expressible in
AGI because stage `type` carries no uniqueness constraint — only stage `name` must be
unique within an articulation — so `zRotate / xRotate / zRotate` is legal output.

Including proper-Euler charts matters for singularity avoidance. Tait-Bryan charts are
singular when the middle angle reaches ±90°; proper-Euler charts are singular when it
reaches 0 or π. These are different regions of SO(3), so a joint ill-conditioned in all
six Tait-Bryan charts is frequently well-conditioned in a proper-Euler one.

**For three distinct axes in a fixed order the fit is exact**; there is no residual at
3 DOF. Residual is a property of *reduced*-DOF candidates and serves as the acceptance
test in the search, not as an error signal on accepted full-rank fits.

Both chart families yield exactly **two** solution branches per frame generically, so the
branch set is uniform at size 2 regardless of chart.

### Branch resolution by dynamic programming

Given the previous frame's angle, the 2π lift is taken as the **principal difference** —
the multiple of 2π minimising `|Δθ|`. This is always well defined and always yields
`|Δθ| ≤ π`, so there is no continuity-infeasible case to handle. An aliased fast rotation
(350°/frame) resolves to its minimal reading (−10°/frame), which is both unavoidable and
the most restrictive interpretation consistent with the samples.

The remaining freedom is which of the two branches to take at each frame. This is a
Viterbi pass:

- **State:** `(frame, branch)` — two states per frame.
- **Transition cost:** `Σ_axes (Δθ)²` over principal differences.
- **Memo entry:** a `float` cost plus a one-byte backpointer. Angles are recomputed on the
  backward walk rather than stored.
- **Complexity:** `O(T · 4)` per chart per animation.

Total variation is used rather than range because range (`max − min`) is not additive over
time and therefore not amenable to Bellman recursion. Minimising total variation is a
proxy that also directly penalises the discontinuities we want to avoid. An exact
minimum-range formulation would require lifting the state to carry Pareto-pruned running
intervals; this was considered and rejected as disproportionate.

**Animations are solved independently.** Continuity holds only within an animation;
ranges union across them afterwards. The sophia test model has 88 animations, so flattening
would manufacture 87 artificial discontinuities per joint.

### Cross-animation anchoring

Solving animations independently leaves a free parameter that must be pinned before the
union, or the union is wrong.

The Viterbi pass determines a trajectory only up to its choice of starting branch and a
global multiple of 2π per axis — every such variant has identical transition costs, so the
DP is indifferent between them. Two animations of the same joint can therefore settle in
offsets differing by 2π, describing the same physical pose as `+190°` in one and `−170°` in
the other. Unioning those yields a 360° range for a joint that moved 20°. This is the
cross-animation unwrapping defect identified in review, and per-animation DP reintroduces
it unless anchoring is explicit.

Anchoring rule: after the DP, shift each animation's solved trajectory, per axis, by the
multiple of 2π that places the trajectory's **midpoint** — `(min + max) / 2` — into
`(−π, π]`. The rest pose is the origin at 0, and a physical joint does not sit more than
half a turn from rest, so this anchor is well founded. It is O(1) per axis per animation.

The midpoint is used rather than the first sample because an animation may begin at an
extreme of its range; anchoring on the midpoint keeps the whole excursion centred on rest
rather than pushing one tail across the wrap boundary.

Anchoring happens before summarisation, so `CandidateSummary` bounds are already in the
common frame and the union is a plain per-axis min/max.

### Reduced-DOF candidates

- **One stage:** closed form. The nearest rotation about axis `a` to `q` is
  `θ = 2·atan2(dot(q_vec, a), q_w)`.
- **Two stages:** Gauss-Newton in two unknowns, seeded from the 3-stage solution with the
  third angle zeroed. Converges in a handful of iterations.

A reduced candidate is accepted only if the **maximum** per-frame residual across all
scanned animations is within `Options::max_fit_residual_rad` (default 0.02 rad ≈ 1.2°).
Maximum rather than RMS, because the claim being made is "explains every pose", not
"explains poses on average".

### Scoring

Lexicographic: `dof_count` ascending, then `Σ range` ascending, then conditioning
descending.

Minimising summed range automatically avoids gimbal-degenerate charts: near a singularity
the outer two angles diverge in opposite directions and cancel, which presents as a chart
with very large ranges. Conditioning therefore acts as a tiebreak and a diagnostic rather
than a gate.

**Four-stage fallback (deferred).** SO(3) admits no singularity-free three-angle chart, so
a trajectory crossing the singular set of all 12 charts cannot be covered by any 3-stage
articulation. A redundant fourth stage removes the singularity, exactly as a fourth gimbal
ring does on an inertial platform, and cannot be collapsed back to three in general. It is
deferred rather than built: 4 stages is strictly less restrictive and so ranks below every
3-stage fit, and the redundant DOF replaces the finite branch set with a continuous family,
substantially complicating the DP. It will be added only if real models exhibit joints
scoring badly across all 12 charts. The integration test on sophia reports the
worst-conditioned chart score so we have evidence either way.

### `AGI ` configuration animations

An artist-authored configuration animation exercises one DOF at a time, returning to rest
between phases, in the order the stages should apply. Verified in
`treefrog.glb` (`Head`, `Spine.003`), rest-relative:

```
t = 0.00 – 1.67    pure X    -13.4° → +13.4° → rest
t = 1.67 – 3.33    pure Z    -13.4° → +13.4° → rest
t = 3.33 – 5.00    pure Y    -13.4° → +13.4° → rest
```

Joints that do not articulate carry two keyframes and hold rest for the full duration, so
the configuration animation also declares which joints are locked.

Such an animation supplies stage set, stage order, ranges, and locked joints directly; no
search runs. Note that the order cannot be *derived* from the data, since only one DOF is
non-zero at a time and `R_x(θ)·I·I` equals `I·I·R_x(θ)`. The phase sequence is the author
declaring the order, and is taken as authoritative on that basis.

Observed names in the sample models are `"AGI Configuration"`, `"AGI configuration"`, and
`"AGI Configuration.001"` — a space, not an underscore, with Blender's duplicate suffix.
Detection is a case-insensitive `"agi "` prefix test, which catches all three. The existing
`AGI_` prefix matcher in the tonton bridge matches none of them.

Consequence to document: when the scan narrows to configuration animations, velocity and
acceleration derive from whatever sweep speed the artist authored, typically a slow uniform
ramp rather than a real speed limit. Ranges and stage order are the trustworthy outputs in
that mode.

### Velocity and acceleration

Both are measured from the solved angle trajectory **resampled at a fixed rate**, using
central differences over a window of several frames. Naive per-keyframe differencing is
invalid here: the sophia corpus is 100% LINEAR, and the second derivative of a
piecewise-linear signal is a train of impulses whose magnitude scales with keyframe density
rather than with motion. The resample rate and window width are exposed as options.

Effort and inertia are explicitly **not** in scope. Torque limits require mass and inertia
data, and joint effort depends on the motion of the entire subtree; this is not derivable
from keyframes and is not ChaCha's problem.

## API

### `include/chacha_types.h`

```cpp
struct Animation {
    std::string_view name;   // used to auto-detect "AGI " configuration animations
};

struct AnimationChannel {
    int node{-1};
    int animation{0};                    // NEW: index into the animations span
    Property property{Property::Rotation};
    InterpolationType interp{InterpolationType::Linear};
    std::span<const float> times;
    std::span<const float> values;
};

struct Skeleton {
    std::span<const int> parents;
    std::span<const glm::quat> rest_rotations;
    std::span<const glm::vec3> rest_translations;
    std::span<const glm::vec3> rest_scales;    // NEW: empty means all-ones
};

struct Articulation {
    int node{-1};
    std::string name;
    std::vector<Stage> stages;
    glm::vec3 pointing_vector{0.0f, 1.0f, 0.0f};
    uint8_t dof_count{0};                // NEW: rotation stages in the accepted fit
    float fit_residual_rad{0.0f};        // NEW: max per-frame residual; ~0 at 3 DOF
};

struct Options {
    float rotation_threshold_rad{0.01f};
    float translation_threshold_m{0.001f};
    float scale_threshold{0.01f};
    float max_fit_residual_rad{0.02f};   // NEW: reduced-DOF acceptance
    float resample_rate_hz{60.0f};       // NEW: for velocity/acceleration
    int   derivative_window{5};          // NEW: central-difference width, in samples
    bool  prioritize_rom_animations{true};  // auto-narrow to "AGI " animations
};

struct Diagnostic {
    int node;
    int animation;
    enum Kind { NonUnitQuaternion, EmptyChannel, MalformedValues } kind;
};
```

### `include/chacha_stage.h`

```cpp
struct Stage {
    StageType type;
    float min_value{};
    float max_value{};
    float initial_value{};
    float max_velocity{};
    float max_acceleration{};   // RENAMED from max_effort
};
```

### `include/chacha.h`

```cpp
std::vector<Articulation> analyze(
    std::span<const AnimationChannel> channels,
    std::span<const Animation> animations,
    const Skeleton& skeleton,
    const Options& options = {},
    std::span<const int> scan = {},
    std::vector<Diagnostic>* diagnostics = nullptr);
```

`scan` empty means all animations. In that case, if `options.prioritize_rom_animations` is
true and any animation name begins with `"AGI "` (case-insensitive), the scan narrows to
those animations and all others are ignored. A non-empty `scan` is honoured verbatim with
no auto-narrowing.

**Index-space contract**, to be stated in `chacha.h`: `AnimationChannel::node` indexes the
same array as `Skeleton::parents`. ChaCha works in **glTF node space**. Skin-joint space is
incorrect — AGI articulations are per-node, and glTF animation channels target nodes, not
skin joints.

### Behavioural changes to existing fields

**Range seeding.** `segment_and_merge` seeds min/max from the first observed sample instead
of zero. A joint living in 30°–60° reports exactly that.

**`initial_value`.** Set to the rest-relative value at the earliest keyframe of the
earliest scanned animation, then clamped into `[min, max]`. This satisfies AGI's
requirement without inflating the range. When 0 falls outside the observed range that is
preserved as information, not smoothed away.

**Scale.** Stored as a multiplicative factor rather than an additive delta.
`initial_value` defaults to 1.0. The noise filter compares `|factor − 1|` against
`scale_threshold`.

## Consumer conformance

`gltfRepackager/src/steps/infer_articulations.cpp` is the reference implementation;
`tonton-example/src/chacha_fxgltf_bridge.cpp` follows it. Both are kept.

| Item | gltfRepackager | tonton-example |
|---|---|---|
| Radians → degrees for rotate stages | **Fix** (currently raw radians) | Already correct |
| Stage `name`, unique within articulation | Fix uniqueness for repeated types | **Add** (absent) |
| Articulation `name` sanitisation + dedup | Add | Add |
| `pointing_vector` forwarded | Already correct | **Fix** (hardcoded `{0,0,1}`) |
| `chachaMaximumSpeed` / `chachaMaximumAcceleration` in stage `extras` | Add | Add |
| `channel.animation` populated | **Required** | **Required** |
| Node-space skeleton | Already correct | **Fix** (remove skin remap) |
| `"agi "` prefix detection | Replace normalised substring match | **Fix** (`AGI_` matches nothing) |

Stage names are `"<type>"` for the first occurrence and `"<type>2"`, `"<type>3"` for
repeats, which is required once proper-Euler charts emit two stages of the same type.

Articulation names collapse whitespace to `_` and take a `_2` suffix on collision. This is
shared logic and lives in ChaCha as `sanitize_articulation_name()` rather than being
written twice.

`chachaMaximumSpeed` is degrees/second for rotate stages; `chachaMaximumAcceleration` is
degrees/second². Both go in the stage's `extras` and are documented as ChaCha extensions,
not AGI. Keys are omitted entirely when the value is zero.

The node-space ruling deletes the `node_to_joint` remap in `chacha_main.cpp:161-173` and
changes `extract_skeleton` to build arrays over all `doc.nodes` rather than `skin.joints`.

## Testing

A new GTest target in ChaCha, matching the tonton-example house style
(`GTest::gtest` / `gtest_discover_tests`).

### Unit

- **Solver round-trip.** Random quaternions across all 12 charts reconstruct within
  1e-5 rad. This property underwrites everything else.
- **Phantom DOF regression.** A pure two-axis swing (fixed X=45°, Z sweeping 0→90°)
  produces **no** `yRotate` stage. Currently produces `yRotate −45°..0°`.
- **Range seeding regression.** A joint living in 30°–60° reports `min=30°, max=60°`.
  Currently reports `0°..60°`.
- **DP continuity.** A trajectory crossing ±180° unwraps monotonically; an aliased
  rotation resolves to its minimal reading.
- **Cross-animation anchoring.** Two animations of one joint straddling ±180° from
  opposite sides must union to the true narrow range, not to ~360°. This is the specific
  defect the anchoring rule exists to prevent, and it cannot be caught by any
  single-animation test.
- **Global chart selection.** A joint whose animations individually favour different
  charts must still emit a single chart, with ranges expressed in it. Guards against a
  regression to per-animation chart choice, which would silently produce incommensurable
  angles.
- **DOF search.** Synthetic 1-, 2-, and 3-DOF joints yield exactly 1, 2, and 3 stages with
  residual within tolerance.
- **Name sanitisation.** Spaces collapse, collisions get suffixes, output matches
  `^[^\s]+$`.

### Integration

- **`treefrog.glb`** — ground truth, since the configuration animation is authoritative.
  Assert `Head` and `Spine.003` produce exactly three stages in order X, Z, Y at ±13.4°,
  and that joints held at rest produce no articulation.
- **`sophia-2_9.glb`** — volume. 49 joints, 88 animations, 12,936 channels, ~2M keyframes,
  all LINEAR. Assert round-trip reconstruction: apply each emitted articulation at its
  solved per-frame angles and compare against the original keyframe quaternion across
  every joint and animation. Assert mixamo's constant scale and translation channels
  produce no stages except at the hips. **Report** the `dof_count` histogram, the
  worst-conditioned chart score, and whether elbows and knees came out as hinges — these
  are smell tests, not assertions, because mixamo's rig is anatomically inaccurate (the
  elbow bends off-axis to compensate for a shoulder that does not twist).
- **Performance guard** so the roughly 8M-solve search does not silently regress. The
  search is embarrassingly parallel per joint if it proves too slow.

Test models live in `chacha/testdata/`, which is gitignored.

## Rollout

1. ChaCha: solver, DP, search, configuration path, API change, full test suite. Consumers
   build against a temporary shim.
2. gltfRepackager: adopt the new API, apply conformance fixes, remove the shim.
3. tonton-example: adopt the new API, apply conformance fixes, remove the skin-space remap.

## Deletions

- `swing_to_angles`, `twist_to_angle`, `decompose_swing_twist`, and the `SwingTwist` struct.
- `optimize_stage_order` and its brute-force permutation loop.
- The `node_to_joint` remap in `chacha_main.cpp`.

## Out of scope

- **Animation cleaning.** `chacha_clean.cpp` is untouched here. A separate spec covers
  interning the mature implementation from
  `gltfRepackager/src/steps/clean_animations.cpp` (Douglas-Peucker simplification,
  Schneider bezier fitting, quaternion log/exp, sign alignment) and deleting the
  half-finished version.
- **Effort and inertia**, per the reasoning above.
- **The four-stage fallback**, deferred pending evidence from the sophia integration test.
