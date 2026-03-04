# CLAUDE.md

## Project Overview

ChaCha is a standalone library that deduces joint articulation constraints from skeletal animation data. Given a set of animation channels (rotation/translation/scale keyframes per joint), it produces per-joint constraint descriptors: ranges of motion, velocity limits, and effort limits for each degree of freedom.

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
// Per animation channel: one joint, one property, one set of keyframes
struct AnimationChannel {
    int node;                    // Joint/node index in skeleton
    Property property;           // Translation, Rotation, Scale
    InterpolationType interp;    // Linear, Step, CubicSpline
    std::span<const float> times;       // Keyframe timestamps (seconds)
    std::span<const float> values;      // Keyframe values (3 or 4 floats per key)
};

// Skeleton hierarchy (just parents array + rest poses)
struct Skeleton {
    std::span<const int> parents;
    std::span<const glm::quat> rest_rotations;   // Rest pose per joint
    std::span<const glm::vec3> rest_translations;
};
```

### Output

```cpp
// Per degree of freedom for a joint
struct Stage {
    StageType type;         // xRotate, yRotate, zRotate, xTranslate, ...
    float min_value;        // Lower bound (radians or meters)
    float max_value;        // Upper bound
    float initial_value;    // Rest/neutral value (usually 0)
    float max_velocity;     // Speed limit (rad/s or m/s)
    float max_effort;       // Force/torque limit (optional, mass-dependent)
};

// All constraints for one joint
struct Articulation {
    int node;
    std::string name;           // Human-readable (e.g., "left_elbow")
    std::vector<Stage> stages;  // One per degree of freedom
};
```

### Processing Pipeline

```
Animation Channels (per joint)
  │
  ├─ 1. Convert keyframes to rest-pose-relative values
  │     (subtract rest rotation/translation so constraints are relative to bind pose)
  │
  ├─ 2. Compute per-keyframe derivatives
  │     (velocity = Δvalue / Δtime between consecutive keyframes)
  │
  ├─ 3. Segment into motion stages
  │     A "stage" is a continuous motion away from and back to the rest pose.
  │     Track when the joint deviates from rest (start) and returns (end).
  │     Multiple stages may exist for the same DOF (e.g., flex then extend).
  │
  ├─ 4. Decompose multi-axis motion into single-axis DOFs
  │     For rotations: use swing-twist decomposition to separate
  │     cone-of-motion (swing) from axial rotation (twist).
  │     For translations: project onto bone-local axes.
  │     Key challenge: quaternion keyframes that rotate around multiple
  │     axes simultaneously need to be decomposed into independent DOFs.
  │
  ├─ 5. Merge compatible stages
  │     Stages of the same type (e.g., both yRotate) with similar
  │     directions get merged. Take union of min/max ranges,
  │     maximum of velocity/effort limits.
  │
  ├─ 6. Filter noise
  │     Discard stages with negligible motion (< threshold).
  │     Default thresholds: 0.01 rad for rotation, 0.001m for translation.
  │
  └─ 7. Produce Articulation per joint
        Collect all surviving stages into a single Articulation struct.
        Joints with no significant motion get no articulation (locked).
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

### Multi-axis rotation decomposition (Step 4)

**Chosen approach: Swing-twist decomposition.** Each rest-relative quaternion is decomposed into:
- **Twist**: axial rotation around bone's local Y axis → `yRotate` stage
- **Swing**: cone-of-motion perpendicular to Y → `xRotate` + `zRotate` stages

This maps naturally to anatomical joints (elbow = pure swing, forearm = pure twist, shoulder = swing + twist). Implementation is in `src/chacha_decompose.cpp`.

Alternatives considered and rejected:
- Euler decomposition: suffers from gimbal lock on multi-axis joints
- PCA on rotation vectors: axes don't align with intuitive anatomical directions

### Stage segmentation (Step 3)

Multiple animations contributing to the same joint are merged by taking the union of observed ranges and the maximum of observed velocities. This correctly handles oscillating animations (walking cycles) by capturing the full swing range rather than treating each direction as a separate stage.

### Limitations

ChaCha reports *observed* ROM, not *possible* ROM. If no animation fully extends a joint, the range will be underestimated. Callers can add safety margins or provide a dedicated ROM animation.

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

The example program:
```
chacha-analyze input.glb [output.glb]
```
Return codes:
- `0` — Success: AGI articulations generated and written
- `1` — Already has AGI_articulations (skip, or overwrite with `--force`)
- `2` — No animations in file (nothing to analyze)
- `3` — Cannot disentangle motion stages from animations (ambiguous multi-axis motion)
- `4` — File I/O error

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
│   └── chacha_clean.h        // Public API: clean() for keyframe reduction
└── src/
    ├── chacha_internal.h     // Internal types: DofTrack, RawStage, pipeline decls
    ├── chacha_analyzer.cpp   // Main analysis pipeline (steps 1-7)
    ├── chacha_decompose.cpp  // Swing-twist decomposition (step 4)
    ├── chacha_segment.cpp    // Stage segmentation and merging (steps 3, 5)
    ├── chacha_filter.cpp     // Noise filtering and thresholds (step 6)
    └── chacha_clean.cpp      // Redundant keyframe removal and interpolation promotion
```
