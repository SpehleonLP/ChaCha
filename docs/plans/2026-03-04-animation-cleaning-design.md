# Animation Cleaning Design

## Summary

Add a standalone public API to ChaCha that removes redundant keyframes from animation channels via interpolation analysis. Operates on raw keyframe data (quaternions, vec3s) independently of the constraint analysis pipeline.

## Public API

### New header: `include/chacha_clean.h`

```cpp
struct CleanOptions {
    float tolerance = 1e-4f;              // Max error for redundant keyframe removal
    float frame_time = 0.0f;             // Target frame duration (e.g., 1/30). 0 = disabled.
    bool promote_step_to_linear = false;  // Requires frame_time > 0
    bool promote_linear_to_cubic = false; // Requires frame_time > 0
};

struct CleanedChannel {
    int node;
    Property property;
    InterpolationType interp;   // May change if promotion occurred
    std::vector<float> times;
    std::vector<float> values;
};

std::vector<CleanedChannel> clean(
    std::span<const AnimationChannel> channels,
    const CleanOptions& options = {}
);
```

## Cleaning Pipeline (per channel)

### Step 1: Redundant Keyframe Removal (always active)

Controlled by `tolerance`. Removes keyframes that can be reconstructed from neighbors.

- **Step channels**: Remove middle keyframe where `value[i-1] == value[i] == value[i+1]` (exact match or within tolerance).
- **Linear vec3**: Remove keyframe B if `lerp(A, C, t_B) ≈ B` within tolerance, where `t_B = (time_B - time_A) / (time_C - time_A)`.
- **Linear quaternion**: Remove keyframe B if `slerp(A, C, t_B) ≈ B` within tolerance. Use ULPS-based float comparison for near-identity quaternion differences.
- **CubicSpline**: Remove keyframe B if cubic spline evaluation at `t_B` produces values within tolerance. CubicSpline keyframes have in-tangent, value, out-tangent (3x the values per keyframe).

Algorithm: iterative pass — scan triplets (A, B, C), remove B if redundant, then re-evaluate neighbors. Repeat until stable. First and last keyframes are always kept.

### Step 2: Interpolation Type Promotion (requires `frame_time > 0`)

After redundant removal, optionally promote interpolation types for further reduction.

- **Step → Linear** (`promote_step_to_linear`): Sample the step channel at every `frame_time` interval. If the linear interpolation between remaining keyframes produces values within `tolerance` of the step-sampled values at each frame boundary, promote to Linear.

- **Linear → CubicSpline** (`promote_linear_to_cubic`): Fit cubic spline segments through the linear keyframes. If fewer cubic keyframes can represent the same curve within `tolerance` when sampled at `frame_time` intervals, promote to CubicSpline and use the reduced set.

## Architecture

### New files
- `include/chacha_clean.h` — public types and `clean()` declaration
- `src/chacha_clean.cpp` — implementation

### No changes to existing files
The cleaning API is fully independent of the analysis pipeline. No modifications to `chacha_analyzer.cpp`, `chacha_types.h`, or any existing code.

### Build
Add `chacha_clean.cpp` to CMakeLists.txt source list and install `chacha_clean.h`.

## Implementation Notes

### Quaternion comparison
For quaternion tolerance checking, compare angular distance: `acos(2 * dot(q1, q2)^2 - 1)`. This gives the geodesic distance in radians, which can be compared directly against `tolerance`.

### CubicSpline format
glTF CubicSpline stores 3 elements per keyframe per component: `[in_tangent, value, out_tangent]`. So for a vec3 property, each keyframe has 9 floats in the values array. The redundancy check must evaluate the cubic Hermite spline correctly.

### Multi-pass vs single-pass
Use a single-pass greedy algorithm (similar to Ramer-Douglas-Peucker but for temporal interpolation). Walk forward, keeping a "last kept" keyframe. For each subsequent keyframe, check if all intermediate keyframes can be reconstructed. When one can't, mark the previous keyframe as kept and continue.

### Edge cases
- Channels with ≤ 2 keyframes: return as-is (nothing to remove)
- Promotion flags without `frame_time > 0`: ignored (no-op)
- Mixed promotion: step→linear runs first, then linear→cubic on the result
