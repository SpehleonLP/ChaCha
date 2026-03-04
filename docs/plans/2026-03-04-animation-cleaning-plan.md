# Animation Cleaning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a standalone `clean()` API to ChaCha that removes redundant keyframes from animation channels via interpolation analysis, with optional interpolation type promotion.

**Architecture:** New public header `include/chacha_clean.h` with `CleanOptions`, `CleanedChannel` structs and `clean()` function. Implementation in `src/chacha_clean.cpp`. Completely independent of the existing analysis pipeline — no modifications to existing files except `CMakeLists.txt` (to add the new header to `PUBLIC_HEADER`).

**Tech Stack:** C++20, GLM (slerp, quat, vec3), no other dependencies.

---

### Task 1: Create public header `include/chacha_clean.h`

**Files:**
- Create: `include/chacha_clean.h`

**Step 1: Write the header file**

```cpp
#ifndef CHACHA_CLEAN_H
#define CHACHA_CLEAN_H

#include "chacha_types.h"
#include <span>
#include <vector>

namespace ChaCha {

struct CleanOptions {
    float tolerance = 1e-4f;              // Max error for redundant keyframe removal
    float frame_time = 0.0f;             // Target frame duration (s). 0 = disabled.
    bool promote_step_to_linear = false;  // Requires frame_time > 0
    bool promote_linear_to_cubic = false; // Requires frame_time > 0
};

struct CleanedChannel {
    int node{-1};
    Property property{Property::Rotation};
    InterpolationType interp{InterpolationType::Linear};
    std::vector<float> times;
    std::vector<float> values;
};

/// Remove redundant keyframes from animation channels.
/// Returns new owned data with reduced keyframe counts.
std::vector<CleanedChannel> clean(
    std::span<const AnimationChannel> channels,
    const CleanOptions& options = {}
);

} // namespace ChaCha

#endif // CHACHA_CLEAN_H
```

**Step 2: Update CMakeLists.txt to include the new header**

In `CMakeLists.txt` line 31, add `chacha_clean.h` to the `PUBLIC_HEADER` property:

```cmake
set_target_properties(chacha PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/chacha.h;${CMAKE_CURRENT_SOURCE_DIR}/include/chacha_types.h;${CMAKE_CURRENT_SOURCE_DIR}/include/chacha_stage.h;${CMAKE_CURRENT_SOURCE_DIR}/include/chacha_clean.h"
)
```

Note: source files are auto-globbed (`file(GLOB_RECURSE CHACHA_SOURCES ...)`), so `src/chacha_clean.cpp` will be picked up automatically.

**Step 3: Verify build compiles**

Run: `cd /mnt/Passport/Libraries/Spehleon/tonton-example/modules/chacha && mkdir -p build && cd build && cmake .. && make`
Expected: Compiles with no errors (header is included but not yet used by anything).

**Step 4: Commit**

```bash
git add include/chacha_clean.h CMakeLists.txt
git commit -m "feat: add chacha_clean.h header with CleanOptions and CleanedChannel types"
```

---

### Task 2: Implement redundant keyframe removal for Step channels

**Files:**
- Create: `src/chacha_clean.cpp`

**Step 1: Write the Step channel cleaning logic**

Create `src/chacha_clean.cpp` with the `clean()` function skeleton and Step channel handling.

For Step interpolation, a middle keyframe B between A and C is redundant if the values at B are identical (within tolerance) to the values at A. This is because Step interpolation holds the previous value until the next keyframe — if B's value equals A's value, removing B doesn't change the animation (C's time still triggers the step change).

For vec3 properties (Translation/Scale), `values_per_key = 3`. For quaternions (Rotation), `values_per_key = 4`. Compare each component within tolerance.

```cpp
#include "chacha_clean.h"
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace ChaCha {

namespace {

int values_per_key(Property prop)
{
    return (prop == Property::Rotation) ? 4 : 3;
}

int stride_for(Property prop, InterpolationType interp)
{
    int vpk = values_per_key(prop);
    return (interp == InterpolationType::CubicSpline) ? vpk * 3 : vpk;
}

// Return pointer to the "value" portion of keyframe i (skipping in-tangent for CubicSpline)
const float* key_value(const float* values, int i, int vpk, InterpolationType interp)
{
    int s = (interp == InterpolationType::CubicSpline) ? vpk * 3 : vpk;
    int offset = i * s;
    if (interp == InterpolationType::CubicSpline) offset += vpk; // skip in-tangent
    return values + offset;
}

bool vec_approx_equal(const float* a, const float* b, int n, float tol)
{
    for (int i = 0; i < n; ++i) {
        if (std::abs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

// Greedy single-pass keyframe removal for Step channels.
// A middle keyframe is redundant if its value matches the previous kept keyframe's value.
CleanedChannel clean_step(const AnimationChannel& ch, float tolerance)
{
    int vpk = values_per_key(ch.property);
    int num_keys = static_cast<int>(ch.times.size());

    // Build list of kept keyframe indices
    std::vector<int> kept;
    kept.reserve(num_keys);
    kept.push_back(0); // always keep first

    for (int i = 1; i < num_keys - 1; ++i) {
        const float* prev_val = key_value(ch.values.data(), kept.back(), vpk, ch.interp);
        const float* cur_val = key_value(ch.values.data(), i, vpk, ch.interp);

        // For step, B is redundant if value[B] == value[A] (the previous kept key)
        if (!vec_approx_equal(prev_val, cur_val, vpk, tolerance)) {
            kept.push_back(i);
        }
    }

    if (num_keys > 1) kept.push_back(num_keys - 1); // always keep last

    // Build output
    int s = stride_for(ch.property, ch.interp);
    CleanedChannel out;
    out.node = ch.node;
    out.property = ch.property;
    out.interp = ch.interp;
    out.times.reserve(kept.size());
    out.values.reserve(kept.size() * s);

    for (int idx : kept) {
        out.times.push_back(ch.times[idx]);
        const float* src = ch.values.data() + idx * s;
        out.values.insert(out.values.end(), src, src + s);
    }

    return out;
}

} // anonymous namespace

std::vector<CleanedChannel> clean(
    std::span<const AnimationChannel> channels,
    const CleanOptions& options)
{
    std::vector<CleanedChannel> result;
    result.reserve(channels.size());

    for (const auto& ch : channels) {
        int num_keys = static_cast<int>(ch.times.size());

        // Channels with <= 2 keyframes: return as-is
        if (num_keys <= 2) {
            CleanedChannel out;
            out.node = ch.node;
            out.property = ch.property;
            out.interp = ch.interp;
            out.times.assign(ch.times.begin(), ch.times.end());
            out.values.assign(ch.values.begin(), ch.values.end());
            result.push_back(std::move(out));
            continue;
        }

        switch (ch.interp) {
        case InterpolationType::Step:
            result.push_back(clean_step(ch, options.tolerance));
            break;
        case InterpolationType::Linear:
            // TODO Task 3: linear cleaning
            {
                CleanedChannel out;
                out.node = ch.node;
                out.property = ch.property;
                out.interp = ch.interp;
                out.times.assign(ch.times.begin(), ch.times.end());
                out.values.assign(ch.values.begin(), ch.values.end());
                result.push_back(std::move(out));
            }
            break;
        case InterpolationType::CubicSpline:
            // TODO Task 4: cubic spline cleaning
            {
                CleanedChannel out;
                out.node = ch.node;
                out.property = ch.property;
                out.interp = ch.interp;
                out.times.assign(ch.times.begin(), ch.times.end());
                out.values.assign(ch.values.begin(), ch.values.end());
                result.push_back(std::move(out));
            }
            break;
        }
    }

    return result;
}

} // namespace ChaCha
```

**Step 2: Verify build compiles**

Run: `cd /mnt/Passport/Libraries/Spehleon/tonton-example/modules/chacha/build && cmake .. && make`
Expected: Compiles with no errors.

**Step 3: Commit**

```bash
git add src/chacha_clean.cpp
git commit -m "feat: implement clean() with Step channel redundancy removal"
```

---

### Task 3: Implement redundant keyframe removal for Linear channels

**Files:**
- Modify: `src/chacha_clean.cpp`

**Step 1: Add linear interpolation helpers and clean_linear function**

Add these functions to the anonymous namespace in `src/chacha_clean.cpp`:

```cpp
// Quaternion angular distance in radians
float quat_distance(const float* a, const float* b)
{
    // q and -q represent the same rotation, so use abs(dot)
    float dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
    dot = std::min(std::abs(dot), 1.0f);
    return 2.0f * std::acos(dot);
}

// Linear interpolation of vec3/vec4 at parameter t
void vec_lerp(const float* a, const float* b, float t, float* out, int n)
{
    for (int i = 0; i < n; ++i) {
        out[i] = a[i] + t * (b[i] - a[i]);
    }
}

// Slerp for quaternions stored as [x,y,z,w] (glTF order)
void quat_slerp(const float* a, const float* b, float t, float* out)
{
    glm::quat qa(a[3], a[0], a[1], a[2]); // glm is w,x,y,z
    glm::quat qb(b[3], b[0], b[1], b[2]);
    glm::quat result = glm::slerp(qa, qb, t);
    out[0] = result.x;
    out[1] = result.y;
    out[2] = result.z;
    out[3] = result.w;
}

// Check if interpolated value at keyframe B (between A and C) is within tolerance
bool is_redundant_linear(
    const float* values, int idx_a, int idx_b, int idx_c,
    const float* times, int vpk, Property prop,
    InterpolationType interp, float tolerance)
{
    const float* va = key_value(values, idx_a, vpk, interp);
    const float* vb = key_value(values, idx_b, vpk, interp);
    const float* vc = key_value(values, idx_c, vpk, interp);

    float ta = times[idx_a], tb = times[idx_b], tc = times[idx_c];
    float t = (tc - ta > 1e-10f) ? (tb - ta) / (tc - ta) : 0.0f;

    if (prop == Property::Rotation) {
        float interp_val[4];
        quat_slerp(va, vc, t, interp_val);
        return quat_distance(vb, interp_val) <= tolerance;
    } else {
        float interp_val[3];
        vec_lerp(va, vc, t, interp_val, vpk);
        return vec_approx_equal(vb, interp_val, vpk, tolerance);
    }
}

// Greedy single-pass keyframe removal for Linear channels.
// Uses the "sliding window" approach: walk forward keeping a "last kept" anchor.
// For each candidate middle key, check if ALL keys between the anchor and the
// next candidate can be reconstructed by interpolating between anchor and candidate.
CleanedChannel clean_linear(const AnimationChannel& ch, float tolerance)
{
    int vpk = values_per_key(ch.property);
    int num_keys = static_cast<int>(ch.times.size());

    std::vector<int> kept;
    kept.reserve(num_keys);
    kept.push_back(0);

    int anchor = 0;
    int candidate = 1;

    while (candidate < num_keys - 1) {
        int next = candidate + 1;

        // Check if ALL keys from anchor+1..candidate can be reconstructed
        // by interpolating between anchor and next
        bool all_ok = true;
        for (int mid = anchor + 1; mid <= candidate; ++mid) {
            if (!is_redundant_linear(
                    ch.values.data(), anchor, mid, next,
                    ch.times.data(), vpk, ch.property, ch.interp, tolerance)) {
                all_ok = false;
                break;
            }
        }

        if (all_ok) {
            // candidate is still redundant, try extending window
            candidate = next;
        } else {
            // candidate cannot be removed — keep it as new anchor
            kept.push_back(candidate);
            anchor = candidate;
            candidate = anchor + 1;
        }
    }

    if (num_keys > 1) kept.push_back(num_keys - 1);

    int s = stride_for(ch.property, ch.interp);
    CleanedChannel out;
    out.node = ch.node;
    out.property = ch.property;
    out.interp = ch.interp;
    out.times.reserve(kept.size());
    out.values.reserve(kept.size() * s);

    for (int idx : kept) {
        out.times.push_back(ch.times[idx]);
        const float* src = ch.values.data() + idx * s;
        out.values.insert(out.values.end(), src, src + s);
    }

    return out;
}
```

**Step 2: Replace the Linear TODO in `clean()` switch**

Replace the `case InterpolationType::Linear:` block with:

```cpp
        case InterpolationType::Linear:
            result.push_back(clean_linear(ch, options.tolerance));
            break;
```

**Step 3: Verify build compiles**

Run: `cd /mnt/Passport/Libraries/Spehleon/tonton-example/modules/chacha/build && cmake .. && make`
Expected: Compiles with no errors.

**Step 4: Commit**

```bash
git add src/chacha_clean.cpp
git commit -m "feat: implement linear channel redundancy removal with slerp for quaternions"
```

---

### Task 4: Implement redundant keyframe removal for CubicSpline channels

**Files:**
- Modify: `src/chacha_clean.cpp`

**Step 1: Add cubic Hermite evaluation helper and clean_cubic function**

CubicSpline in glTF uses cubic Hermite interpolation. Per keyframe, values are stored as `[in_tangent, value, out_tangent]` — each of size `vpk` (3 or 4 floats). The formula between keyframes k and k+1 at parameter t ∈ [0,1] is:

```
p(t) = (2t³ - 3t² + 1) * p₀ + (t³ - 2t² + t) * m₀ * Δt + (-2t³ + 3t²) * p₁ + (t³ - t²) * m₁ * Δt
```

Where `p₀` = value at k, `m₀` = out-tangent at k, `p₁` = value at k+1, `m₁` = in-tangent at k+1, `Δt = time[k+1] - time[k]`.

```cpp
// Evaluate cubic Hermite spline between two keyframes at parameter t ∈ [0,1]
// values_k: pointer to start of keyframe k data [in_tangent(vpk), value(vpk), out_tangent(vpk)]
// values_k1: pointer to start of keyframe k+1 data
// dt: time[k+1] - time[k]
void cubic_hermite_eval(
    const float* values_k, const float* values_k1,
    float t, float dt, int vpk, float* out)
{
    const float* p0 = values_k + vpk;        // value at k
    const float* m0 = values_k + 2 * vpk;    // out-tangent at k
    const float* p1 = values_k1 + vpk;       // value at k+1
    const float* m1 = values_k1;             // in-tangent at k+1

    float t2 = t * t;
    float t3 = t2 * t;

    float h00 = 2*t3 - 3*t2 + 1;
    float h10 = t3 - 2*t2 + t;
    float h01 = -2*t3 + 3*t2;
    float h11 = t3 - t2;

    for (int i = 0; i < vpk; ++i) {
        out[i] = h00 * p0[i] + h10 * m0[i] * dt + h01 * p1[i] + h11 * m1[i] * dt;
    }
}

// For CubicSpline, we need to check that removing keyframe B doesn't change
// the curve. This is only possible when the curve segments A→B and B→C
// already form a single smooth cubic that passes through B at the right time.
// We evaluate the "merged" segment A→C at time_B and compare.
//
// However, merging two cubic segments into one is not generally lossless.
// Instead, we sample the original two-segment curve at several points and
// check if a single segment A→C reproduces those samples within tolerance.
CleanedChannel clean_cubic(const AnimationChannel& ch, float tolerance)
{
    int vpk = values_per_key(ch.property);
    int s = vpk * 3; // stride per keyframe: in_tangent + value + out_tangent
    int num_keys = static_cast<int>(ch.times.size());

    // For CubicSpline, removing a middle control point changes the curve shape.
    // We use a conservative approach: check if the value at B matches what
    // the single-segment cubic A→C would produce at time_B, AND check several
    // intermediate samples.
    //
    // Build kept list using greedy approach.
    std::vector<int> kept;
    kept.reserve(num_keys);
    kept.push_back(0);

    const int num_samples = 4; // intermediate samples to check per segment pair

    for (int i = 1; i < num_keys - 1; ++i) {
        int a = kept.back();
        int c = i + 1;
        if (c >= num_keys) c = num_keys - 1;

        const float* va = ch.values.data() + a * s;
        const float* vc = ch.values.data() + c * s;
        float ta = ch.times[a];
        float tc = ch.times[c];
        float dt_ac = tc - ta;

        if (dt_ac < 1e-10f) {
            kept.push_back(i);
            continue;
        }

        // Check: does the single cubic A→C reproduce the original two-segment
        // curve at keyframe B and at intermediate points?
        bool redundant = true;

        // Check original curve at several time samples between A and C
        // Original curve: A→B segment and B→C segment
        float tb = ch.times[i];
        const float* vb = ch.values.data() + i * s;

        // Sample points include the keyframe time and intermediate times
        for (int si = 0; si <= num_samples; ++si) {
            float sample_t_abs;
            if (si == 0) {
                sample_t_abs = tb; // always check the keyframe time itself
            } else {
                sample_t_abs = ta + dt_ac * si / num_samples;
            }

            // Evaluate original two-segment curve
            float orig[4];
            if (sample_t_abs <= tb) {
                float dt_ab = tb - ta;
                float t_local = (dt_ab > 1e-10f) ? (sample_t_abs - ta) / dt_ab : 0.0f;
                cubic_hermite_eval(va, vb, t_local, dt_ab, vpk, orig);
            } else {
                float dt_bc = tc - tb;
                float t_local = (dt_bc > 1e-10f) ? (sample_t_abs - tb) / dt_bc : 0.0f;
                cubic_hermite_eval(vb, vc, t_local, dt_bc, vpk, orig);
            }

            // Evaluate single-segment A→C
            float merged[4];
            float t_local = (sample_t_abs - ta) / dt_ac;
            cubic_hermite_eval(va, vc, t_local, dt_ac, vpk, merged);

            if (ch.property == Property::Rotation) {
                if (quat_distance(orig, merged) > tolerance) {
                    redundant = false;
                    break;
                }
            } else {
                if (!vec_approx_equal(orig, merged, vpk, tolerance)) {
                    redundant = false;
                    break;
                }
            }
        }

        if (!redundant) {
            kept.push_back(i);
        }
    }

    if (num_keys > 1) kept.push_back(num_keys - 1);

    CleanedChannel out;
    out.node = ch.node;
    out.property = ch.property;
    out.interp = ch.interp;
    out.times.reserve(kept.size());
    out.values.reserve(kept.size() * s);

    for (int idx : kept) {
        out.times.push_back(ch.times[idx]);
        const float* src = ch.values.data() + idx * s;
        out.values.insert(out.values.end(), src, src + s);
    }

    return out;
}
```

**Step 2: Replace the CubicSpline TODO in `clean()` switch**

```cpp
        case InterpolationType::CubicSpline:
            result.push_back(clean_cubic(ch, options.tolerance));
            break;
```

**Step 3: Verify build compiles**

Run: `cd /mnt/Passport/Libraries/Spehleon/tonton-example/modules/chacha/build && cmake .. && make`
Expected: Compiles with no errors.

**Step 4: Commit**

```bash
git add src/chacha_clean.cpp
git commit -m "feat: implement CubicSpline channel redundancy removal"
```

---

### Task 5: Implement Step→Linear promotion

**Files:**
- Modify: `src/chacha_clean.cpp`

**Step 1: Add step-to-linear promotion function**

This runs after redundant removal. For each Step channel, sample the original step function at every `frame_time` interval, then check if linear interpolation between the (cleaned) keyframes reproduces those samples within tolerance.

Key insight: Step interpolation holds the value of keyframe A until the exact time of keyframe B. Linear interpolation transitions gradually. The promotion is valid if the visual difference at frame boundaries is within tolerance.

```cpp
// Try promoting a Step channel to Linear interpolation.
// Returns true if promotion succeeded (modifies channel in-place).
bool try_promote_step_to_linear(CleanedChannel& ch, float frame_time, float tolerance)
{
    if (ch.interp != InterpolationType::Step) return false;
    if (ch.times.size() < 2) return false;
    if (frame_time <= 0.0f) return false;

    int vpk = values_per_key(ch.property);
    float t_start = ch.times.front();
    float t_end = ch.times.back();

    // For each frame time, compare step-sampled value vs linear-interpolated value
    for (float t = t_start; t <= t_end; t += frame_time) {
        // Find the segment this time falls in
        // Step: find last keyframe with time <= t
        int step_idx = 0;
        for (int i = 1; i < static_cast<int>(ch.times.size()); ++i) {
            if (ch.times[i] <= t) step_idx = i;
            else break;
        }
        const float* step_val = ch.values.data() + step_idx * vpk;

        // Linear: find the segment [a, b] where time[a] <= t < time[b]
        int lin_a = 0;
        for (int i = 1; i < static_cast<int>(ch.times.size()); ++i) {
            if (ch.times[i] <= t) lin_a = i;
            else break;
        }
        int lin_b = std::min(lin_a + 1, static_cast<int>(ch.times.size()) - 1);

        float lerp_val[4];
        if (lin_a == lin_b) {
            std::copy_n(ch.values.data() + lin_a * vpk, vpk, lerp_val);
        } else {
            float seg_t = (t - ch.times[lin_a]) / (ch.times[lin_b] - ch.times[lin_a]);
            const float* va = ch.values.data() + lin_a * vpk;
            const float* vb = ch.values.data() + lin_b * vpk;

            if (ch.property == Property::Rotation) {
                quat_slerp(va, vb, seg_t, lerp_val);
            } else {
                vec_lerp(va, vb, seg_t, lerp_val, vpk);
            }
        }

        // Compare
        if (ch.property == Property::Rotation) {
            if (quat_distance(step_val, lerp_val) > tolerance) return false;
        } else {
            if (!vec_approx_equal(step_val, lerp_val, vpk, tolerance)) return false;
        }
    }

    ch.interp = InterpolationType::Linear;
    return true;
}
```

**Step 2: Call promotion after redundancy removal in `clean()`**

Add after the switch statement, before pushing to result (restructure the switch to always produce `CleanedChannel cleaned`, then apply promotions):

```cpp
    // After redundancy removal, apply promotions
    if (options.frame_time > 0.0f) {
        if (options.promote_step_to_linear) {
            try_promote_step_to_linear(cleaned, options.frame_time, options.tolerance);
        }
    }
```

This requires refactoring the switch to store the result in a local `CleanedChannel cleaned` variable first, then apply promotions, then push to result.

**Step 3: Verify build compiles**

Run: `cd /mnt/Passport/Libraries/Spehleon/tonton-example/modules/chacha/build && cmake .. && make`
Expected: Compiles with no errors.

**Step 4: Commit**

```bash
git add src/chacha_clean.cpp
git commit -m "feat: implement Step-to-Linear interpolation promotion"
```

---

### Task 6: Implement Linear→CubicSpline promotion

**Files:**
- Modify: `src/chacha_clean.cpp`

**Step 1: Add linear-to-cubic promotion function**

Strategy: Fit a cubic spline through the linear keyframes using a greedy approach. Start with keyframe 0, try to fit a single cubic segment through as many keyframes as possible. The cubic segment uses the keyframe values as control points with tangents derived from finite differences. Check if the cubic reproduces the original linear curve (sampled at `frame_time` intervals) within tolerance.

```cpp
// Compute finite difference tangent at keyframe i
void compute_tangent(const CleanedChannel& ch, int i, int vpk, float* tangent)
{
    int n = static_cast<int>(ch.times.size());

    if (n <= 1) {
        std::fill_n(tangent, vpk, 0.0f);
        return;
    }

    const float* vals = ch.values.data();

    if (i == 0) {
        // Forward difference
        float dt = ch.times[1] - ch.times[0];
        if (dt < 1e-10f) dt = 1.0f;
        for (int j = 0; j < vpk; ++j)
            tangent[j] = (vals[1*vpk + j] - vals[0*vpk + j]) / dt;
    } else if (i == n - 1) {
        // Backward difference
        float dt = ch.times[n-1] - ch.times[n-2];
        if (dt < 1e-10f) dt = 1.0f;
        for (int j = 0; j < vpk; ++j)
            tangent[j] = (vals[(n-1)*vpk + j] - vals[(n-2)*vpk + j]) / dt;
    } else {
        // Central difference
        float dt = ch.times[i+1] - ch.times[i-1];
        if (dt < 1e-10f) dt = 1.0f;
        for (int j = 0; j < vpk; ++j)
            tangent[j] = (vals[(i+1)*vpk + j] - vals[(i-1)*vpk + j]) / dt;
    }
}

// Evaluate what the original linear channel produces at time t
void eval_linear_at(const CleanedChannel& ch, float t, int vpk, float* out)
{
    int n = static_cast<int>(ch.times.size());
    if (n == 0) return;

    // Find segment
    int seg = 0;
    for (int i = 1; i < n; ++i) {
        if (ch.times[i] <= t) seg = i;
        else break;
    }
    int next = std::min(seg + 1, n - 1);

    if (seg == next) {
        std::copy_n(ch.values.data() + seg * vpk, vpk, out);
        return;
    }

    float seg_t = (t - ch.times[seg]) / (ch.times[next] - ch.times[seg]);
    const float* va = ch.values.data() + seg * vpk;
    const float* vb = ch.values.data() + next * vpk;

    if (ch.property == Property::Rotation) {
        quat_slerp(va, vb, seg_t, out);
    } else {
        vec_lerp(va, vb, seg_t, out, vpk);
    }
}

// Try promoting a Linear channel to CubicSpline.
// Only promotes if the cubic representation uses fewer keyframes.
bool try_promote_linear_to_cubic(CleanedChannel& ch, float frame_time, float tolerance)
{
    if (ch.interp != InterpolationType::Linear) return false;
    if (ch.times.size() < 4) return false; // need at least 4 to potentially reduce
    if (frame_time <= 0.0f) return false;

    int vpk = values_per_key(ch.property);
    int n = static_cast<int>(ch.times.size());

    // Greedy: try to span as many original keyframes as possible per cubic segment
    struct CubicKey {
        float time;
        float in_tangent[4];
        float value[4];
        float out_tangent[4];
    };

    std::vector<CubicKey> cubic_keys;
    cubic_keys.reserve(n);

    // Always include first keyframe
    CubicKey first;
    first.time = ch.times[0];
    std::copy_n(ch.values.data(), vpk, first.value);
    compute_tangent(ch, 0, vpk, first.out_tangent);
    std::copy_n(first.out_tangent, vpk, first.in_tangent); // not used for first
    cubic_keys.push_back(first);

    int anchor = 0;

    while (anchor < n - 1) {
        // Try to reach as far as possible from anchor
        int best_end = anchor + 1;

        for (int end = anchor + 2; end < n; ++end) {
            // Build cubic segment from anchor to end
            float t0 = ch.times[anchor];
            float t1 = ch.times[end];
            float dt = t1 - t0;
            if (dt < 1e-10f) break;

            const float* p0 = ch.values.data() + anchor * vpk;
            const float* p1 = ch.values.data() + end * vpk;

            float m0[4], m1[4];
            compute_tangent(ch, anchor, vpk, m0);
            compute_tangent(ch, end, vpk, m1);

            // Sample at frame intervals and check against original linear curve
            bool ok = true;
            for (float t = t0; t <= t1 && ok; t += frame_time) {
                float local_t = (t - t0) / dt;

                // Evaluate cubic hermite
                float t2 = local_t * local_t;
                float t3 = t2 * local_t;
                float h00 = 2*t3 - 3*t2 + 1;
                float h10 = t3 - 2*t2 + local_t;
                float h01 = -2*t3 + 3*t2;
                float h11 = t3 - t2;

                float cubic_val[4];
                for (int j = 0; j < vpk; ++j) {
                    cubic_val[j] = h00*p0[j] + h10*m0[j]*dt + h01*p1[j] + h11*m1[j]*dt;
                }

                // Evaluate original linear at same time
                float linear_val[4];
                eval_linear_at(ch, t, vpk, linear_val);

                if (ch.property == Property::Rotation) {
                    if (quat_distance(cubic_val, linear_val) > tolerance) {
                        ok = false;
                    }
                } else {
                    if (!vec_approx_equal(cubic_val, linear_val, vpk, tolerance)) {
                        ok = false;
                    }
                }
            }

            if (ok) {
                best_end = end;
            } else {
                break;
            }
        }

        // Add best_end as a cubic keyframe
        CubicKey key;
        key.time = ch.times[best_end];
        std::copy_n(ch.values.data() + best_end * vpk, vpk, key.value);
        compute_tangent(ch, best_end, vpk, key.out_tangent);
        compute_tangent(ch, best_end, vpk, key.in_tangent);
        cubic_keys.push_back(key);

        anchor = best_end;
    }

    // Only promote if we actually reduced the keyframe count
    if (static_cast<int>(cubic_keys.size()) >= n) return false;

    // Build output
    int s = vpk * 3; // CubicSpline stride
    ch.interp = InterpolationType::CubicSpline;
    ch.times.clear();
    ch.values.clear();
    ch.times.reserve(cubic_keys.size());
    ch.values.reserve(cubic_keys.size() * s);

    for (const auto& ck : cubic_keys) {
        ch.times.push_back(ck.time);
        ch.values.insert(ch.values.end(), ck.in_tangent, ck.in_tangent + vpk);
        ch.values.insert(ch.values.end(), ck.value, ck.value + vpk);
        ch.values.insert(ch.values.end(), ck.out_tangent, ck.out_tangent + vpk);
    }

    return true;
}
```

**Step 2: Add the promotion call in `clean()` after step-to-linear promotion**

```cpp
        if (options.promote_linear_to_cubic) {
            try_promote_linear_to_cubic(cleaned, options.frame_time, options.tolerance);
        }
```

Note: this should come after `promote_step_to_linear` so that a step channel that was promoted to linear can then be considered for cubic promotion if both flags are set.

**Step 3: Verify build compiles**

Run: `cd /mnt/Passport/Libraries/Spehleon/tonton-example/modules/chacha/build && cmake .. && make`
Expected: Compiles with no errors.

**Step 4: Commit**

```bash
git add src/chacha_clean.cpp
git commit -m "feat: implement Linear-to-CubicSpline interpolation promotion"
```

---

### Task 7: Refactor clean() flow and final cleanup

**Files:**
- Modify: `src/chacha_clean.cpp`

**Step 1: Refactor the clean() function for clean flow**

Ensure the final `clean()` function has this structure:

```cpp
std::vector<CleanedChannel> clean(
    std::span<const AnimationChannel> channels,
    const CleanOptions& options)
{
    std::vector<CleanedChannel> result;
    result.reserve(channels.size());

    for (const auto& ch : channels) {
        int num_keys = static_cast<int>(ch.times.size());

        // Channels with <= 2 keyframes: return as-is
        if (num_keys <= 2) {
            CleanedChannel out;
            out.node = ch.node;
            out.property = ch.property;
            out.interp = ch.interp;
            out.times.assign(ch.times.begin(), ch.times.end());
            out.values.assign(ch.values.begin(), ch.values.end());
            result.push_back(std::move(out));
            continue;
        }

        // Step 1: Redundant keyframe removal
        CleanedChannel cleaned;
        switch (ch.interp) {
        case InterpolationType::Step:
            cleaned = clean_step(ch, options.tolerance);
            break;
        case InterpolationType::Linear:
            cleaned = clean_linear(ch, options.tolerance);
            break;
        case InterpolationType::CubicSpline:
            cleaned = clean_cubic(ch, options.tolerance);
            break;
        }

        // Step 2: Interpolation type promotion (requires frame_time)
        if (options.frame_time > 0.0f) {
            if (options.promote_step_to_linear) {
                try_promote_step_to_linear(cleaned, options.frame_time, options.tolerance);
            }
            if (options.promote_linear_to_cubic) {
                try_promote_linear_to_cubic(cleaned, options.frame_time, options.tolerance);
            }
        }

        result.push_back(std::move(cleaned));
    }

    return result;
}
```

**Step 2: Remove all TODO comments, ensure no dead code**

**Step 3: Verify build compiles**

Run: `cd /mnt/Passport/Libraries/Spehleon/tonton-example/modules/chacha/build && cmake .. && make`
Expected: Clean compile with no warnings.

**Step 4: Commit**

```bash
git add src/chacha_clean.cpp
git commit -m "refactor: clean up clean() function flow"
```

---

### Task 8: Update CLAUDE.md with new file and API

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add `chacha_clean.h` to the File Structure section**

In the file structure tree, add under `include/`:
```
│   └── chacha_clean.h        // Public API: clean() function for keyframe reduction
```

Add under `src/`:
```
    └── chacha_clean.cpp      // Redundant keyframe removal and interpolation promotion
```

**Step 2: Add a brief section about animation cleaning**

Add after the "Processing Pipeline" section, a short description of the cleaning API:

```markdown
### Animation Cleaning (Optional)

A separate `clean()` API removes redundant keyframes from animation channels:
- Redundant keyframe removal (Step, Linear, CubicSpline)
- Optional interpolation type promotion (Step→Linear, Linear→CubicSpline)

See `include/chacha_clean.h` for the `CleanOptions` and `CleanedChannel` types.
```

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document animation cleaning API in CLAUDE.md"
```
