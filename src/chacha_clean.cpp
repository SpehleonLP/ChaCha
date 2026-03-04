#include "chacha_clean.h"
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <cmath>
#include <cassert>

namespace ChaCha {
namespace {

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

int values_per_key(Property prop)
{
    return prop == Property::Rotation ? 4 : 3;
}

int stride_for(Property prop, InterpolationType interp)
{
    int vpk = values_per_key(prop);
    return interp == InterpolationType::CubicSpline ? vpk * 3 : vpk;
}

const float* key_value(const float* values, int i, int vpk, InterpolationType interp)
{
    if (interp == InterpolationType::CubicSpline) {
        // Layout per key: [in_tangent(vpk), value(vpk), out_tangent(vpk)]
        return values + i * vpk * 3 + vpk;
    }
    return values + i * vpk;
}

bool vec_approx_equal(const float* a, const float* b, int n, float tol)
{
    for (int i = 0; i < n; ++i) {
        if (std::abs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

float quat_distance(const float* a, const float* b)
{
    // [x,y,z,w] layout
    float dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
    float clamped = std::min(std::abs(dot), 1.0f);
    return 2.0f * std::acos(clamped);
}

void vec_lerp(const float* a, const float* b, float t, float* out, int n)
{
    for (int i = 0; i < n; ++i) {
        out[i] = a[i] + t * (b[i] - a[i]);
    }
}

void quat_slerp(const float* a, const float* b, float t, float* out)
{
    // glTF stores [x,y,z,w], glm::quat constructor is (w,x,y,z)
    glm::quat qa(a[3], a[0], a[1], a[2]);
    glm::quat qb(b[3], b[0], b[1], b[2]);
    glm::quat result = glm::slerp(qa, qb, t);
    out[0] = result.x;
    out[1] = result.y;
    out[2] = result.z;
    out[3] = result.w;
}

void cubic_hermite_eval(const float* values_k, const float* values_k1,
                        float t, float dt, int vpk, float* out)
{
    // values_k layout:  [in_tangent(vpk), value(vpk), out_tangent(vpk)]
    const float* p0 = values_k + vpk;       // value of keyframe k
    const float* m0 = values_k + 2 * vpk;   // out-tangent of keyframe k
    const float* p1 = values_k1 + vpk;      // value of keyframe k+1
    const float* m1 = values_k1;             // in-tangent of keyframe k+1

    float t2 = t * t;
    float t3 = t2 * t;
    float h00 = 2.0f*t3 - 3.0f*t2 + 1.0f;
    float h10 = t3 - 2.0f*t2 + t;
    float h01 = -2.0f*t3 + 3.0f*t2;
    float h11 = t3 - t2;

    for (int i = 0; i < vpk; ++i) {
        out[i] = h00*p0[i] + h10*m0[i]*dt + h01*p1[i] + h11*m1[i]*dt;
    }
}

// ---------------------------------------------------------------------------
// Interpolation helpers for cleaned channels
// ---------------------------------------------------------------------------

bool is_close(const float* a, const float* b, int vpk, Property prop, float tol)
{
    if (prop == Property::Rotation) {
        return quat_distance(a, b) <= tol;
    }
    return vec_approx_equal(a, b, vpk, tol);
}

void interpolate_at(const float* a, const float* b, float t, float* out,
                    int vpk, Property prop)
{
    if (prop == Property::Rotation) {
        quat_slerp(a, b, t, out);
    } else {
        vec_lerp(a, b, t, out, vpk);
    }
}

// Evaluate a Linear CleanedChannel at arbitrary time t
void eval_linear_at(const CleanedChannel& ch, float t, int vpk, float* out)
{
    int n = static_cast<int>(ch.times.size());
    if (n == 0) return;

    // Clamp to range
    if (t <= ch.times[0]) {
        const float* v = ch.values.data();
        for (int i = 0; i < vpk; ++i) out[i] = v[i];
        return;
    }
    if (t >= ch.times[n - 1]) {
        const float* v = ch.values.data() + (n - 1) * vpk;
        for (int i = 0; i < vpk; ++i) out[i] = v[i];
        return;
    }

    // Binary search for segment
    int lo = 0, hi = n - 1;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (ch.times[mid] <= t) lo = mid;
        else hi = mid;
    }

    float seg_t = (t - ch.times[lo]) / (ch.times[hi] - ch.times[lo]);
    const float* a = ch.values.data() + lo * vpk;
    const float* b = ch.values.data() + hi * vpk;
    interpolate_at(a, b, seg_t, out, vpk, ch.property);
}

// Evaluate a Step CleanedChannel at arbitrary time t
void eval_step_at(const CleanedChannel& ch, float t, int vpk, float* out)
{
    int n = static_cast<int>(ch.times.size());
    if (n == 0) return;

    // Find the last keyframe with time <= t
    int idx = 0;
    for (int i = n - 1; i >= 0; --i) {
        if (ch.times[i] <= t) {
            idx = i;
            break;
        }
    }

    const float* v = ch.values.data() + idx * vpk;
    for (int i = 0; i < vpk; ++i) out[i] = v[i];
}

// Compute tangent for linear-to-cubic promotion via finite differences
void compute_tangent(const CleanedChannel& ch, int i, int vpk, float* tangent)
{
    int n = static_cast<int>(ch.times.size());

    if (n <= 1) {
        for (int j = 0; j < vpk; ++j) tangent[j] = 0.0f;
        return;
    }

    const float* vals = ch.values.data();

    if (i == 0) {
        // Forward difference
        float dt = ch.times[1] - ch.times[0];
        if (dt <= 0.0f) dt = 1.0f;
        for (int j = 0; j < vpk; ++j) {
            tangent[j] = (vals[1 * vpk + j] - vals[0 * vpk + j]) / dt;
        }
    } else if (i == n - 1) {
        // Backward difference
        float dt = ch.times[n - 1] - ch.times[n - 2];
        if (dt <= 0.0f) dt = 1.0f;
        for (int j = 0; j < vpk; ++j) {
            tangent[j] = (vals[(n - 1) * vpk + j] - vals[(n - 2) * vpk + j]) / dt;
        }
    } else {
        // Central difference
        float dt = ch.times[i + 1] - ch.times[i - 1];
        if (dt <= 0.0f) dt = 1.0f;
        for (int j = 0; j < vpk; ++j) {
            tangent[j] = (vals[(i + 1) * vpk + j] - vals[(i - 1) * vpk + j]) / dt;
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 1: Redundant keyframe removal
// ---------------------------------------------------------------------------

CleanedChannel clean_step(const AnimationChannel& ch, float tolerance)
{
    int vpk = values_per_key(ch.property);
    int s = stride_for(ch.property, ch.interp);
    int n = static_cast<int>(ch.times.size());

    CleanedChannel result;
    result.node = ch.node;
    result.property = ch.property;
    result.interp = InterpolationType::Step;

    if (n == 0) return result;

    // Always keep first keyframe
    auto push_key = [&](int i) {
        result.times.push_back(ch.times[i]);
        const float* v = key_value(ch.values.data(), i, vpk, ch.interp);
        result.values.insert(result.values.end(), v, v + vpk);
    };

    push_key(0);

    for (int i = 1; i < n - 1; ++i) {
        const float* prev_val = result.values.data() + (result.times.size() - 1) * vpk;
        const float* cur_val = key_value(ch.values.data(), i, vpk, ch.interp);

        if (!is_close(cur_val, prev_val, vpk, ch.property, tolerance)) {
            push_key(i);
        }
    }

    // Always keep last keyframe if more than one
    if (n > 1) {
        push_key(n - 1);
    }

    return result;
}

bool is_redundant_linear(const float* times, const float* values,
                         int idx_a, int idx_b, int idx_c,
                         int vpk, Property prop, InterpolationType interp,
                         float tolerance)
{
    float time_a = times[idx_a];
    float time_b = times[idx_b];
    float time_c = times[idx_c];

    float dt = time_c - time_a;
    if (dt <= 0.0f) return true;

    float t = (time_b - time_a) / dt;

    const float* val_a = key_value(values, idx_a, vpk, interp);
    const float* val_b = key_value(values, idx_b, vpk, interp);
    const float* val_c = key_value(values, idx_c, vpk, interp);

    float interpolated[4];
    interpolate_at(val_a, val_c, t, interpolated, vpk, prop);

    return is_close(interpolated, val_b, vpk, prop, tolerance);
}

CleanedChannel clean_linear(const AnimationChannel& ch, float tolerance)
{
    int vpk = values_per_key(ch.property);
    int n = static_cast<int>(ch.times.size());

    CleanedChannel result;
    result.node = ch.node;
    result.property = ch.property;
    result.interp = InterpolationType::Linear;

    if (n == 0) return result;

    auto push_key = [&](int i) {
        result.times.push_back(ch.times[i]);
        const float* v = key_value(ch.values.data(), i, vpk, ch.interp);
        result.values.insert(result.values.end(), v, v + vpk);
    };

    push_key(0); // Always keep first

    if (n <= 2) {
        if (n == 2) push_key(1);
        return result;
    }

    int anchor = 0;

    for (int candidate = 1; candidate < n - 1; ++candidate) {
        // Check if ALL keys in [anchor+1..candidate] can be reconstructed
        // by interpolating between anchor and candidate+1
        bool all_redundant = true;
        for (int k = anchor + 1; k <= candidate; ++k) {
            if (!is_redundant_linear(ch.times.data(), ch.values.data(),
                                     anchor, k, candidate + 1,
                                     vpk, ch.property, ch.interp, tolerance)) {
                all_redundant = false;
                break;
            }
        }

        if (!all_redundant) {
            // candidate cannot be skipped; keep it as new anchor
            push_key(candidate);
            anchor = candidate;
        }
    }

    // Always keep last
    push_key(n - 1);

    return result;
}

CleanedChannel clean_cubic(const AnimationChannel& ch, float tolerance)
{
    int vpk = values_per_key(ch.property);
    int full_stride = vpk * 3; // CubicSpline stride per key
    int n = static_cast<int>(ch.times.size());

    CleanedChannel result;
    result.node = ch.node;
    result.property = ch.property;
    result.interp = InterpolationType::CubicSpline;

    if (n == 0) return result;

    auto push_key = [&](int i) {
        result.times.push_back(ch.times[i]);
        const float* base = ch.values.data() + i * full_stride;
        result.values.insert(result.values.end(), base, base + full_stride);
    };

    push_key(0);

    if (n <= 2) {
        if (n == 2) push_key(1);
        return result;
    }

    // Track which keyframes are kept
    std::vector<bool> kept(n, false);
    kept[0] = true;
    kept[n - 1] = true;

    // For each middle keyframe, check if it can be removed
    // Greedy: walk through and try to skip consecutive keyframes
    int anchor = 0;

    for (int candidate = 1; candidate < n - 1; ++candidate) {
        // Check if we can remove candidate by merging anchor→candidate→next_kept
        // into anchor→next_kept using the single cubic segment
        int next = candidate + 1;

        const float* vals_a = ch.values.data() + anchor * full_stride;
        const float* vals_c = ch.values.data() + next * full_stride;
        float time_a = ch.times[anchor];
        float time_c = ch.times[next];
        float dt_ac = time_c - time_a;

        if (dt_ac <= 0.0f) continue;

        bool can_remove = true;

        // Check all keys between anchor and next by sampling
        for (int mid = anchor; mid < next && can_remove; ++mid) {
            int seg_end = mid + 1;
            float time_start = ch.times[mid];
            float time_end = ch.times[seg_end];
            float dt_seg = time_end - time_start;
            if (dt_seg <= 0.0f) continue;

            const float* vals_seg_start = ch.values.data() + mid * full_stride;
            const float* vals_seg_end = ch.values.data() + seg_end * full_stride;

            // Sample at keyframe times and 4 intermediate points
            for (int s = 0; s <= 4; ++s) {
                float frac = static_cast<float>(s) / 4.0f;
                float sample_time = time_start + frac * dt_seg;

                // Original curve evaluation (on segment mid→seg_end)
                float orig[4];
                cubic_hermite_eval(vals_seg_start, vals_seg_end,
                                   frac, dt_seg, vpk, orig);

                // Merged curve evaluation (on segment anchor→next)
                float t_merged = (sample_time - time_a) / dt_ac;
                float merged[4];
                cubic_hermite_eval(vals_a, vals_c, t_merged, dt_ac, vpk, merged);

                if (!is_close(orig, merged, vpk, ch.property, tolerance)) {
                    can_remove = false;
                    break;
                }
            }
        }

        if (!can_remove) {
            push_key(candidate);
            anchor = candidate;
        }
    }

    push_key(n - 1);

    return result;
}

// ---------------------------------------------------------------------------
// Phase 2: Interpolation type promotion
// ---------------------------------------------------------------------------

bool try_promote_step_to_linear(CleanedChannel& ch, float frame_time, float tolerance)
{
    int vpk = values_per_key(ch.property);
    int n = static_cast<int>(ch.times.size());

    if (n < 2 || frame_time <= 0.0f) return false;

    float t_start = ch.times.front();
    float t_end = ch.times.back();

    // Sample at every frame_time interval
    for (float t = t_start; t <= t_end; t += frame_time) {
        float step_val[4];
        float linear_val[4];

        // What step interpolation produces
        eval_step_at(ch, t, vpk, step_val);

        // What linear interpolation would produce
        eval_linear_at(ch, t, vpk, linear_val);

        if (!is_close(step_val, linear_val, vpk, ch.property, tolerance)) {
            return false;
        }
    }

    ch.interp = InterpolationType::Linear;
    return true;
}

bool try_promote_linear_to_cubic(CleanedChannel& ch, float frame_time, float tolerance)
{
    int vpk = values_per_key(ch.property);
    int n = static_cast<int>(ch.times.size());

    if (n < 4 || frame_time <= 0.0f) return false;

    // Build cubic representation greedily
    struct CubicKey {
        float time;
        std::vector<float> in_tangent;
        std::vector<float> value;
        std::vector<float> out_tangent;
    };

    std::vector<CubicKey> cubic_keys;

    auto make_cubic_key = [&](int i) -> CubicKey {
        CubicKey key;
        key.time = ch.times[i];
        key.value.assign(ch.values.data() + i * vpk, ch.values.data() + (i + 1) * vpk);
        key.in_tangent.resize(vpk);
        key.out_tangent.resize(vpk);
        compute_tangent(ch, i, vpk, key.in_tangent.data());
        // For simplicity, in_tangent = out_tangent at each key
        std::copy(key.in_tangent.begin(), key.in_tangent.end(), key.out_tangent.begin());
        return key;
    };

    // Greedy: from anchor, try to span as many linear keyframes as possible with one cubic segment
    int anchor = 0;
    cubic_keys.push_back(make_cubic_key(0));

    while (anchor < n - 1) {
        int best_end = anchor + 1;

        for (int end = anchor + 2; end < n; ++end) {
            // Try cubic segment from anchor to end
            CubicKey key_a = make_cubic_key(anchor);
            CubicKey key_b = make_cubic_key(end);

            float dt = ch.times[end] - ch.times[anchor];
            if (dt <= 0.0f) break;

            // Build temporary cubic data for evaluation
            std::vector<float> vals_a(vpk * 3);
            std::vector<float> vals_b(vpk * 3);
            std::copy(key_a.in_tangent.begin(), key_a.in_tangent.end(), vals_a.begin());
            std::copy(key_a.value.begin(), key_a.value.end(), vals_a.begin() + vpk);
            std::copy(key_a.out_tangent.begin(), key_a.out_tangent.end(), vals_a.begin() + 2 * vpk);
            std::copy(key_b.in_tangent.begin(), key_b.in_tangent.end(), vals_b.begin());
            std::copy(key_b.value.begin(), key_b.value.end(), vals_b.begin() + vpk);
            std::copy(key_b.out_tangent.begin(), key_b.out_tangent.end(), vals_b.begin() + 2 * vpk);

            // Validate by sampling at frame_time intervals
            bool valid = true;
            for (float t = ch.times[anchor]; t <= ch.times[end] + frame_time * 0.5f; t += frame_time) {
                if (t > ch.times[end]) t = ch.times[end];

                float local_t = (t - ch.times[anchor]) / dt;
                local_t = std::clamp(local_t, 0.0f, 1.0f);

                float cubic_val[4];
                cubic_hermite_eval(vals_a.data(), vals_b.data(), local_t, dt, vpk, cubic_val);

                float linear_val[4];
                eval_linear_at(ch, t, vpk, linear_val);

                if (!is_close(cubic_val, linear_val, vpk, ch.property, tolerance)) {
                    valid = false;
                    break;
                }

                if (t >= ch.times[end]) break;
            }

            if (valid) {
                best_end = end;
            } else {
                break;
            }
        }

        if (best_end > anchor) {
            if (best_end != anchor + 1 || cubic_keys.back().time != ch.times[anchor]) {
                // Only add if not already added
            }
            cubic_keys.push_back(make_cubic_key(best_end));
            anchor = best_end;
        } else {
            break;
        }
    }

    // Only promote if cubic has fewer keyframes
    if (static_cast<int>(cubic_keys.size()) >= n) return false;

    // Rebuild the channel in CubicSpline format
    ch.interp = InterpolationType::CubicSpline;
    ch.times.clear();
    ch.values.clear();

    for (const auto& key : cubic_keys) {
        ch.times.push_back(key.time);
        ch.values.insert(ch.values.end(), key.in_tangent.begin(), key.in_tangent.end());
        ch.values.insert(ch.values.end(), key.value.begin(), key.value.end());
        ch.values.insert(ch.values.end(), key.out_tangent.begin(), key.out_tangent.end());
    }

    return true;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<CleanedChannel> clean(
    std::span<const AnimationChannel> channels,
    const CleanOptions& options)
{
    std::vector<CleanedChannel> result;
    result.reserve(channels.size());

    for (const auto& ch : channels) {
        int n = static_cast<int>(ch.times.size());

        // Channels with <= 2 keyframes: copy as-is
        if (n <= 2) {
            CleanedChannel out;
            out.node = ch.node;
            out.property = ch.property;
            out.interp = ch.interp;
            out.times.assign(ch.times.begin(), ch.times.end());

            int vpk = values_per_key(ch.property);
            int s = stride_for(ch.property, ch.interp);
            out.values.assign(ch.values.begin(), ch.values.begin() + n * s);
            result.push_back(std::move(out));
            continue;
        }

        // Phase 1: redundancy removal
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

        // Phase 2: interpolation promotion (only when frame_time > 0)
        if (options.frame_time > 0.0f) {
            if (options.promote_step_to_linear &&
                cleaned.interp == InterpolationType::Step) {
                try_promote_step_to_linear(cleaned, options.frame_time, options.tolerance);
            }

            if (options.promote_linear_to_cubic &&
                cleaned.interp == InterpolationType::Linear) {
                try_promote_linear_to_cubic(cleaned, options.frame_time, options.tolerance);
            }
        }

        result.push_back(std::move(cleaned));
    }

    return result;
}

} // namespace ChaCha
