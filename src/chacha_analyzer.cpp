#include "chacha_internal.h"
#include "chacha.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>

namespace ChaCha {

const char* stage_type_name(StageType type)
{
    switch (type) {
    case StageType::xTranslate: return "xTranslate";
    case StageType::yTranslate: return "yTranslate";
    case StageType::zTranslate: return "zTranslate";
    case StageType::xRotate:    return "xRotate";
    case StageType::yRotate:    return "yRotate";
    case StageType::zRotate:    return "zRotate";
    case StageType::xScale:     return "xScale";
    case StageType::yScale:     return "yScale";
    case StageType::zScale:     return "zScale";
    // Invalid is a sentinel for unused Candidate slots (see chacha_internal.h)
    // and must never appear on a real, emitted Stage. Name it explicitly and
    // distinctly from "unknown" so that if it ever does leak into output --
    // a bug -- it is instantly recognisable in a diff or log rather than
    // reading as a plausible-but-wrong stage name.
    case StageType::Invalid:    return "invalid";
    }
    return "unknown";
}

namespace detail {
namespace {

int stride_for(Property p, InterpolationType i)
{
    const int vpk = (p == Property::Rotation) ? 4 : 3;
    return (i == InterpolationType::CubicSpline) ? vpk * 3 : vpk;
}

const float* key_at(const AnimationChannel& ch, int i)
{
    const int vpk    = (ch.property == Property::Rotation) ? 4 : 3;
    const int stride = stride_for(ch.property, ch.interp);
    const int offset = (ch.interp == InterpolationType::CubicSpline) ? vpk : 0;
    return ch.values.data() + static_cast<size_t>(i) * stride + offset;
}

bool channel_is_well_formed(const AnimationChannel& ch)
{
    const int n = static_cast<int>(ch.times.size());
    if (n == 0) return false;
    return ch.values.size() >= static_cast<size_t>(n) * stride_for(ch.property, ch.interp);
}

bool starts_with_agi(std::string_view name)
{
    if (name.size() < 4) return false;
    auto lower = [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); };
    return lower(name[0]) == 'a' && lower(name[1]) == 'g' && lower(name[2]) == 'i' && name[3] == ' ';
}

} // namespace

std::vector<int> resolve_scan(
    std::span<const Animation> animations,
    std::span<const int>       scan,
    const Options&             options)
{
    std::vector<int> out;
    if (!scan.empty()) {                       // explicit request is honoured verbatim
        out.assign(scan.begin(), scan.end());
        return out;
    }

    if (options.prioritize_rom_animations) {
        for (int i = 0; i < static_cast<int>(animations.size()); ++i)
            if (starts_with_agi(animations[i].name)) out.push_back(i);
        if (!out.empty()) return out;          // artist-authored spec wins outright
    }

    for (int i = 0; i < static_cast<int>(animations.size()); ++i) out.push_back(i);
    return out;
}

} // namespace detail

std::vector<Articulation> analyze(
    std::span<const AnimationChannel> channels,
    std::span<const Animation>        animations,
    const Skeleton&                   skeleton,
    const Options&                    options,
    std::span<const int>              scan,
    std::vector<Diagnostic>*          diagnostics)
{
    using namespace detail;

    const int num_nodes = static_cast<int>(skeleton.parents.size());
    if (num_nodes == 0) return {};

    // Rest pose spans are indexed directly by node throughout this function
    // and in infer_pointing_vectors. Unlike rest_scales (documented as
    // optionally empty -> assume all-ones), rest_rotations and
    // rest_translations are a load-bearing contract: a short or empty span
    // here previously segfaulted (ASAN-confirmed) either in the
    // rest-relative rotation computation below or deeper inside
    // infer_pointing_vectors, for callers that supplied no translation
    // channels at all and so had no other reason to notice a malformed
    // Skeleton. Reject outright rather than silently substituting a guessed
    // identity pose per node, which would fabricate constraint data instead
    // of surfacing the caller's error.
    if (static_cast<int>(skeleton.rest_rotations.size()) < num_nodes ||
        static_cast<int>(skeleton.rest_translations.size()) < num_nodes ||
        (!skeleton.rest_scales.empty() &&
         static_cast<int>(skeleton.rest_scales.size()) < num_nodes)) {
        return {};
    }

    const std::vector<int> scanned = resolve_scan(animations, scan, options);
    std::vector<bool> in_scan(animations.size(), false);
    for (int a : scanned)
        if (a >= 0 && a < static_cast<int>(animations.size())) in_scan[a] = true;

    auto note = [&](int node, int anim, Diagnostic::Kind k) {
        if (diagnostics) diagnostics->push_back(Diagnostic{node, anim, k});
    };

    // Per node: rotation trajectories grouped by animation, plus vec3 accumulators
    // for translation/scale. A Vec3Acc's per-axis fields are unit-agnostic: for
    // translation, `lo`/`hi`/`initial` are metres (rest-relative deltas, so the
    // neutral/rest value is 0); for scale they are multiplicative factors (rest
    // value 1).
    std::map<int, JointMotion> rotation_by_node;
    struct Vec3Acc {
        bool  seen[3]{};
        float lo[3]{}, hi[3]{}, vel[3]{}, acc[3]{};
    };
    std::map<int, Vec3Acc> translation_by_node, scale_by_node;

    for (const auto& ch : channels) {
        if (ch.node < 0 || ch.node >= num_nodes) continue;
        if (ch.animation < 0 || ch.animation >= static_cast<int>(animations.size())) {
            // A channel referencing a nonexistent animation index is silently
            // dropped otherwise -- including EVERY channel when `animations`
            // is empty, which is exactly the shape a bridge layer's bug
            // (Tasks 13/14) would take: analyze() would return an empty
            // result with zero diagnostics, indistinguishable from "no
            // motion observed".
            note(ch.node, ch.animation, Diagnostic::UnknownAnimationIndex);
            continue;
        }
        if (!in_scan[ch.animation]) continue;
        if (!channel_is_well_formed(ch)) {
            note(ch.node, ch.animation,
                 ch.times.empty() ? Diagnostic::EmptyChannel : Diagnostic::MalformedValues);
            continue;
        }

        const int n = static_cast<int>(ch.times.size());

        if (ch.property == Property::Rotation) {
            const glm::quat rest_inv = glm::inverse(glm::normalize(skeleton.rest_rotations[ch.node]));
            std::vector<glm::quat> rel;
            std::vector<float>     times;
            rel.reserve(n); times.reserve(n);

            bool degenerate = false;
            for (int i = 0; i < n; ++i) {
                const float* v = key_at(ch, i);
                glm::quat key(v[3], v[0], v[1], v[2]);
                const float len = std::sqrt(glm::dot(key, key));
                if (!(len > 1e-6f)) { degenerate = true; break; }
                if (std::fabs(len - 1.0f) > 1e-3f) degenerate = true;   // still usable
                rel.push_back(rest_inv * (key / len));
                times.push_back(ch.times[i]);
            }
            if (degenerate) note(ch.node, ch.animation, Diagnostic::NonUnitQuaternion);
            if (rel.empty()) continue;

            JointMotion& m = rotation_by_node[ch.node];
            m.rel_by_animation.push_back(std::move(rel));
            m.times_by_animation.push_back(std::move(times));
            continue;
        }

        // Translation and scale. Scale is a MULTIPLICATIVE factor relative to
        // rest, never an additive delta: a rest scale of 1 going to 2 must
        // report [1, 2], not [0, 1]. Guard the near-zero rest divisor.
        const bool is_scale = (ch.property == Property::Scale);
        glm::vec3 rest(0.0f);
        if (is_scale) {
            rest = skeleton.rest_scales.empty() ? glm::vec3(1.0f)
                                                : skeleton.rest_scales[ch.node];
        } else {
            rest = skeleton.rest_translations[ch.node];
        }

        Vec3Acc& acc = is_scale ? scale_by_node[ch.node] : translation_by_node[ch.node];

        // A near-zero rest scale amplifies the ratio arbitrarily: a rest of
        // 1e-7 with the old 1e-8 guard passed through unflagged and reported
        // a stage like [1e7, 2e7] -- numerically "correct" for the ratio
        // definition but useless and silently misleading. kMinRestScale is
        // large enough to catch that case (and the exact-zero case, which
        // previously fell back to raw values with no diagnostic at all);
        // when it trips, report raw (untransformed) values for that axis and
        // flag it rather than fabricate a huge, meaningless range.
        constexpr float kMinRestScale = 1e-4f;
        bool degenerate_rest_scale = false;

        std::vector<float> track[3];
        for (int a = 0; a < 3; ++a) track[a].reserve(n);

        for (int i = 0; i < n; ++i) {
            const float* v = key_at(ch, i);
            for (int a = 0; a < 3; ++a) {
                const float rest_a = rest[a];
                float value;
                if (is_scale) {
                    if (std::fabs(rest_a) > kMinRestScale) {
                        value = v[a] / rest_a;
                    } else {
                        value = v[a];
                        degenerate_rest_scale = true;
                    }
                } else {
                    value = v[a] - rest_a;
                }
                track[a].push_back(value);
            }
        }
        if (degenerate_rest_scale) note(ch.node, ch.animation, Diagnostic::DegenerateRestScale);

        Trajectory t;
        t.time.assign(ch.times.begin(), ch.times.begin() + n);
        for (int a = 0; a < 3; ++a) t.angle[a] = track[a];
        const CandidateSummary s = summarise(t, 0.0f, options);

        for (int a = 0; a < 3; ++a) {
            if (!s.axis_valid[a]) continue;
            if (!acc.seen[a]) {
                acc.seen[a] = true;
                acc.lo[a]   = s.min_value[a];
                acc.hi[a]   = s.max_value[a];
            } else {
                acc.lo[a] = std::min(acc.lo[a], s.min_value[a]);
                acc.hi[a] = std::max(acc.hi[a], s.max_value[a]);
            }
            acc.vel[a] = std::max(acc.vel[a], s.max_velocity[a]);
            acc.acc[a] = std::max(acc.acc[a], s.max_acceleration[a]);
        }
    }

    // Assemble per-node articulations.
    std::map<int, std::vector<RawStage>> raw_by_node;
    std::map<int, Candidate>             candidate_by_node;

    auto push_vec3 = [&](const std::map<int, Vec3Acc>& src, StageType base, bool is_scale) {
        const float neutral = is_scale ? 1.0f : 0.0f;
        for (const auto& [node, acc] : src) {
            for (int a = 0; a < 3; ++a) {
                if (!acc.seen[a]) continue;
                RawStage r;
                r.type             = static_cast<StageType>(static_cast<int>(base) + a);
                r.min_value        = acc.lo[a];
                r.max_value        = acc.hi[a];
                r.initial_value    = neutral;   // clamped into range by filter_stages
                r.max_velocity     = acc.vel[a];
                r.max_acceleration = acc.acc[a];
                raw_by_node[node].push_back(r);
            }
        }
    };

    // Stage emission order across properties is a deliberate, fixed
    // contract, not an accident of these two calls' position relative to
    // the rotation loop below: translate, then scale, then rotate, for
    // every node, regardless of the order the caller's channels arrived in.
    // AGI_articulations stages are order-significant (each stage applies
    // after the previous one in sequence), so downstream consumers need
    // this ordering to be stable across runs and future changes rather than
    // incidental. See Analyze.StageOrderIsTranslateThenScaleThenRotate.
    push_vec3(translation_by_node, StageType::xTranslate, /*is_scale=*/false);
    push_vec3(scale_by_node,       StageType::xScale,     /*is_scale=*/true);

    for (const auto& [node, motion] : rotation_by_node) {
        // An artist-authored configuration animation is a direct
        // specification: it declares the stage set, order and range, so
        // no search is run. resolve_scan already narrowed the scan to
        // "AGI "-prefixed animations when scanning all and any exist, but
        // that only selects WHICH animations were scanned -- it says
        // nothing about whether the motion actually has configuration
        // shape (one axis at a time, returning to rest between phases).
        // is_configuration_motion checks the shape itself, so an
        // AGI-named animation that doesn't actually behave like one falls
        // back to the search below rather than emitting a fabricated
        // stage order.
        Candidate c = is_configuration_motion(motion, options)
                    ? solve_configuration(motion, options)
                    : select_candidate(motion, options);
        if (!c.summary.valid) continue;
        candidate_by_node[node] = c;
        // Stages are an ORDERED SEQUENCE, not a set keyed by StageType: for
        // proper-Euler charts stage[0] == stage[2] by construction (the same
        // physical axis rotated twice, once before and once after the middle
        // stage), and that repetition is legal AGI semantics, not a
        // duplicate to be merged. Map slot s -> the s-th emitted Stage,
        // preserving order exactly; never key or deduplicate by StageType.
        for (int s = 0; s < c.dof; ++s) {
            if (c.stage[s] == StageType::Invalid) continue;
            if (!c.summary.axis_valid[s]) continue;
            RawStage r;
            r.type             = c.stage[s];
            r.min_value        = c.summary.min_value[s];
            r.max_value        = c.summary.max_value[s];
            r.initial_value    = 0.0f;              // rest pose is the origin
            r.max_velocity     = c.summary.max_velocity[s];
            r.max_acceleration = c.summary.max_acceleration[s];
            raw_by_node[node].push_back(r);
        }
    }

    std::vector<Articulation> result;
    for (auto& [node, raws] : raw_by_node) {
        std::vector<Stage> stages = filter_stages(raws, options);
        if (stages.empty()) continue;

        Articulation art;
        art.node = node;
        art.stages = std::move(stages);

        auto it = candidate_by_node.find(node);
        if (it != candidate_by_node.end()) {
            art.dof_count        = static_cast<uint8_t>(it->second.dof);
            art.fit_residual_rad = it->second.summary.max_residual_rad;
        }
        result.push_back(std::move(art));
    }

    std::sort(result.begin(), result.end(),
              [](const Articulation& a, const Articulation& b) { return a.node < b.node; });

    detail::infer_pointing_vectors(result, skeleton);
    return result;
}

} // namespace ChaCha
