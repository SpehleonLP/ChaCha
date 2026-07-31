#include "chacha_internal.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace ChaCha {
namespace detail {

static float threshold_for_type(StageType type, const Options& options)
{
    switch (type) {
    case StageType::xRotate:
    case StageType::yRotate:
    case StageType::zRotate:
        return options.rotation_threshold_rad;

    case StageType::xTranslate:
    case StageType::yTranslate:
    case StageType::zTranslate:
        return options.translation_threshold_m;

    case StageType::xScale:
    case StageType::yScale:
    case StageType::zScale:
        return options.scale_threshold;

    case StageType::Invalid:
        // Invalid is the sentinel for an unused Candidate slot; it must
        // never reach filter_stages, let alone survive it into emitted
        // output. Returning 0 here (the old fallthrough default) would be
        // actively wrong: since range() >= 0 always, `range < 0` is false,
        // so a stray Invalid-typed RawStage would sail through the filter
        // unfiltered instead of being caught. Return +inf so range() can
        // never clear the bar, guaranteeing it is always dropped even if
        // it does turn up here as a bug upstream.
        return std::numeric_limits<float>::infinity();
    }
    return 0.0f;
}

static bool is_rotation_type(StageType type)
{
    return type == StageType::xRotate ||
           type == StageType::yRotate ||
           type == StageType::zRotate;
}

std::vector<Stage> filter_stages(
    std::span<const RawStage> raw_stages,
    const Options& options)
{
    std::vector<Stage> result;
    result.reserve(raw_stages.size());

    // Rotation stages from a single joint's chart are an ordered, inseparable
    // triple: a zero-range stage sitting between two varying stages of an
    // Euler/proper-Euler composition (e.g. a tilted-axis hinge solved as
    // XYX) is a fixed frame change, not "no information" -- dropping it
    // alone while keeping its neighbour corrupts what the neighbour's range
    // even means. So rotation stages are filtered per JOINT, not per stage:
    // if ANY rotation stage produced for this joint clears the noise
    // threshold on range, every rotation stage that joint produced is kept,
    // including zero-range ones. If none clears it, all are dropped -- this
    // preserves the existing "locked joint" behaviour exactly (a joint whose
    // motion is genuine noise below threshold on every axis still emits no
    // rotation stages, regardless of any stage's constant *value*).
    //
    // This is safe to decide with a single flat scan here because every
    // rotation-typed RawStage passed into one filter_stages call originates
    // from exactly one Candidate/chart for one node (chacha_analyzer.cpp
    // calls filter_stages once per node, and a node has at most one
    // rotational Candidate) -- there is no cross-joint or cross-chart
    // mixing to disentangle.
    //
    // Translation and scale stages remain independent per axis and keep the
    // original per-stage threshold filtering.
    bool keep_rotation = false;
    for (const auto& raw : raw_stages) {
        if (!is_rotation_type(raw.type)) continue;
        if (raw.range() >= threshold_for_type(raw.type, options)) {
            keep_rotation = true;
            break;
        }
    }

    for (const auto& raw : raw_stages) {
        const bool rotation = is_rotation_type(raw.type);
        if (rotation) {
            if (!keep_rotation) continue;
        } else {
            const float threshold = threshold_for_type(raw.type, options);
            if (raw.range() < threshold) continue;
        }

        Stage stage;
        stage.type             = raw.type;
        stage.min_value        = raw.min_value;
        stage.max_value        = raw.max_value;
        stage.initial_value    = std::clamp(raw.initial_value, raw.min_value, raw.max_value);
        stage.max_velocity     = raw.max_velocity;
        stage.max_acceleration = raw.max_acceleration;
        result.push_back(stage);
    }

    return result;
}

} // namespace detail
} // namespace ChaCha
