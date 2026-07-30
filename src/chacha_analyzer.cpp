#include "chacha_internal.h"
#include "chacha.h"
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>

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
    }
    return "unknown";
}

namespace detail {

static void extract_rotation_tracks(
    const AnimationChannel& channel,
    const Skeleton& skeleton,
    std::vector<DofTrack>& out)
{
    const int node = channel.node;
    const glm::quat& rest_q = skeleton.rest_rotations[node];
    const glm::quat rest_inv = glm::inverse(rest_q);
    const glm::vec3 twist_axis(0.0f, 1.0f, 0.0f);

    const int num_keys = static_cast<int>(channel.times.size());
    if (num_keys == 0) return;

    DofTrack x_track{node, StageType::xRotate, {}};
    DofTrack y_track{node, StageType::yRotate, {}};
    DofTrack z_track{node, StageType::zRotate, {}};

    x_track.samples.reserve(num_keys);
    y_track.samples.reserve(num_keys);
    z_track.samples.reserve(num_keys);

    int values_per_key = 4;
    int stride = values_per_key;
    if (channel.interp == InterpolationType::CubicSpline) {
        stride = values_per_key * 3;
    }

    float prev_x = 0.0f, prev_y = 0.0f, prev_z = 0.0f;
    float prev_time = 0.0f;

    for (int i = 0; i < num_keys; ++i) {
        const float t = channel.times[i];
        const int offset = i * stride;
        const int val_offset = (channel.interp == InterpolationType::CubicSpline)
            ? offset + values_per_key : offset;

        const glm::quat key_q(
            channel.values[val_offset + 3],
            channel.values[val_offset + 0],
            channel.values[val_offset + 1],
            channel.values[val_offset + 2]
        );

        const glm::quat rel_q = rest_inv * key_q;

        auto [swing, twist] = decompose_swing_twist(rel_q, twist_axis);
        glm::vec2 swing_angles = swing_to_angles(swing);
        float twist_angle = twist_to_angle(twist, twist_axis);

        float x_val = swing_angles.x;
        float y_val = twist_angle;
        float z_val = swing_angles.y;

        if (i > 0) {
            auto unwrap = [](float cur, float prev) -> float {
                float d = cur - prev;
                if (d > 3.14159265f) cur -= 6.28318530f;
                else if (d < -3.14159265f) cur += 6.28318530f;
                return cur;
            };
            x_val = unwrap(x_val, prev_x);
            y_val = unwrap(y_val, prev_y);
            z_val = unwrap(z_val, prev_z);
        }

        float x_vel = 0.0f, y_vel = 0.0f, z_vel = 0.0f;
        if (i > 0) {
            float dt = t - prev_time;
            if (dt > 1e-10f) {
                x_vel = (x_val - prev_x) / dt;
                y_vel = (y_val - prev_y) / dt;
                z_vel = (z_val - prev_z) / dt;
            }
        }

        x_track.samples.push_back({t, x_val, x_vel});
        y_track.samples.push_back({t, y_val, y_vel});
        z_track.samples.push_back({t, z_val, z_vel});

        prev_x = x_val; prev_y = y_val; prev_z = z_val;
        prev_time = t;
    }

    out.push_back(std::move(x_track));
    out.push_back(std::move(y_track));
    out.push_back(std::move(z_track));
}

static void extract_vec3_tracks(
    const AnimationChannel& channel,
    const Skeleton& skeleton,
    Property property,
    std::vector<DofTrack>& out)
{
    const int node = channel.node;
    const int num_keys = static_cast<int>(channel.times.size());
    if (num_keys == 0) return;

    glm::vec3 rest_val(0.0f);
    StageType types[3];

    if (property == Property::Translation) {
        rest_val = skeleton.rest_translations[node];
        types[0] = StageType::xTranslate;
        types[1] = StageType::yTranslate;
        types[2] = StageType::zTranslate;
    } else {
        rest_val = glm::vec3(1.0f);
        types[0] = StageType::xScale;
        types[1] = StageType::yScale;
        types[2] = StageType::zScale;
    }

    DofTrack tracks[3] = {
        {node, types[0], {}},
        {node, types[1], {}},
        {node, types[2], {}},
    };

    for (auto& tr : tracks) tr.samples.reserve(num_keys);

    int values_per_key = 3;
    int stride = values_per_key;
    if (channel.interp == InterpolationType::CubicSpline) {
        stride = values_per_key * 3;
    }

    float prev[3] = {0.0f, 0.0f, 0.0f};
    float prev_time = 0.0f;

    for (int i = 0; i < num_keys; ++i) {
        const float t = channel.times[i];
        const int offset = i * stride;
        const int val_offset = (channel.interp == InterpolationType::CubicSpline)
            ? offset + values_per_key : offset;

        float vals[3] = {
            channel.values[val_offset + 0] - rest_val.x,
            channel.values[val_offset + 1] - rest_val.y,
            channel.values[val_offset + 2] - rest_val.z,
        };

        for (int axis = 0; axis < 3; ++axis) {
            float vel = 0.0f;
            if (i > 0) {
                float dt = t - prev_time;
                if (dt > 1e-10f) {
                    vel = (vals[axis] - prev[axis]) / dt;
                }
            }
            tracks[axis].samples.push_back({t, vals[axis], vel});
            prev[axis] = vals[axis];
        }
        prev_time = t;
    }

    for (auto& tr : tracks) out.push_back(std::move(tr));
}

std::vector<DofTrack> extract_dof_tracks(
    std::span<const AnimationChannel> channels,
    const Skeleton& skeleton)
{
    std::vector<DofTrack> tracks;
    tracks.reserve(channels.size() * 3);

    for (const auto& ch : channels) {
        if (ch.node < 0 || ch.node >= static_cast<int>(skeleton.parents.size()))
            continue;

        switch (ch.property) {
        case Property::Rotation:
            extract_rotation_tracks(ch, skeleton, tracks);
            break;
        case Property::Translation:
        case Property::Scale:
            extract_vec3_tracks(ch, skeleton, ch.property, tracks);
            break;
        }
    }

    return tracks;
}

void optimize_stage_order(
    std::vector<Stage>& stages,
    const std::vector<DofTrack>& rotation_tracks,
    std::span<const AnimationChannel> channels,
    const Skeleton& skeleton,
    int node)
{
    // Separate rotation stages from non-rotation stages
    std::vector<size_t> rot_indices;
    for (size_t i = 0; i < stages.size(); ++i) {
        auto t = stages[i].type;
        if (t == StageType::xRotate || t == StageType::yRotate || t == StageType::zRotate) {
            rot_indices.push_back(i);
        }
    }

    if (rot_indices.size() <= 1) return;

    // Build map from StageType to DofTrack samples
    std::map<StageType, const DofTrack*> track_map;
    for (const auto& tr : rotation_tracks) {
        if (tr.node == node) {
            track_map[tr.type] = &tr;
        }
    }

    // Collect rotation channels for this node to recompute ground-truth rel_q
    std::vector<const AnimationChannel*> rot_channels;
    for (const auto& ch : channels) {
        if (ch.node == node && ch.property == Property::Rotation) {
            rot_channels.push_back(&ch);
        }
    }
    if (rot_channels.empty()) return;

    const glm::quat rest_inv = glm::inverse(skeleton.rest_rotations[node]);
    const glm::vec3 twist_axis(0.0f, 1.0f, 0.0f);

    // Build permutation indices over rotation stages
    std::vector<size_t> perm(rot_indices.size());
    std::iota(perm.begin(), perm.end(), 0);

    float best_error = std::numeric_limits<float>::max();
    std::vector<size_t> best_perm = perm;

    // Helper: axis rotation from stage type and angle
    auto axis_rotation = [](StageType type, float angle_rad) -> glm::quat {
        float half = angle_rad * 0.5f;
        float s = std::sin(half);
        float c = std::cos(half);
        switch (type) {
        case StageType::xRotate: return glm::quat(c, s, 0.0f, 0.0f);
        case StageType::yRotate: return glm::quat(c, 0.0f, s, 0.0f);
        case StageType::zRotate: return glm::quat(c, 0.0f, 0.0f, s);
        default: return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        }
    };

    do {
        float total_error = 0.0f;

        // Evaluate over all rotation channels for this node
        for (const auto* ch : rot_channels) {
            int num_keys = static_cast<int>(ch->times.size());
            int values_per_key = 4;
            int stride = values_per_key;
            if (ch->interp == InterpolationType::CubicSpline) {
                stride = values_per_key * 3;
            }

            for (int i = 0; i < num_keys; ++i) {
                float t = ch->times[i];
                int offset = i * stride;
                int val_offset = (ch->interp == InterpolationType::CubicSpline)
                    ? offset + values_per_key : offset;

                glm::quat key_q(
                    ch->values[val_offset + 3],
                    ch->values[val_offset + 0],
                    ch->values[val_offset + 1],
                    ch->values[val_offset + 2]
                );
                glm::quat rel_q = rest_inv * key_q;

                // Decompose to get per-axis angles
                auto [swing, twist] = decompose_swing_twist(rel_q, twist_axis);
                glm::vec2 swing_angles = swing_to_angles(swing);
                float twist_angle = twist_to_angle(twist, twist_axis);

                // Map stage types to their angles
                std::map<StageType, float> angle_map;
                angle_map[StageType::xRotate] = swing_angles.x;
                angle_map[StageType::yRotate] = twist_angle;
                angle_map[StageType::zRotate] = swing_angles.y;

                // Compose in permutation order
                glm::quat composed(1.0f, 0.0f, 0.0f, 0.0f);
                for (size_t pi = 0; pi < perm.size(); ++pi) {
                    StageType st = stages[rot_indices[perm[pi]]].type;
                    auto it = angle_map.find(st);
                    if (it != angle_map.end()) {
                        composed = composed * axis_rotation(st, it->second);
                    }
                }

                // Angular distance
                float d = std::abs(glm::dot(composed, rel_q));
                if (d > 1.0f) d = 1.0f;
                total_error += 2.0f * std::acos(d);
            }
        }

        if (total_error < best_error) {
            best_error = total_error;
            best_perm = perm;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    // Reorder rotation stages according to best permutation
    std::vector<Stage> reordered_rot;
    reordered_rot.reserve(rot_indices.size());
    for (size_t pi = 0; pi < best_perm.size(); ++pi) {
        reordered_rot.push_back(stages[rot_indices[best_perm[pi]]]);
    }

    // Replace rotation stages in-place
    for (size_t i = 0; i < rot_indices.size(); ++i) {
        stages[rot_indices[i]] = reordered_rot[i];
    }
}

} // namespace detail

std::vector<Articulation> analyze(
    std::span<const AnimationChannel> channels,
    const Skeleton& skeleton,
    const Options& options)
{
    // TODO(Task 10): rebuild analyze() on top of resolve_branches/anchor_trajectory/
    // summarise instead of the retired segment_and_merge pipeline. Stubbed to keep
    // the tree building in the interim (Task 6).
    (void)channels;
    (void)skeleton;
    (void)options;
    return {};
}

} // namespace ChaCha
