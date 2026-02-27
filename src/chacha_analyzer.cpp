#include "chacha_internal.h"
#include "chacha.h"
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
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

} // namespace detail

std::vector<Articulation> analyze(
    std::span<const AnimationChannel> channels,
    const Skeleton& skeleton,
    const Options& options)
{
    auto all_tracks = detail::extract_dof_tracks(channels, skeleton);

    std::map<int, std::vector<const detail::DofTrack*>> tracks_by_node;
    for (const auto& track : all_tracks) {
        tracks_by_node[track.node].push_back(&track);
    }

    std::vector<Articulation> result;
    const int num_joints = static_cast<int>(skeleton.parents.size());

    for (auto& [node, node_tracks] : tracks_by_node) {
        if (node < 0 || node >= num_joints) continue;

        std::vector<detail::DofTrack> joint_tracks;
        joint_tracks.reserve(node_tracks.size());
        for (const auto* tp : node_tracks) {
            joint_tracks.push_back(*tp);
        }

        auto raw_stages = detail::segment_and_merge(
            std::span<const detail::DofTrack>(joint_tracks));

        auto stages = detail::filter_stages(
            std::span<const detail::RawStage>(raw_stages), options);

        if (!stages.empty()) {
            Articulation art;
            art.node = node;
            art.name = "joint_" + std::to_string(node);
            art.stages = std::move(stages);
            result.push_back(std::move(art));
        }
    }

    std::sort(result.begin(), result.end(),
        [](const Articulation& a, const Articulation& b) {
            return a.node < b.node;
        });

    return result;
}

} // namespace ChaCha
