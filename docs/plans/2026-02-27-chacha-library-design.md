# ChaCha Library Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a C++20 static library that deduces joint articulation constraints from skeletal animation data, producing AGI_articulations-compatible output.

**Architecture:** 7-step pipeline: rest-relative conversion, derivative computation, stage segmentation, swing-twist decomposition, stage merging, noise filtering, output assembly. Pure function API with no global state or file I/O. Input via `std::span`, output via `std::vector`.

**Tech Stack:** C++20, GLM (vec3, quat, mat4), CMake 3.15+

---

### Task 1: CMakeLists.txt

**Files:**
- Create: `CMakeLists.txt`

**Step 1: Create the build file**

Follow the dodeedum CMakeLists.txt pattern exactly (no submodules, no extensions).

```cmake
cmake_minimum_required(VERSION 3.15)
project(chacha VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Glob source files
file(GLOB_RECURSE CHACHA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Glob header files for IDE support
file(GLOB_RECURSE CHACHA_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
)

# Create static library
add_library(chacha STATIC
    ${CHACHA_SOURCES}
    ${CHACHA_HEADERS}
)

# Set include directories
target_include_directories(chacha
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Find and link GLM
find_package(glm QUIET)
if(glm_FOUND)
    target_link_libraries(chacha PUBLIC glm::glm)
else()
    message(STATUS "GLM not found via find_package, assuming it's in include path")
endif()

# Set library properties
set_target_properties(chacha PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/chacha.h;${CMAKE_CURRENT_SOURCE_DIR}/include/chacha_types.h;${CMAKE_CURRENT_SOURCE_DIR}/include/chacha_stage.h"
)
```

**Step 2: Commit**

```bash
git add CMakeLists.txt
git commit -m "feat: add CMakeLists.txt for chacha static library"
```

---

### Task 2: StageType enum and Stage struct (`include/chacha_stage.h`)

**Files:**
- Create: `include/chacha_stage.h`

**Step 1: Write the header**

This defines the per-DOF constraint output. Maps directly to AGI_articulations stage types.

```cpp
#ifndef CHACHA_STAGE_H
#define CHACHA_STAGE_H

#include <cstdint>

namespace ChaCha {

// Matches AGI_articulations stage types.
// Each describes one degree of freedom for a joint.
enum class StageType : uint8_t {
    xTranslate,
    yTranslate,
    zTranslate,
    xRotate,
    yRotate,
    zRotate,
    xScale,
    yScale,
    zScale,
};

// Constraint descriptor for a single degree of freedom.
// Units: radians for rotation, meters for translation, unitless for scale.
struct Stage {
    StageType type;
    float min_value{};       // Lower bound (rad, m, or unitless)
    float max_value{};       // Upper bound
    float initial_value{};   // Rest/neutral value (usually 0)
    float max_velocity{};    // Speed limit (rad/s, m/s, or 1/s)
    float max_effort{};      // Force/torque limit (optional, 0 = unset)
};

// Returns "xRotate", "yTranslate", etc.
const char* stage_type_name(StageType type);

} // namespace ChaCha

#endif // CHACHA_STAGE_H
```

**Step 2: Commit**

```bash
git add include/chacha_stage.h
git commit -m "feat: add StageType enum and Stage struct"
```

---

### Task 3: Input/output types (`include/chacha_types.h`)

**Files:**
- Create: `include/chacha_types.h`

**Step 1: Write the header**

Defines all input/output data structures. Uses `std::span` for inputs (caller owns data), `std::vector` for outputs.

```cpp
#ifndef CHACHA_TYPES_H
#define CHACHA_TYPES_H

#include "chacha_stage.h"
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>
#include <span>
#include <string>
#include <vector>
#include <cstdint>

namespace ChaCha {

// How keyframes are interpolated between samples.
enum class InterpolationType : uint8_t {
    Linear,
    Step,
    CubicSpline,
};

// Which transform property this channel animates.
enum class Property : uint8_t {
    Translation,
    Rotation,
    Scale,
};

// One animation channel: keyframes for a single property of a single joint.
// The caller owns all pointed-to data; it must outlive the analyze() call.
struct AnimationChannel {
    int node{-1};                           // Joint/node index in skeleton
    Property property{Property::Rotation};
    InterpolationType interp{InterpolationType::Linear};
    std::span<const float> times;           // Keyframe timestamps (seconds), ascending
    std::span<const float> values;          // Keyframe values:
                                            //   Translation: 3 floats per key (x,y,z)
                                            //   Rotation:    4 floats per key (x,y,z,w)
                                            //   Scale:       3 floats per key (x,y,z)
};

// Skeleton hierarchy: parents array + rest poses.
// parents[i] < i for all i > 0. parents[0] == -1 (root).
struct Skeleton {
    std::span<const int> parents;
    std::span<const glm::quat> rest_rotations;      // Rest pose rotation per joint
    std::span<const glm::vec3> rest_translations;    // Rest pose translation per joint
};

// All constraints for one joint.
struct Articulation {
    int node{-1};
    std::string name;               // Human-readable (e.g., "left_elbow")
    std::vector<Stage> stages;      // One per active degree of freedom
};

// Tuning parameters for the analysis pipeline.
struct Options {
    float rotation_threshold_rad{0.01f};    // Discard rotation DOFs below this
    float translation_threshold_m{0.001f};  // Discard translation DOFs below this
    float scale_threshold{0.01f};           // Discard scale DOFs below this
    bool prioritize_rom_animations{true};   // Prefer channels from ROM-exercise animations
};

} // namespace ChaCha

#endif // CHACHA_TYPES_H
```

**Step 2: Commit**

```bash
git add include/chacha_types.h
git commit -m "feat: add input/output types for animation channels and articulations"
```

---

### Task 4: Public API (`include/chacha.h`)

**Files:**
- Create: `include/chacha.h`

**Step 1: Write the aggregate header with analyze() declaration**

```cpp
#ifndef CHACHA_H
#define CHACHA_H

#include "chacha_types.h"
#include "chacha_stage.h"

namespace ChaCha {

// Analyze animation channels to deduce per-joint articulation constraints.
//
// Examines all keyframes across all channels to determine, for each joint,
// which degrees of freedom are active and what their ranges, velocities,
// and effort limits are.
//
// Joints with no significant motion (below threshold) produce no output.
//
// @param channels  Animation channels (rotation/translation/scale keyframes per joint)
// @param skeleton  Skeleton hierarchy with rest poses
// @param options   Tuning parameters (thresholds, etc.)
// @return One Articulation per joint that has significant motion
std::vector<Articulation> analyze(
    std::span<const AnimationChannel> channels,
    const Skeleton& skeleton,
    const Options& options = {}
);

} // namespace ChaCha

#endif // CHACHA_H
```

**Step 2: Commit**

```bash
git add include/chacha.h
git commit -m "feat: add public API header with analyze() declaration"
```

---

### Task 5: Internal helpers header (`src/chacha_internal.h`)

**Files:**
- Create: `src/chacha_internal.h`

**Step 1: Write internal types shared between implementation files**

```cpp
#ifndef CHACHA_INTERNAL_H
#define CHACHA_INTERNAL_H

#include "chacha_types.h"
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

namespace ChaCha {
namespace detail {

// A single sample in rest-relative space with its timestamp.
struct Sample {
    float time{};
    float value{};          // Single DOF value (rad, m, or unitless)
    float velocity{};       // Derivative (rad/s, m/s, or 1/s)
};

// Samples for one DOF of one joint, extracted from animation data.
struct DofTrack {
    int node{-1};
    StageType type;
    std::vector<Sample> samples;
};

// Intermediate stage before filtering: includes raw statistics.
struct RawStage {
    StageType type;
    float min_value{};
    float max_value{};
    float initial_value{};
    float max_velocity{};
    float max_effort{};

    float range() const { return max_value - min_value; }
};

// ---- Pipeline step functions ----

// Step 1-2: Convert channels to rest-relative DOF tracks with derivatives.
// Rotation channels produce up to 3 tracks (xRotate, yRotate, zRotate) via swing-twist.
// Translation channels produce 3 tracks (xTranslate, yTranslate, zTranslate).
// Scale channels produce 3 tracks (xScale, yScale, zScale).
std::vector<DofTrack> extract_dof_tracks(
    std::span<const AnimationChannel> channels,
    const Skeleton& skeleton
);

// Step 3+5: Segment DOF tracks into raw stages and merge compatible ones.
std::vector<RawStage> segment_and_merge(
    std::span<const DofTrack> tracks_for_joint
);

// Step 6: Filter out stages below noise threshold.
std::vector<Stage> filter_stages(
    std::span<const RawStage> raw_stages,
    const Options& options
);

// Step 4: Decompose a quaternion into swing-twist components.
// twist_axis is typically (0,1,0) for bone-local Y.
// Returns (swing, twist) where q ≈ swing * twist.
struct SwingTwist {
    glm::quat swing;
    glm::quat twist;
};

SwingTwist decompose_swing_twist(const glm::quat& q, const glm::vec3& twist_axis);

// Convert swing quaternion to two rotation angles (xRotate, zRotate).
// Returns {x_angle_rad, z_angle_rad}.
glm::vec2 swing_to_angles(const glm::quat& swing);

// Convert twist quaternion to one rotation angle (yRotate).
float twist_to_angle(const glm::quat& twist, const glm::vec3& twist_axis);

} // namespace detail
} // namespace ChaCha

#endif // CHACHA_INTERNAL_H
```

**Step 2: Commit**

```bash
git add src/chacha_internal.h
git commit -m "feat: add internal types and pipeline function declarations"
```

---

### Task 6: Swing-twist decomposition (`src/chacha_decompose.cpp`)

**Files:**
- Create: `src/chacha_decompose.cpp`

**Step 1: Implement decomposition**

This is the mathematical core. For each rest-relative quaternion:
1. Project onto twist axis to get twist component
2. Remainder is swing component
3. Convert swing to two angles, twist to one angle

```cpp
#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>
#include <cmath>

namespace ChaCha {
namespace detail {

SwingTwist decompose_swing_twist(const glm::quat& q, const glm::vec3& twist_axis)
{
    // Project quaternion's vector part onto twist axis.
    // twist = normalize(q.w, dot(q.xyz, axis) * axis)
    glm::vec3 q_vec(q.x, q.y, q.z);
    float projection = glm::dot(q_vec, twist_axis);
    glm::vec3 twist_vec = projection * twist_axis;

    glm::quat twist(q.w, twist_vec.x, twist_vec.y, twist_vec.z);
    float twist_len = std::sqrt(twist.w * twist.w + twist.x * twist.x
                                + twist.y * twist.y + twist.z * twist.z);

    if (twist_len < 1e-10f) {
        // No twist component: pure swing.
        twist = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    } else {
        twist = twist * (1.0f / twist_len);
    }

    // swing = q * inverse(twist)
    glm::quat swing = q * glm::inverse(twist);

    return {swing, twist};
}

glm::vec2 swing_to_angles(const glm::quat& swing)
{
    // Swing rotates around axes perpendicular to the twist axis.
    // For twist_axis = Y, swing is in the XZ plane.
    // x_angle: rotation around local X axis (pitch)
    // z_angle: rotation around local Z axis (yaw)
    //
    // For a small swing: swing ≈ (1, sin(θx/2), 0, sin(θz/2))
    // Use atan2 for full range.
    float x_angle_rad = 2.0f * std::atan2(swing.x, swing.w);
    float z_angle_rad = 2.0f * std::atan2(swing.z, swing.w);
    return {x_angle_rad, z_angle_rad};
}

float twist_to_angle(const glm::quat& twist, const glm::vec3& twist_axis)
{
    // Twist quaternion: (cos(θ/2), sin(θ/2) * axis)
    glm::vec3 t_vec(twist.x, twist.y, twist.z);
    float sin_half = glm::dot(t_vec, twist_axis);
    float angle_rad = 2.0f * std::atan2(sin_half, twist.w);
    return angle_rad;
}

} // namespace detail
} // namespace ChaCha
```

**Step 2: Commit**

```bash
git add src/chacha_decompose.cpp
git commit -m "feat: implement swing-twist decomposition for rotation channels"
```

---

### Task 7: DOF track extraction (`src/chacha_analyzer.cpp` — steps 1, 2, 4)

**Files:**
- Create: `src/chacha_analyzer.cpp`

**Step 1: Implement extract_dof_tracks and the top-level analyze()**

This orchestrates the full pipeline. `extract_dof_tracks` handles steps 1-2-4:
- For each channel, convert keyframes to rest-relative
- For rotation channels: decompose via swing-twist, produce 3 DOF tracks
- For translation/scale: project onto axes, produce 3 DOF tracks
- Compute finite-difference velocities

```cpp
#include "chacha_internal.h"
#include "chacha.h"
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <cmath>
#include <map>

namespace ChaCha {

// ---- stage_type_name (from chacha_stage.h) ----

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

// ---- Step 1+2+4: Extract DOF tracks ----

static void extract_rotation_tracks(
    const AnimationChannel& channel,
    const Skeleton& skeleton,
    std::vector<DofTrack>& out)
{
    const int node = channel.node;
    const glm::quat& rest_q = skeleton.rest_rotations[node];
    const glm::quat rest_inv = glm::inverse(rest_q);
    const glm::vec3 twist_axis(0.0f, 1.0f, 0.0f); // bone-local Y

    const int num_keys = static_cast<int>(channel.times.size());
    if (num_keys == 0) return;

    DofTrack x_track{node, StageType::xRotate, {}};
    DofTrack y_track{node, StageType::yRotate, {}};
    DofTrack z_track{node, StageType::zRotate, {}};

    x_track.samples.reserve(num_keys);
    y_track.samples.reserve(num_keys);
    z_track.samples.reserve(num_keys);

    int values_per_key = 4; // quaternion: x,y,z,w
    int stride = values_per_key;
    if (channel.interp == InterpolationType::CubicSpline) {
        stride = values_per_key * 3; // in-tangent, value, out-tangent
    }

    float prev_x = 0.0f, prev_y = 0.0f, prev_z = 0.0f;
    float prev_time = 0.0f;

    for (int i = 0; i < num_keys; ++i) {
        const float t = channel.times[i];
        const int offset = i * stride;
        // For CubicSpline, the value is the middle third.
        const int val_offset = (channel.interp == InterpolationType::CubicSpline)
            ? offset + values_per_key : offset;

        const glm::quat key_q(
            channel.values[val_offset + 3], // w
            channel.values[val_offset + 0], // x
            channel.values[val_offset + 1], // y
            channel.values[val_offset + 2]  // z
        );

        // Step 1: rest-relative
        const glm::quat rel_q = rest_inv * key_q;

        // Step 4: swing-twist decomposition
        auto [swing, twist] = decompose_swing_twist(rel_q, twist_axis);
        glm::vec2 swing_angles = swing_to_angles(swing);
        float twist_angle = twist_to_angle(twist, twist_axis);

        float x_val = swing_angles.x;
        float y_val = twist_angle;
        float z_val = swing_angles.y;

        // Step 2: finite-difference velocity
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
    } else { // Scale
        rest_val = glm::vec3(1.0f); // rest scale is identity
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

// ---- Top-level pipeline ----

std::vector<Articulation> analyze(
    std::span<const AnimationChannel> channels,
    const Skeleton& skeleton,
    const Options& options)
{
    // Steps 1+2+4: Extract DOF tracks from all channels.
    auto all_tracks = detail::extract_dof_tracks(channels, skeleton);

    // Group tracks by node.
    std::map<int, std::vector<const detail::DofTrack*>> tracks_by_node;
    for (const auto& track : all_tracks) {
        tracks_by_node[track.node].push_back(&track);
    }

    // Process each joint.
    std::vector<Articulation> result;
    const int num_joints = static_cast<int>(skeleton.parents.size());

    for (auto& [node, node_tracks] : tracks_by_node) {
        if (node < 0 || node >= num_joints) continue;

        // Collect DofTrack spans for this joint.
        // We need contiguous storage for span, so copy pointers' data.
        std::vector<detail::DofTrack> joint_tracks;
        joint_tracks.reserve(node_tracks.size());
        for (const auto* tp : node_tracks) {
            joint_tracks.push_back(*tp);
        }

        // Steps 3+5: Segment and merge.
        auto raw_stages = detail::segment_and_merge(
            std::span<const detail::DofTrack>(joint_tracks));

        // Step 6: Filter.
        auto stages = detail::filter_stages(
            std::span<const detail::RawStage>(raw_stages), options);

        // Step 7: Produce articulation if any stages survived.
        if (!stages.empty()) {
            Articulation art;
            art.node = node;
            art.name = "joint_" + std::to_string(node);
            art.stages = std::move(stages);
            result.push_back(std::move(art));
        }
    }

    // Sort by node index for deterministic output.
    std::sort(result.begin(), result.end(),
        [](const Articulation& a, const Articulation& b) {
            return a.node < b.node;
        });

    return result;
}

} // namespace ChaCha
```

**Step 2: Commit**

```bash
git add src/chacha_analyzer.cpp
git commit -m "feat: implement DOF track extraction and top-level analyze pipeline"
```

---

### Task 8: Stage segmentation and merging (`src/chacha_segment.cpp` — steps 3, 5)

**Files:**
- Create: `src/chacha_segment.cpp`

**Step 1: Implement segment_and_merge**

For each DOF type present in the joint's tracks:
- Collect all samples across all channels (union of animations)
- Compute the overall min/max range and max velocity
- Merge tracks of the same StageType by taking the union of ranges and max of velocities

```cpp
#include "chacha_internal.h"
#include <algorithm>
#include <cmath>
#include <map>

namespace ChaCha {
namespace detail {

std::vector<RawStage> segment_and_merge(
    std::span<const DofTrack> tracks_for_joint)
{
    // Group all tracks by StageType and merge.
    // Multiple animation channels may contribute to the same DOF of the same joint.
    // We take the union (widest observed range, highest velocity).
    std::map<StageType, RawStage> merged;

    for (const auto& track : tracks_for_joint) {
        auto it = merged.find(track.type);
        if (it == merged.end()) {
            RawStage stage;
            stage.type = track.type;
            stage.min_value = 0.0f;
            stage.max_value = 0.0f;
            stage.initial_value = 0.0f;
            stage.max_velocity = 0.0f;
            stage.max_effort = 0.0f;
            merged.emplace(track.type, stage);
            it = merged.find(track.type);
        }

        RawStage& stage = it->second;

        for (const auto& sample : track.samples) {
            stage.min_value = std::min(stage.min_value, sample.value);
            stage.max_value = std::max(stage.max_value, sample.value);
            stage.max_velocity = std::max(stage.max_velocity, std::abs(sample.velocity));
        }
    }

    // Convert to vector.
    std::vector<RawStage> result;
    result.reserve(merged.size());
    for (auto& [type, stage] : merged) {
        result.push_back(std::move(stage));
    }

    return result;
}

} // namespace detail
} // namespace ChaCha
```

**Step 2: Commit**

```bash
git add src/chacha_segment.cpp
git commit -m "feat: implement stage segmentation and merging across animations"
```

---

### Task 9: Noise filtering (`src/chacha_filter.cpp` — step 6)

**Files:**
- Create: `src/chacha_filter.cpp`

**Step 1: Implement filter_stages**

Discard stages where the observed range of motion is below the noise threshold. Convert surviving RawStages to final Stage structs.

```cpp
#include "chacha_internal.h"
#include <cmath>

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
    }
    return 0.0f;
}

std::vector<Stage> filter_stages(
    std::span<const RawStage> raw_stages,
    const Options& options)
{
    std::vector<Stage> result;
    result.reserve(raw_stages.size());

    for (const auto& raw : raw_stages) {
        float range = raw.range();
        float threshold = threshold_for_type(raw.type, options);

        if (range < threshold) continue;

        Stage stage;
        stage.type = raw.type;
        stage.min_value = raw.min_value;
        stage.max_value = raw.max_value;
        stage.initial_value = raw.initial_value;
        stage.max_velocity = raw.max_velocity;
        stage.max_effort = raw.max_effort;
        result.push_back(stage);
    }

    return result;
}

} // namespace detail
} // namespace ChaCha
```

**Step 2: Commit**

```bash
git add src/chacha_filter.cpp
git commit -m "feat: implement noise filtering with configurable thresholds"
```

---

### Task 10: Build verification

**Step 1: Verify the library compiles**

From the chacha module directory:

```bash
mkdir -p build && cd build
cmake .. && make
```

Expected: Successful compilation of `libchacha.a` with no errors.

**Step 2: Fix any compilation issues**

If there are compiler errors, fix them. Common issues:
- Missing includes
- GLM header paths
- C++20 span support flags

**Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: resolve build issues"
```

---

### Task 11: Integration with parent project

**Files:**
- Modify: `/mnt/Passport/Libraries/Spehleon/tonton-example/CMakeLists.txt`

**Step 1: Add chacha subdirectory to the parent build**

Add after the tonton subdirectory line:

```cmake
# chacha (Apache-2.0) - Joint articulation constraint analysis
add_subdirectory(modules/chacha)
```

And update the build summary to include chacha.

**Step 2: Verify parent project builds with chacha included**

```bash
cd /mnt/Passport/Libraries/Spehleon/tonton-example
mkdir -p build && cd build
cmake .. && make
```

**Step 3: Commit**

```bash
git add CMakeLists.txt
git commit -m "feat: integrate chacha module into parent build"
```

Note: This commit is in the parent repo, not the chacha submodule.

---

### Summary of files created

```
chacha/
├── CMakeLists.txt
├── include/
│   ├── chacha.h              (public API)
│   ├── chacha_types.h        (input/output structs)
│   └── chacha_stage.h        (StageType enum, Stage struct)
└── src/
    ├── chacha_internal.h     (internal types, pipeline function decls)
    ├── chacha_analyzer.cpp   (steps 1+2+4+7, top-level analyze())
    ├── chacha_decompose.cpp  (step 4, swing-twist math)
    ├── chacha_segment.cpp    (steps 3+5, segmentation and merging)
    └── chacha_filter.cpp     (step 6, noise filtering)
```
