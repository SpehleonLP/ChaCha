#ifndef CHACHA_TYPES_H
#define CHACHA_TYPES_H

#include "chacha_stage.h"
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>

namespace ChaCha {

enum class InterpolationType : uint8_t {
    Linear,
    Step,
    CubicSpline,
};

enum class Property : uint8_t {
    Translation,
    Rotation,
    Scale,
};

// One animation clip, referenced by AnimationChannel::animation.
//
// LIFETIME: `name` is a non-owning view. The caller must keep the backing
// storage alive for the duration of the analyze() call (and, if the caller
// inspects it afterward for its own purposes, for as long as it does so --
// ChaCha itself never retains it past the call). A bridge layer that builds
// animation names on the fly (e.g. from a glTF JSON string) must keep owned
// storage for each name alive at least until analyze() returns; handing
// analyze() a view into a temporary is a use-after-free that nothing in
// this library can detect.
struct Animation {
    std::string_view name;   // used to auto-detect "AGI " configuration animations
};

struct AnimationChannel {
    int node{-1};
    int animation{0};                    // index into the animations span
    Property property{Property::Rotation};
    InterpolationType interp{InterpolationType::Linear};
    std::span<const float> times;
    std::span<const float> values;
};

struct Skeleton {
    std::span<const int>       parents;
    std::span<const glm::quat> rest_rotations;
    std::span<const glm::vec3> rest_translations;
    std::span<const glm::vec3> rest_scales;   // empty means all-ones; never assume populated
};

struct Articulation {
    int node{-1};
    std::string name;
    std::vector<Stage> stages;
    glm::vec3 pointing_vector{0.0f, 1.0f, 0.0f};
    uint8_t dof_count{0};
    float   fit_residual_rad{0.0f};
};

struct Options {
    float rotation_threshold_rad{0.01f};
    float translation_threshold_m{0.001f};
    float scale_threshold{0.01f};
    bool prioritize_rom_animations{true};
    float resample_rate_hz{60.0f};
    int derivative_window{5};
    // Acceptance gate for reduced-DOF (1- or 2-axis) search candidates: a
    // candidate is only admissible if its worst observed fit residual (see
    // residual_one_dof/residual_two_dof) is at or below this bound.
    // Full-rank (3-DOF) chart candidates always reconstruct exactly and are
    // not subject to this gate.
    float max_fit_residual_rad{0.02f};
};

// One noteworthy event encountered while analyzing a joint's animation
// data, reported through analyze()'s optional `diagnostics` out-parameter.
struct Diagnostic {
    int node{-1};
    int animation{-1};
    enum Kind {
        NonUnitQuaternion,   // a rotation keyframe was not (near) unit length
        EmptyChannel,        // a channel had zero keyframes
        MalformedValues,     // values span size didn't match times/stride
    } kind{EmptyChannel};
};

} // namespace ChaCha

#endif // CHACHA_TYPES_H
