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

struct AnimationChannel {
    int node{-1};
    Property property{Property::Rotation};
    InterpolationType interp{InterpolationType::Linear};
    std::span<const float> times;
    std::span<const float> values;
};

struct Skeleton {
    std::span<const int> parents;
    std::span<const glm::quat> rest_rotations;
    std::span<const glm::vec3> rest_translations;
};

struct Articulation {
    int node{-1};
    std::string name;
    std::vector<Stage> stages;
    glm::vec3 pointing_vector{0.0f, 1.0f, 0.0f};
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

} // namespace ChaCha

#endif // CHACHA_TYPES_H
