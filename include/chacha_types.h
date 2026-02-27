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
};

struct Options {
    float rotation_threshold_rad{0.01f};
    float translation_threshold_m{0.001f};
    float scale_threshold{0.01f};
    bool prioritize_rom_animations{true};
};

} // namespace ChaCha

#endif // CHACHA_TYPES_H
