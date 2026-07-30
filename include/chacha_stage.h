#ifndef CHACHA_STAGE_H
#define CHACHA_STAGE_H

#include <cstdint>

namespace ChaCha {

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
    // Sentinel for "this slot carries no stage" (e.g. an unused trailing
    // slot in Candidate::stage, see chacha_internal.h). Never appears on a
    // real Stage produced by analyze(); appended after the real AGI values
    // so it cannot collide with a valid StageType.
    Invalid,
};

struct Stage {
    StageType type;
    float min_value{};
    float max_value{};
    float initial_value{};
    float max_velocity{};
    float max_effort{};
};

const char* stage_type_name(StageType type);

} // namespace ChaCha

#endif // CHACHA_STAGE_H
