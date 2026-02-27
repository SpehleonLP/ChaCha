#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>
#include <cmath>

namespace ChaCha {
namespace detail {

SwingTwist decompose_swing_twist(const glm::quat& q, const glm::vec3& twist_axis)
{
    // Project quaternion's vector part onto twist axis.
    glm::vec3 q_vec(q.x, q.y, q.z);
    float projection = glm::dot(q_vec, twist_axis);
    glm::vec3 twist_vec = projection * twist_axis;

    glm::quat twist(q.w, twist_vec.x, twist_vec.y, twist_vec.z);
    float twist_len = std::sqrt(twist.w * twist.w + twist.x * twist.x
                                + twist.y * twist.y + twist.z * twist.z);

    if (twist_len < 1e-10f) {
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
    float x_angle_rad = 2.0f * std::atan2(swing.x, swing.w);
    float z_angle_rad = 2.0f * std::atan2(swing.z, swing.w);
    return {x_angle_rad, z_angle_rad};
}

float twist_to_angle(const glm::quat& twist, const glm::vec3& twist_axis)
{
    glm::vec3 t_vec(twist.x, twist.y, twist.z);
    float sin_half = glm::dot(t_vec, twist_axis);
    float angle_rad = 2.0f * std::atan2(sin_half, twist.w);
    return angle_rad;
}

} // namespace detail
} // namespace ChaCha
