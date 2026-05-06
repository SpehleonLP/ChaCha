#include "chacha_internal.h"
#include <glm/geometric.hpp>
#include <cmath>
#include <vector>

namespace ChaCha {
namespace detail {

namespace {

constexpr float kAxisDotThreshold = 0.985f; // ~10 deg
constexpr float kDominantAxisFraction = 0.66f;

struct AxisCandidate {
    glm::vec3 dir;
    float     score; // |dot| against the candidate axis
};

glm::vec3 normalize_safe(glm::vec3 v, glm::vec3 fallback = {0, 1, 0})
{
    float len2 = glm::dot(v, v);
    if (len2 < 1e-12f) return fallback;
    return v * (1.0f / std::sqrt(len2));
}

// Returns axis index (0=X,1=Y,2=Z) and sign (+1/-1) of the closest principal
// axis for `dir`, plus |dot| with that axis.
struct AxisMatch { int axis; float sign; float dot; };

AxisMatch closest_axis(glm::vec3 dir)
{
    AxisMatch best{1, 1.0f, 0.0f};
    float comps[3] = {dir.x, dir.y, dir.z};
    for (int i = 0; i < 3; ++i) {
        float a = std::fabs(comps[i]);
        if (a > best.dot) {
            best.dot = a;
            best.axis = i;
            best.sign = comps[i] >= 0.0f ? 1.0f : -1.0f;
        }
    }
    return best;
}

glm::vec3 axis_vector(int axis, float sign)
{
    glm::vec3 v(0.0f);
    v[axis] = sign;
    return v;
}

} // namespace

void infer_pointing_vectors(
    std::vector<Articulation>& articulations,
    const Skeleton& skeleton)
{
    if (articulations.empty()) return;

    const int num_nodes = static_cast<int>(skeleton.parents.size());

    // Build child list and per-node child count.
    std::vector<std::vector<int>> children(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        int p = skeleton.parents[i];
        if (p >= 0 && p < num_nodes) {
            children[p].push_back(i);
        }
    }

    // Index articulations by node for quick parent lookups.
    std::vector<int> art_index_by_node(num_nodes, -1);
    for (size_t i = 0; i < articulations.size(); ++i) {
        int n = articulations[i].node;
        if (n >= 0 && n < num_nodes) art_index_by_node[n] = static_cast<int>(i);
    }

    // ----- Loop 1: gather single-child candidates and look for a dominant axis.
    std::vector<AxisCandidate> single_child_dirs;
    single_child_dirs.reserve(articulations.size());

    for (auto& art : articulations) {
        if (art.node < 0 || art.node >= num_nodes) continue;
        const auto& kids = children[art.node];
        if (kids.size() != 1) continue;

        glm::vec3 dir = normalize_safe(skeleton.rest_translations[kids[0]]);
        single_child_dirs.push_back({dir, 0.0f});
    }

    int axis_votes[3] = {0, 0, 0};
    float axis_sign_sum[3] = {0.0f, 0.0f, 0.0f};
    int total_single = static_cast<int>(single_child_dirs.size());

    for (auto& cand : single_child_dirs) {
        AxisMatch m = closest_axis(cand.dir);
        if (m.dot >= kAxisDotThreshold) {
            axis_votes[m.axis]++;
            axis_sign_sum[m.axis] += m.sign;
        }
    }

    int winning_axis = -1;
    float winning_sign = 1.0f;
    if (total_single > 0) {
        int best_votes = 0;
        for (int i = 0; i < 3; ++i) {
            if (axis_votes[i] > best_votes) {
                best_votes = axis_votes[i];
                winning_axis = i;
                winning_sign = axis_sign_sum[i] >= 0.0f ? 1.0f : -1.0f;
            }
        }
        if (best_votes < static_cast<int>(std::ceil(total_single * kDominantAxisFraction))) {
            winning_axis = -1;
        }
    }

    if (winning_axis >= 0) {
        glm::vec3 v = axis_vector(winning_axis, winning_sign);
        for (auto& art : articulations) art.pointing_vector = v;
        return;
    }

    // ----- Loop 2: per-joint mean of child directions for joints with children.
    std::vector<bool> assigned(articulations.size(), false);

    for (size_t i = 0; i < articulations.size(); ++i) {
        auto& art = articulations[i];
        if (art.node < 0 || art.node >= num_nodes) continue;
        const auto& kids = children[art.node];
        if (kids.empty()) continue;

        glm::vec3 acc(0.0f);
        int count = 0;
        for (int k : kids) {
            glm::vec3 d = skeleton.rest_translations[k];
            if (glm::dot(d, d) < 1e-12f) continue;
            acc += normalize_safe(d);
            ++count;
        }
        if (count == 0) continue;
        art.pointing_vector = normalize_safe(acc);
        assigned[i] = true;
    }

    // ----- Loop 3: leaves inherit from nearest articulated ancestor.
    for (size_t i = 0; i < articulations.size(); ++i) {
        if (assigned[i]) continue;
        auto& art = articulations[i];
        if (art.node < 0 || art.node >= num_nodes) continue;

        int p = skeleton.parents[art.node];
        glm::vec3 inherited(0.0f, 1.0f, 0.0f);
        while (p >= 0 && p < num_nodes) {
            int idx = art_index_by_node[p];
            if (idx >= 0 && assigned[idx]) {
                inherited = articulations[idx].pointing_vector;
                break;
            }
            p = skeleton.parents[p];
        }
        art.pointing_vector = inherited;
    }
}

} // namespace detail
} // namespace ChaCha
