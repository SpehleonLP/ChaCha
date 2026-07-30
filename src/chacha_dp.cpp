#include "chacha_internal.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace ChaCha {
namespace detail {

namespace {
constexpr float kPi    = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;
} // namespace

float principal_difference(float current, float previous)
{
    float d = current - previous;
    return d - kTwoPi * std::round(d / kTwoPi);
}

Trajectory resolve_branches(
    std::span<const glm::quat> rel,
    std::span<const float>     times,
    const Chart&               chart)
{
    Trajectory out;

    // Defined policy for mismatched lengths: process only as many frames
    // as both spans actually provide. `rel` and `times` are expected to be
    // parallel arrays produced by the same caller; silently truncating to
    // the shorter one avoids reading past the end of either span while
    // still producing a usable (if partial) trajectory rather than
    // crashing or fabricating timestamps.
    const int n = static_cast<int>(std::min(rel.size(), times.size()));
    if (n == 0) return out;

    // Two candidate solutions per frame.
    std::vector<EulerSolution> cand(static_cast<size_t>(n) * 2);
    std::vector<float>         cond(static_cast<size_t>(n) * 2);
    for (int t = 0; t < n; ++t) {
        EulerSolution a = solve_euler(rel[t], chart);
        EulerSolution b = alternate_branch(a, chart);
        cand[t * 2 + 0] = a;  cond[t * 2 + 0] = chart_conditioning(a, chart);
        cand[t * 2 + 1] = b;  cond[t * 2 + 1] = chart_conditioning(b, chart);
    }

    if (n == 1) {
        out.time.assign(times.begin(), times.begin() + 1);
        // With a single frame there is no history to disambiguate branches
        // via transition cost, but the two branches are not equally good
        // readings: one may sit near the rest pose and the other near its
        // chart-flip antipode (e.g. (179,89,179) deg vs (-1,91,-1) deg have
        // squared rest-distances 21.9 vs ~0.0006). Pick whichever branch is
        // closer to the rest pose (all-zero angles) -- the same criterion
        // the n>1 path below applies to a whole reconstructed trajectory,
        // collapsed to a single frame.
        float d0 = 0.0f, d1 = 0.0f;
        for (int a = 0; a < 3; ++a) {
            float w0 = principal_difference(cand[0].angle[a], 0.0f);
            float w1 = principal_difference(cand[1].angle[a], 0.0f);
            d0 += w0 * w0;
            d1 += w1 * w1;
        }
        const int best = (d1 < d0) ? 1 : 0;
        for (int a = 0; a < 3; ++a) out.angle[a].assign(1, cand[best].angle[a]);
        out.worst_conditioning = cond[best];
        return out;
    }

    // Viterbi. Cost of a transition is the summed squared principal difference
    // across the three axes; the 2pi lift is forced, so branch is the only
    // choice. `reachable` tracks which lattice nodes have a finite cost —
    // using an explicit flag instead of comparing against a float sentinel
    // avoids any risk of a finite accumulated cost coincidentally aliasing
    // the sentinel value (or the sentinel silently becoming a huge-but-finite
    // number once a step cost is added to it, which would defeat an
    // exact-equality guard). In this two-branch lattice both nodes at t==0
    // are reachable and every node has both predecessor branches available,
    // so every subsequent node is provably reachable too: `reachable` can
    // never actually be false past t==0, and the `!found`/`!reachable[...]`
    // fallback paths below are dead code by construction, kept only as a
    // structural safety net (not live logic) in case that invariant is ever
    // broken by a future change to this function.
    std::vector<float>   cost(static_cast<size_t>(n) * 2, 0.0f);
    std::vector<uint8_t> back(static_cast<size_t>(n) * 2, 0);
    std::vector<bool>    reachable(static_cast<size_t>(n) * 2, false);

    reachable[0] = true;
    reachable[1] = true;
    cost[0] = 0.0f;
    cost[1] = 0.0f;

    // NOTE on a whole-path ambiguity the DP below cannot see or resolve:
    // alternate_branch is (a0+pi, pi-a1, a2+pi) for Tait-Bryan charts and
    // (a0+pi, -a1, a2+pi) for proper-Euler charts -- an involution mod 2pi.
    // Because the Viterbi cost below only ever sums *consecutive*
    // differences, flipping EVERY frame's branch choice at once (chosen[t]
    // -> chosen[t] ^ 1 for all t) leaves every transition cost identical:
    // the constant +-pi offsets on axes 0 and 2 cancel exactly in any
    // consecutive difference, and negating axis 1 about pi leaves its
    // square unchanged. So `chosen` and its pointwise flip are two equally
    // optimal solutions to this DP, always, for every input on every chart
    // -- not just a corner case. Which one gets reconstructed if left to
    // this loop's `total < best` tie-breaking is decided by float-rounding
    // noise on the accumulated cost, not by anything physically meaningful,
    // and the two differ by a global ~180 degree relabelling (a joint at
    // rest can come out reading (0,0,0) or (180,180,180) depending on which
    // twin won). This is resolved *after* reconstruction, below, by
    // comparing the whole path's distance from the rest pose against its
    // flip's -- deliberately not here, since any per-frame or seed-based
    // tie-break only sees frame 0 and can be wrong when frame 0 favours the
    // opposite family from the rest of the path.
    for (int t = 1; t < n; ++t) {
        for (int b = 0; b < 2; ++b) {
            float best = std::numeric_limits<float>::infinity();
            uint8_t best_prev = 0;
            bool found = false;
            for (int p = 0; p < 2; ++p) {
                if (!reachable[(t - 1) * 2 + p]) continue;
                float step = 0.0f;
                for (int a = 0; a < 3; ++a) {
                    float d = principal_difference(cand[t * 2 + b].angle[a],
                                                   cand[(t - 1) * 2 + p].angle[a]);
                    step += d * d;
                }
                float total = cost[(t - 1) * 2 + p] + step;
                if (!found || total < best) { best = total; best_prev = static_cast<uint8_t>(p); found = true; }
            }
            cost[t * 2 + b] = best;
            back[t * 2 + b] = best_prev;
            reachable[t * 2 + b] = found;
        }
    }

    // Backward pass. Both branches at the last frame are always reachable
    // (every node at t==0 is reachable, and every subsequent node has at
    // least one finite-cost predecessor since both predecessor branches are
    // reachable), so no fallback is needed here.
    const int last = n - 1;
    std::vector<uint8_t> chosen(static_cast<size_t>(n), 0);
    chosen[last] = (reachable[last * 2 + 1] &&
                    (!reachable[last * 2 + 0] || cost[last * 2 + 1] < cost[last * 2 + 0])) ? 1 : 0;
    for (int t = last; t > 0; --t)
        chosen[t - 1] = back[t * 2 + chosen[t]];

    // Reconstruct continuous angles by accumulating principal differences.
    out.time.assign(times.begin(), times.begin() + n);
    for (int a = 0; a < 3; ++a) out.angle[a].resize(n);

    for (int a = 0; a < 3; ++a)
        out.angle[a][0] = cand[0 * 2 + chosen[0]].angle[a];

    for (int t = 1; t < n; ++t)
        for (int a = 0; a < 3; ++a) {
            float d = principal_difference(cand[t * 2 + chosen[t]].angle[a],
                                           cand[(t - 1) * 2 + chosen[t - 1]].angle[a]);
            out.angle[a][t] = out.angle[a][t - 1] + d;
        }

    // Whole-path tie-break (see the NOTE above the Viterbi loop): `chosen`
    // and its pointwise flip are both valid, equally-costed reconstructions
    // of this rotation sequence. Reconstruct the flipped path the same way
    // and keep whichever of the two sits closer to the rest pose overall,
    // summed across every frame and axis; an exact tie keeps the unflipped
    // (as computed above) reconstruction. This can only choose between the
    // two twins -- it cannot alter where a genuine, cost-driven mid-path
    // branch switch happens, because both twins share every transition cost.
    {
        std::vector<float> flipped[3];
        for (int a = 0; a < 3; ++a) flipped[a].resize(static_cast<size_t>(n));

        for (int a = 0; a < 3; ++a)
            flipped[a][0] = cand[0 * 2 + (chosen[0] ^ 1)].angle[a];

        for (int t = 1; t < n; ++t)
            for (int a = 0; a < 3; ++a) {
                float d = principal_difference(cand[t * 2 + (chosen[t] ^ 1)].angle[a],
                                               cand[(t - 1) * 2 + (chosen[t - 1] ^ 1)].angle[a]);
                flipped[a][t] = flipped[a][t - 1] + d;
            }

        float dist_chosen = 0.0f, dist_flipped = 0.0f;
        for (int t = 0; t < n; ++t)
            for (int a = 0; a < 3; ++a) {
                float dc = principal_difference(out.angle[a][t], 0.0f);
                dist_chosen += dc * dc;
                float df = principal_difference(flipped[a][t], 0.0f);
                dist_flipped += df * df;
            }

        if (dist_flipped < dist_chosen) {
            for (int a = 0; a < 3; ++a) out.angle[a] = std::move(flipped[a]);
            for (int t = 0; t < n; ++t) chosen[t] = static_cast<uint8_t>(chosen[t] ^ 1);
        }
    }

    out.worst_conditioning = 1.0f;
    for (int t = 0; t < n; ++t)
        out.worst_conditioning = std::min(out.worst_conditioning, cond[t * 2 + chosen[t]]);

    return out;
}

void anchor_trajectory(Trajectory& t, const float reference[3])
{
    for (int a = 0; a < 3; ++a) {
        if (t.angle[a].empty()) continue;

        float lo = t.angle[a][0], hi = t.angle[a][0];
        for (float v : t.angle[a]) { lo = std::min(lo, v); hi = std::max(hi, v); }

        // Midpoint rather than first sample: an animation may begin at an
        // extreme of its range, and anchoring on that would push the
        // opposite tail across the wrap boundary.
        const float mid   = 0.5f * (lo + hi);
        const float shift = kTwoPi * std::round((reference[a] - mid) / kTwoPi);
        if (shift == 0.0f) continue;
        for (float& v : t.angle[a]) v += shift;
    }
}

} // namespace detail
} // namespace ChaCha
