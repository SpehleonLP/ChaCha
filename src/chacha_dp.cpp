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
        // With a single frame there is no history to disambiguate branches;
        // arbitrarily (but deterministically) take branch 0.
        for (int a = 0; a < 3; ++a) out.angle[a].assign(1, cand[0].angle[a]);
        out.worst_conditioning = cond[0];
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

    // Seed t==0 with each branch's squared distance from the rest pose
    // (all-zero angles), scaled by 1e-3. Without this, the whole-path cost
    // of staying entirely on solve_euler's principal branch is an EXACT tie
    // (up to float rounding) with staying entirely on its alternate_branch
    // twin, for every input on every chart: alternate_branch is
    // (a0+pi, pi-a1, a2+pi) for Tait-Bryan charts and (a0+pi, -a1, a2+pi) for
    // proper-Euler charts, and the Viterbi cost only ever sums consecutive
    // differences, so the constant +-pi offsets cancel exactly on axes 0 and
    // 2, and negating axis 1 about pi leaves its square unchanged. That tie
    // is then broken only by float rounding noise (~1e-5 scale), so which
    // "family" resolve_branches returns for a given input is otherwise
    // arbitrary. Seeding with distance-from-rest breaks the tie in favor of
    // the physically meaningful reading: a joint at rest should decompose
    // near (0,0,0), not near (180,180,180).
    //
    // The 1e-3 scale is load-bearing: do not drop it or round it to a
    // "nicer" constant. An unscaled seed can reach 3*pi^2 (~29.6, when a
    // frame's principal-branch reading sits near the far side of the
    // circle on all three axes), which exceeds the ~20 transition cost of a
    // genuine mid-path branch switch (see DP.SwitchesBranchAcrossChartSingularity)
    // and would suppress that legitimate switch. At 1e-3 the seed is
    // decisive against the ~1e-5 float-rounding noise that would otherwise
    // decide the tie, while staying negligible against any real transition
    // cost, so it only ever breaks exact (or near-exact) ties and never
    // overrides a genuine mid-path switch.
    for (int b = 0; b < 2; ++b) {
        float s = 0.0f;
        for (int a = 0; a < 3; ++a) {
            float w = principal_difference(cand[b].angle[a], 0.0f);
            s += w * w;
        }
        cost[b] = 1e-3f * s;
    }

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
