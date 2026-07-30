# Articulation Solver Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ChaCha's broken quaternion decomposition with an exact 12-chart Euler solver plus a dynamic-programming branch resolver and a fewest-DOF search, and bring both consumers into AGI schema conformance.

**Architecture:** Per joint, every candidate stage set (12 three-axis charts plus reduced 1- and 2-stage sets) is solved against every scanned animation. Each `(candidate, animation)` pair is solved exactly, has its per-frame branch ambiguity resolved by a Viterbi pass, is anchored to a common 2π frame, and is reduced to a fixed-size summary. Summaries union across animations, and the unioned candidates are scored lexicographically on `(dof_count, Σ range, conditioning)`. Artist-authored `AGI ` configuration animations bypass the search entirely and are treated as authoritative.

**Tech Stack:** C++20, GLM, GoogleTest. No JSON, no glTF, no file I/O inside the library.

**Design doc:** `docs/plans/2026-07-30-articulation-solver-design.md`

## Global Constraints

- C++20 (`CMAKE_CXX_STANDARD 20`, extensions OFF). CMake 3.15+.
- ChaCha depends on **GLM only**. No JSON parser, no glTF parser, no file I/O in `src/` or `include/`.
- Files `snake_case`, types `PascalCase`, fields `snake_case`.
- SI units throughout the library: **radians** and **metres**. Degrees appear only in consumer output.
- Scale stage values are **multiplicative factors**, never additive deltas.
- Unit tests live in `chacha/tests/` and must not link any glTF parser.
- Integration tests live in `tonton-example/tests/` where `fx-gltf` is already available.
- Test models live in `chacha/testdata/` (gitignored): `treefrog.glb`, `emporer scorpion.glb`, `sophia-2_9.glb`.
- Every task ends with a commit.

## File Structure

**ChaCha library:**

| File | Responsibility |
|---|---|
| `include/chacha_stage.h` | `StageType`, `Stage` (gains `max_acceleration`) |
| `include/chacha_types.h` | `Animation`, `AnimationChannel`, `Skeleton`, `Articulation`, `Options`, `Diagnostic` |
| `include/chacha_naming.h` | **New.** `sanitize_articulation_name`, `stage_name_for` — shared with consumers |
| `include/chacha.h` | `analyze()` and the index-space contract |
| `src/chacha_internal.h` | `Chart`, `EulerSolution`, `Trajectory`, `CandidateSummary`, pipeline decls |
| `src/chacha_charts.cpp` | **New.** The 12 chart tables, exact solve, both branches, conditioning |
| `src/chacha_dp.cpp` | **New.** Principal-difference lift, Viterbi branch resolution, anchoring |
| `src/chacha_reduced.cpp` | **New.** 1-DOF closed form, 2-DOF Gauss-Newton, residual |
| `src/chacha_summary.cpp` | **New.** Trajectory → `CandidateSummary`; resampled velocity/acceleration |
| `src/chacha_search.cpp` | **New.** Candidate enumeration, union across animations, scoring |
| `src/chacha_config.cpp` | **New.** `AGI ` configuration animation fast path |
| `src/chacha_naming.cpp` | **New.** Name sanitisation implementation |
| `src/chacha_analyzer.cpp` | Orchestration only |
| `src/chacha_filter.cpp` | Noise thresholds; scale semantics |
| `src/chacha_decompose.cpp` | **Deleted** (swing-twist) |
| `src/chacha_segment.cpp` | **Deleted** (absorbed by `chacha_summary.cpp`) |
| `src/chacha_pointing.cpp` | Unchanged |
| `src/chacha_clean.cpp` | Untouched — separate spec |
| `tests/*.cpp` | Unit tests, no glTF |

**Consumers:**

| File | Responsibility |
|---|---|
| `gltfRepackager/src/steps/infer_articulations.cpp` | Reference consumer; adopt API, fix units and naming |
| `tonton-example/src/chacha_fxgltf_bridge.{h,cpp}` | Adopt API, node space, conformance |
| `tonton-example/src/chacha_main.cpp` | Drop skin remap |
| `tonton-example/tests/chacha_integration_tests.cpp` | **New.** treefrog / scorpion / sophia |

---

### Task 1: Test target and scaffolding

Nothing can be verified until tests run. This task adds the harness and nothing else.

**Files:**
- Modify: `CMakeLists.txt`
- Create: `tests/chacha_smoke_test.cpp`

**Interfaces:**
- Consumes: nothing
- Produces: a `chacha-tests` CMake target runnable via `ctest`

- [ ] **Step 1: Write the failing test**

Create `tests/chacha_smoke_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include "chacha_stage.h"

TEST(Smoke, StageTypeNamesAreStable)
{
    EXPECT_STREQ(ChaCha::stage_type_name(ChaCha::StageType::xRotate), "xRotate");
    EXPECT_STREQ(ChaCha::stage_type_name(ChaCha::StageType::zScale), "zScale");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake -S . -B build && cmake --build build`
Expected: FAIL — no `chacha-tests` target exists.

- [ ] **Step 3: Add the test target**

Append to `CMakeLists.txt`. Note `CONFIGURE_DEPENDS` is added to the existing globs at the same time so new source files are picked up without a manual re-run:

```cmake
# Pick up newly added sources without a manual cmake re-run.
file(GLOB_RECURSE CHACHA_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

option(CHACHA_BUILD_TESTS "Build ChaCha unit tests" ON)

if(CHACHA_BUILD_TESTS)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        enable_testing()
        file(GLOB CHACHA_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
        add_executable(chacha-tests ${CHACHA_TEST_SOURCES})
        target_include_directories(chacha-tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(chacha-tests PRIVATE chacha GTest::gtest GTest::gtest_main)
        if(glm_FOUND)
            target_link_libraries(chacha-tests PRIVATE glm::glm)
        endif()
        include(GoogleTest)
        gtest_discover_tests(chacha-tests)
    else()
        message(STATUS "GTest not found; ChaCha unit tests disabled")
    endif()
endif()
```

Replace the existing non-`CONFIGURE_DEPENDS` `CHACHA_SOURCES` glob at line 9 with the version above.

- [ ] **Step 4: Run to verify it passes**

Run: `cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure`
Expected: PASS, 1 test.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt tests/chacha_smoke_test.cpp
git commit -m "test: add chacha-tests GTest target"
```

---

### Task 2: Chart tables and the exact Euler solve

The single most important task. Everything downstream assumes this is exact.

**Files:**
- Create: `src/chacha_charts.cpp`
- Modify: `src/chacha_internal.h`
- Test: `tests/chacha_charts_test.cpp`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `struct Chart { StageType stage[3]; int axis[3]; bool proper; }`
  - `std::span<const Chart> all_charts()` — exactly 12 entries
  - `struct EulerSolution { float angle[3]; }`
  - `EulerSolution solve_euler(const glm::quat& q, const Chart& c)`
  - `glm::quat compose_chart(const Chart& c, const float angle[3])`

- [ ] **Step 1: Write the failing test**

Create `tests/chacha_charts_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>
#include <random>

using namespace ChaCha::detail;

static float angular_distance(const glm::quat& a, const glm::quat& b)
{
    float d = std::fabs(glm::dot(glm::normalize(a), glm::normalize(b)));
    if (d > 1.0f) d = 1.0f;
    return 2.0f * std::acos(d);
}

TEST(Charts, ThereAreTwelve)
{
    EXPECT_EQ(all_charts().size(), 12u);
    int proper = 0;
    for (const auto& c : all_charts()) if (c.proper) ++proper;
    EXPECT_EQ(proper, 6);
}

TEST(Charts, SolveRoundTripsExactlyForEveryChart)
{
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);

    for (const auto& chart : all_charts()) {
        float worst = 0.0f;
        for (int trial = 0; trial < 500; ++trial) {
            glm::quat q = glm::normalize(glm::quat(u(rng), u(rng), u(rng), u(rng)));
            EulerSolution s = solve_euler(q, chart);
            glm::quat r = compose_chart(chart, s.angle);
            worst = std::max(worst, angular_distance(q, r));
        }
        EXPECT_LT(worst, 1e-4f) << "chart failed round trip";
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build && ctest --test-dir build -R Charts --output-on-failure`
Expected: FAIL — `all_charts` and `solve_euler` undeclared.

- [ ] **Step 3: Declare the types**

Add to `src/chacha_internal.h` inside `namespace ChaCha { namespace detail {`:

```cpp
// A chart is an ordered triple of rotation axes. `axis` holds 0=X, 1=Y, 2=Z.
// `proper` is true for proper-Euler charts (axis[0] == axis[2], e.g. ZXZ) and
// false for Tait-Bryan charts (three distinct axes, e.g. XYZ).
struct Chart {
    StageType stage[3];
    int       axis[3];
    bool      proper;
};

struct EulerSolution {
    float angle[3];
};

std::span<const Chart> all_charts();
EulerSolution solve_euler(const glm::quat& q, const Chart& c);
glm::quat     compose_chart(const Chart& c, const float angle[3]);
glm::quat     axis_quat(int axis, float angle_rad);
```

Ensure `chacha_internal.h` includes `<span>` and `"chacha_stage.h"`.

- [ ] **Step 4: Implement**

Create `src/chacha_charts.cpp`:

```cpp
#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>
#include <array>
#include <cmath>

namespace ChaCha {
namespace detail {

namespace {

constexpr float kPi     = 3.14159265358979323846f;
constexpr float kHalfPi = kPi * 0.5f;

StageType rotate_stage(int axis)
{
    switch (axis) {
    case 0:  return StageType::xRotate;
    case 1:  return StageType::yRotate;
    default: return StageType::zRotate;
    }
}

std::array<Chart, 12> build_charts()
{
    std::array<Chart, 12> out{};
    int n = 0;
    // Tait-Bryan: three distinct axes.
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                if (i != j && j != k && i != k)
                    out[n++] = Chart{{rotate_stage(i), rotate_stage(j), rotate_stage(k)},
                                     {i, j, k}, false};
    // Proper Euler: first and last axis match, middle differs.
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (i != j)
                out[n++] = Chart{{rotate_stage(i), rotate_stage(j), rotate_stage(i)},
                                 {i, j, i}, true};
    return out;
}

float wrap_pi(float a)
{
    while (a >  kPi) a -= 2.0f * kPi;
    while (a <= -kPi) a += 2.0f * kPi;
    return a;
}

} // namespace

glm::quat axis_quat(int axis, float angle_rad)
{
    const float h = angle_rad * 0.5f;
    const float s = std::sin(h), c = std::cos(h);
    switch (axis) {
    case 0:  return glm::quat(c, s, 0.0f, 0.0f);
    case 1:  return glm::quat(c, 0.0f, s, 0.0f);
    default: return glm::quat(c, 0.0f, 0.0f, s);
    }
}

std::span<const Chart> all_charts()
{
    static const std::array<Chart, 12> charts = build_charts();
    return std::span<const Chart>(charts.data(), charts.size());
}

glm::quat compose_chart(const Chart& c, const float angle[3])
{
    // Stages apply in order of appearance, matching AGI semantics:
    //   node_transform * stage0 * stage1 * stage2
    return axis_quat(c.axis[0], angle[0])
         * axis_quat(c.axis[1], angle[1])
         * axis_quat(c.axis[2], angle[2]);
}

// Generic quaternion -> Euler extraction covering all 12 sequences
// (Bernardes & Viollet, PLoS ONE 2022).
EulerSolution solve_euler(const glm::quat& qin, const Chart& chart)
{
    const glm::quat q = glm::normalize(qin);
    const int i = chart.axis[0];
    const int j = chart.axis[1];
    const int k = chart.proper ? (3 - i - j) : chart.axis[2];

    // Sign of the permutation (i, j, k).
    const float eps = static_cast<float>((i - j) * (j - k) * (k - i)) / 2.0f;

    const float qv[3] = {q.x, q.y, q.z};

    float a, b, c, d;
    if (chart.proper) {
        a = q.w;         b = qv[i];
        c = qv[j];       d = qv[k] * eps;
    } else {
        a = q.w - qv[j]; b = qv[i] + qv[k] * eps;
        c = qv[j] + q.w; d = qv[k] * eps - qv[i];
    }

    float theta2 = 2.0f * std::atan2(std::hypot(c, d), std::hypot(a, b));
    const float tp = std::atan2(b, a);
    const float tm = std::atan2(d, c);

    float theta1, theta3;
    if (std::fabs(theta2) < 1e-6f) {
        theta1 = 0.0f;
        theta3 = 2.0f * tp - theta1;
    } else if (std::fabs(theta2 - kPi) < 1e-6f) {
        theta1 = 0.0f;
        theta3 = 2.0f * tm + theta1;
    } else {
        theta1 = tp - tm;
        theta3 = tp + tm;
    }

    if (!chart.proper) {
        theta3 *= eps;
        theta2 -= kHalfPi;
    }

    return EulerSolution{{wrap_pi(theta1), wrap_pi(theta2), wrap_pi(theta3)}};
}

} // namespace detail
} // namespace ChaCha
```

**Implementer note:** the `eps` sign and the proper/not-proper branch are the two places
this algorithm is easy to transcribe wrong. `SolveRoundTripsExactlyForEveryChart` is the
arbiter — if a chart fails, the fault is almost certainly a sign in the `a b c d`
assignment or the trailing `theta3 *= eps` / `theta2 -= kHalfPi` adjustment, not in the
overall structure. Fix against the test rather than by inspection.

- [ ] **Step 5: Run to verify it passes**

Run: `cmake --build build && ctest --test-dir build -R Charts --output-on-failure`
Expected: PASS, both tests. Round-trip error under 1e-4 rad for all 12 charts.

- [ ] **Step 6: Commit**

```bash
git add src/chacha_charts.cpp src/chacha_internal.h tests/chacha_charts_test.cpp
git commit -m "feat: exact 12-chart Euler solver"
```

---

### Task 3: The second solution branch and conditioning

**Files:**
- Modify: `src/chacha_charts.cpp`, `src/chacha_internal.h`
- Test: `tests/chacha_charts_test.cpp`

**Interfaces:**
- Consumes: `Chart`, `EulerSolution`, `solve_euler`, `compose_chart` (Task 2)
- Produces:
  - `EulerSolution alternate_branch(const EulerSolution& s, const Chart& c)`
  - `float chart_conditioning(const EulerSolution& s, const Chart& c)` — 0 at singularity, 1 best

- [ ] **Step 1: Write the failing test**

Append to `tests/chacha_charts_test.cpp`:

```cpp
TEST(Charts, BothBranchesDescribeTheSameRotation)
{
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (const auto& chart : all_charts()) {
        for (int trial = 0; trial < 200; ++trial) {
            glm::quat q = glm::normalize(glm::quat(u(rng), u(rng), u(rng), u(rng)));
            EulerSolution s0 = solve_euler(q, chart);
            EulerSolution s1 = alternate_branch(s0, chart);
            EXPECT_LT(angular_distance(compose_chart(chart, s1.angle), q), 1e-4f);
        }
    }
}

TEST(Charts, ConditioningFallsToZeroAtTheSingularity)
{
    // Tait-Bryan XYZ is singular when the middle angle reaches +-90 degrees.
    Chart xyz{};
    for (const auto& c : all_charts())
        if (!c.proper && c.axis[0] == 0 && c.axis[1] == 1 && c.axis[2] == 2) xyz = c;

    float safe[3]    = {0.3f, 0.0f,             0.2f};
    float singular[3]= {0.3f, 3.14159265f*0.5f, 0.2f};

    EXPECT_GT(chart_conditioning(EulerSolution{{safe[0], safe[1], safe[2]}}, xyz), 0.9f);
    EXPECT_LT(chart_conditioning(EulerSolution{{singular[0], singular[1], singular[2]}}, xyz), 1e-3f);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ctest --test-dir build -R Charts --output-on-failure`
Expected: FAIL — `alternate_branch` and `chart_conditioning` undeclared.

- [ ] **Step 3: Declare**

Add to `src/chacha_internal.h`:

```cpp
EulerSolution alternate_branch(const EulerSolution& s, const Chart& c);
// 1.0 is perfectly conditioned, 0.0 is exactly singular.
float chart_conditioning(const EulerSolution& s, const Chart& c);
```

- [ ] **Step 4: Implement**

Add to `src/chacha_charts.cpp` before the closing namespaces:

```cpp
EulerSolution alternate_branch(const EulerSolution& s, const Chart& c)
{
    // Tait-Bryan:    (t1 + pi,  pi - t2,  t3 + pi)
    // Proper Euler:  (t1 + pi,     -t2,   t3 + pi)
    const float t2 = c.proper ? -s.angle[1] : (kPi - s.angle[1]);
    return EulerSolution{{
        wrap_pi(s.angle[0] + kPi),
        wrap_pi(t2),
        wrap_pi(s.angle[2] + kPi),
    }};
}

float chart_conditioning(const EulerSolution& s, const Chart& c)
{
    // Tait-Bryan degenerates as the middle angle approaches +-pi/2;
    // proper Euler degenerates as it approaches 0 or pi.
    return c.proper ? std::fabs(std::sin(s.angle[1]))
                    : std::fabs(std::cos(s.angle[1]));
}
```

`wrap_pi` and `kPi` are in the anonymous namespace above, so move `alternate_branch` and
`chart_conditioning` below it in the file.

- [ ] **Step 5: Run to verify it passes**

Run: `ctest --test-dir build -R Charts --output-on-failure`
Expected: PASS, 4 tests.

- [ ] **Step 6: Commit**

```bash
git add src/chacha_charts.cpp src/chacha_internal.h tests/chacha_charts_test.cpp
git commit -m "feat: alternate Euler branch and chart conditioning"
```

---

### Task 4: Viterbi branch resolution

**Files:**
- Create: `src/chacha_dp.cpp`
- Modify: `src/chacha_internal.h`
- Test: `tests/chacha_dp_test.cpp`

**Interfaces:**
- Consumes: `Chart`, `EulerSolution`, `solve_euler`, `alternate_branch`, `chart_conditioning`
- Produces:
  - `struct Trajectory { std::vector<float> angle[3]; std::vector<float> time; float worst_conditioning; }`
  - `Trajectory resolve_branches(std::span<const glm::quat> rel, std::span<const float> times, const Chart& c)`
  - `float principal_difference(float current, float previous)`

- [ ] **Step 1: Write the failing test**

Create `tests/chacha_dp_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>
#include <vector>

using namespace ChaCha::detail;
static constexpr float kPi = 3.14159265358979323846f;

static Chart tait_xyz()
{
    for (const auto& c : all_charts())
        if (!c.proper && c.axis[0] == 0 && c.axis[1] == 1 && c.axis[2] == 2) return c;
    return all_charts()[0];
}

TEST(DP, PrincipalDifferenceNeverExceedsPi)
{
    EXPECT_NEAR(principal_difference(3.0f, -3.0f), 3.0f - (-3.0f) - 2.0f*kPi, 1e-5f);
    EXPECT_LE(std::fabs(principal_difference(3.0f, -3.0f)), kPi + 1e-5f);
    EXPECT_NEAR(principal_difference(0.1f, 0.0f), 0.1f, 1e-6f);
}

// A joint sweeping smoothly through +-180 degrees must unwrap monotonically
// rather than snapping back into the principal branch.
TEST(DP, UnwrapsMonotonicallyAcrossPi)
{
    std::vector<glm::quat> rel;
    std::vector<float> times;
    for (int i = 0; i <= 40; ++i) {
        float deg = 150.0f + i * 1.5f;           // 150 -> 210 degrees
        times.push_back(i * 0.05f);
        rel.push_back(glm::angleAxis(glm::radians(deg), glm::vec3(1, 0, 0)));
    }
    Trajectory t = resolve_branches(rel, times, tait_xyz());
    for (size_t i = 1; i < t.angle[0].size(); ++i)
        EXPECT_GT(t.angle[0][i], t.angle[0][i - 1]) << "at " << i;

    float span = t.angle[0].back() - t.angle[0].front();
    EXPECT_NEAR(span, glm::radians(60.0f), glm::radians(2.0f));
}

// An aliased fast rotation must resolve to its minimal reading.
TEST(DP, AliasedRotationTakesTheMinimalReading)
{
    std::vector<glm::quat> rel;
    std::vector<float> times;
    for (int i = 0; i < 12; ++i) {
        times.push_back(i / 30.0f);
        rel.push_back(glm::angleAxis(glm::radians(350.0f * i), glm::vec3(1, 0, 0)));
    }
    Trajectory t = resolve_branches(rel, times, tait_xyz());
    float step = t.angle[0][1] - t.angle[0][0];
    EXPECT_NEAR(glm::degrees(step), -10.0f, 1.0f);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ctest --test-dir build -R DP --output-on-failure`
Expected: FAIL — `resolve_branches` undeclared.

- [ ] **Step 3: Declare**

Add to `src/chacha_internal.h`:

```cpp
struct Trajectory {
    std::vector<float> time;
    std::vector<float> angle[3];   // continuous, unwrapped, in chart stage order
    float worst_conditioning{1.0f};
};

float principal_difference(float current, float previous);

Trajectory resolve_branches(
    std::span<const glm::quat> rel_rotations,
    std::span<const float>     times,
    const Chart&               chart);
```

- [ ] **Step 4: Implement**

Create `src/chacha_dp.cpp`:

```cpp
#include "chacha_internal.h"
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
    const int n = static_cast<int>(rel.size());
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

    // Viterbi. Cost of a transition is the summed squared principal difference
    // across the three axes; the 2pi lift is forced, so branch is the only choice.
    const float kInf = std::numeric_limits<float>::max();
    std::vector<float>   cost(static_cast<size_t>(n) * 2, kInf);
    std::vector<uint8_t> back(static_cast<size_t>(n) * 2, 0);

    cost[0] = 0.0f;
    cost[1] = 0.0f;

    for (int t = 1; t < n; ++t) {
        for (int b = 0; b < 2; ++b) {
            float best = kInf;
            uint8_t best_prev = 0;
            for (int p = 0; p < 2; ++p) {
                if (cost[(t - 1) * 2 + p] == kInf) continue;
                float step = 0.0f;
                for (int a = 0; a < 3; ++a) {
                    float d = principal_difference(cand[t * 2 + b].angle[a],
                                                   cand[(t - 1) * 2 + p].angle[a]);
                    step += d * d;
                }
                float total = cost[(t - 1) * 2 + p] + step;
                if (total < best) { best = total; best_prev = static_cast<uint8_t>(p); }
            }
            cost[t * 2 + b] = best;
            back[t * 2 + b] = best_prev;
        }
    }

    // Backward pass.
    std::vector<uint8_t> chosen(static_cast<size_t>(n), 0);
    chosen[n - 1] = (cost[(n - 1) * 2 + 1] < cost[(n - 1) * 2 + 0]) ? 1 : 0;
    for (int t = n - 1; t > 0; --t)
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

} // namespace detail
} // namespace ChaCha
```

Add `#include <algorithm>` for `std::min` if not already pulled in transitively.

- [ ] **Step 5: Run to verify it passes**

Run: `ctest --test-dir build -R DP --output-on-failure`
Expected: PASS, 3 tests.

- [ ] **Step 6: Commit**

```bash
git add src/chacha_dp.cpp src/chacha_internal.h tests/chacha_dp_test.cpp
git commit -m "feat: Viterbi branch resolution with forced 2pi lift"
```

---

### Task 5: Cross-animation anchoring

The defect this prevents is invisible to any single-animation test.

**Files:**
- Modify: `src/chacha_dp.cpp`, `src/chacha_internal.h`
- Test: `tests/chacha_dp_test.cpp`

**Interfaces:**
- Consumes: `Trajectory` (Task 4)
- Produces: `void anchor_trajectory(Trajectory& t)`

- [ ] **Step 1: Write the failing test**

Append to `tests/chacha_dp_test.cpp`:

```cpp
// Two animations of one joint that straddle +-180 degrees from opposite sides
// must anchor into a common frame, or their union spans a spurious ~360 degrees.
TEST(DP, AnchoringKeepsIndependentAnimationsCommensurable)
{
    auto build = [](float from_deg, float to_deg) {
        std::vector<glm::quat> rel; std::vector<float> times;
        for (int i = 0; i <= 20; ++i) {
            float f = static_cast<float>(i) / 20.0f;
            float deg = from_deg + f * (to_deg - from_deg);
            times.push_back(i * 0.05f);
            rel.push_back(glm::angleAxis(glm::radians(deg), glm::vec3(1, 0, 0)));
        }
        return std::make_pair(rel, times);
    };

    auto [ra, ta] = build(170.0f, 185.0f);   // crosses 180 upward
    auto [rb, tb] = build(190.0f, 175.0f);   // crosses 180 downward

    Trajectory a = resolve_branches(ra, ta, tait_xyz());
    Trajectory b = resolve_branches(rb, tb, tait_xyz());
    anchor_trajectory(a);
    anchor_trajectory(b);

    float lo = std::min(*std::min_element(a.angle[0].begin(), a.angle[0].end()),
                        *std::min_element(b.angle[0].begin(), b.angle[0].end()));
    float hi = std::max(*std::max_element(a.angle[0].begin(), a.angle[0].end()),
                        *std::max_element(b.angle[0].begin(), b.angle[0].end()));

    // True excursion is 170..190 degrees, i.e. 20 degrees wide.
    EXPECT_LT(glm::degrees(hi - lo), 40.0f);
}
```

Add `#include <algorithm>` and `#include <utility>` at the top of the test file.

- [ ] **Step 2: Run to verify it fails**

Run: `ctest --test-dir build -R DP --output-on-failure`
Expected: FAIL — `anchor_trajectory` undeclared.

- [ ] **Step 3: Declare**

Add to `src/chacha_internal.h`:

```cpp
// Shifts each axis by the multiple of 2pi placing the trajectory midpoint into
// (-pi, pi]. Required before summaries from different animations are unioned.
void anchor_trajectory(Trajectory& t);
```

- [ ] **Step 4: Implement**

Add to `src/chacha_dp.cpp`:

```cpp
void anchor_trajectory(Trajectory& t)
{
    for (int a = 0; a < 3; ++a) {
        if (t.angle[a].empty()) continue;

        float lo = t.angle[a][0], hi = t.angle[a][0];
        for (float v : t.angle[a]) { lo = std::min(lo, v); hi = std::max(hi, v); }

        // Midpoint rather than first sample: an animation may begin at an
        // extreme of its range, and anchoring on that would push the opposite
        // tail across the wrap boundary.
        const float mid   = 0.5f * (lo + hi);
        const float shift = -kTwoPi * std::round(mid / kTwoPi);
        if (shift == 0.0f) continue;
        for (float& v : t.angle[a]) v += shift;
    }
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `ctest --test-dir build -R DP --output-on-failure`
Expected: PASS, 4 tests.

- [ ] **Step 6: Commit**

```bash
git add src/chacha_dp.cpp src/chacha_internal.h tests/chacha_dp_test.cpp
git commit -m "feat: anchor trajectories to a common 2pi frame before union"
```

---

### Task 6: Trajectory summary with resampled derivatives

**Files:**
- Create: `src/chacha_summary.cpp`
- Delete: `src/chacha_segment.cpp`
- Modify: `src/chacha_internal.h`
- Test: `tests/chacha_summary_test.cpp`

**Interfaces:**
- Consumes: `Trajectory` (Task 4), `Options`
- Produces:
  - `struct CandidateSummary { float min_value[3], max_value[3], max_velocity[3], max_acceleration[3], max_residual_rad, worst_conditioning; bool valid; }`
  - `CandidateSummary summarise(const Trajectory& t, float residual_rad, const Options& o)`
  - `void union_into(CandidateSummary& dst, const CandidateSummary& src)`

- [ ] **Step 1: Write the failing test**

Create `tests/chacha_summary_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include "chacha_internal.h"
#include "chacha_types.h"
#include <cmath>

using namespace ChaCha;
using namespace ChaCha::detail;

static Trajectory ramp(float from, float to, int n, float duration)
{
    Trajectory t;
    t.time.resize(n);
    for (int a = 0; a < 3; ++a) t.angle[a].assign(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        float f = (n == 1) ? 0.0f : static_cast<float>(i) / (n - 1);
        t.time[i]     = f * duration;
        t.angle[0][i] = from + f * (to - from);
    }
    return t;
}

// Ranges must come from observed samples, never seeded at zero.
TEST(Summary, RangeIsSeededFromObservedSamplesNotZero)
{
    Trajectory t = ramp(0.5f, 1.0f, 21, 1.0f);
    CandidateSummary s = summarise(t, 0.0f, Options{});
    EXPECT_NEAR(s.min_value[0], 0.5f, 1e-4f);
    EXPECT_NEAR(s.max_value[0], 1.0f, 1e-4f);
}

TEST(Summary, VelocityMatchesAConstantRamp)
{
    // 1.0 rad over 2.0 s => 0.5 rad/s.
    Trajectory t = ramp(0.0f, 1.0f, 121, 2.0f);
    CandidateSummary s = summarise(t, 0.0f, Options{});
    EXPECT_NEAR(s.max_velocity[0], 0.5f, 0.05f);
}

// A piecewise-linear signal differentiated per keyframe yields impulses whose
// magnitude tracks keyframe density. Resampling must make acceleration stable.
TEST(Summary, AccelerationIsStableUnderKeyframeDensity)
{
    Trajectory sparse = ramp(0.0f, 1.0f, 11,  2.0f);
    Trajectory dense  = ramp(0.0f, 1.0f, 401, 2.0f);
    CandidateSummary a = summarise(sparse, 0.0f, Options{});
    CandidateSummary b = summarise(dense,  0.0f, Options{});
    EXPECT_LT(a.max_acceleration[0], 0.5f);
    EXPECT_LT(b.max_acceleration[0], 0.5f);
}

TEST(Summary, UnionTakesWidestRangeAndWorstConditioning)
{
    CandidateSummary a = summarise(ramp(0.0f, 1.0f, 21, 1.0f), 0.01f, Options{});
    CandidateSummary b = summarise(ramp(-2.0f, 0.5f, 21, 1.0f), 0.03f, Options{});
    a.worst_conditioning = 0.9f;
    b.worst_conditioning = 0.4f;
    union_into(a, b);
    EXPECT_NEAR(a.min_value[0], -2.0f, 1e-4f);
    EXPECT_NEAR(a.max_value[0],  1.0f, 1e-4f);
    EXPECT_NEAR(a.max_residual_rad, 0.03f, 1e-6f);
    EXPECT_NEAR(a.worst_conditioning, 0.4f, 1e-6f);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ctest --test-dir build -R Summary --output-on-failure`
Expected: FAIL — `summarise` undeclared.

- [ ] **Step 3: Declare**

Add to `src/chacha_internal.h`:

```cpp
struct CandidateSummary {
    bool  valid{false};
    float min_value[3]{};
    float max_value[3]{};
    float max_velocity[3]{};
    float max_acceleration[3]{};
    float max_residual_rad{0.0f};
    float worst_conditioning{1.0f};
};

CandidateSummary summarise(const Trajectory& t, float residual_rad, const Options& options);
void union_into(CandidateSummary& dst, const CandidateSummary& src);
```

- [ ] **Step 4: Implement**

Create `src/chacha_summary.cpp`:

```cpp
#include "chacha_internal.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace ChaCha {
namespace detail {

namespace {

// Linear resample of an already-continuous angle track onto a uniform grid.
std::vector<float> resample(const std::vector<float>& time,
                            const std::vector<float>& value,
                            float rate_hz)
{
    std::vector<float> out;
    const int n = static_cast<int>(time.size());
    if (n == 0 || rate_hz <= 0.0f) return out;
    if (n == 1) { out.push_back(value[0]); return out; }

    const float dt       = 1.0f / rate_hz;
    const float duration = time[n - 1] - time[0];
    if (duration <= 0.0f) { out.push_back(value[0]); return out; }

    const int steps = static_cast<int>(duration * rate_hz) + 1;
    out.reserve(steps);

    int k = 0;
    for (int i = 0; i < steps; ++i) {
        const float t = time[0] + i * dt;
        while (k + 2 < n && time[k + 1] < t) ++k;
        const float span = time[k + 1] - time[k];
        const float f    = (span > 0.0f) ? std::clamp((t - time[k]) / span, 0.0f, 1.0f) : 0.0f;
        out.push_back(value[k] + f * (value[k + 1] - value[k]));
    }
    return out;
}

// Central difference over a symmetric window; returns max |derivative|.
float max_abs_central_difference(const std::vector<float>& v, float dt, int window)
{
    const int n = static_cast<int>(v.size());
    const int h = std::max(1, window / 2);
    if (n <= 2 * h) return 0.0f;

    float worst = 0.0f;
    for (int i = h; i + h < n; ++i) {
        const float d = (v[i + h] - v[i - h]) / (2.0f * h * dt);
        worst = std::max(worst, std::fabs(d));
    }
    return worst;
}

std::vector<float> central_difference(const std::vector<float>& v, float dt, int window)
{
    const int n = static_cast<int>(v.size());
    const int h = std::max(1, window / 2);
    std::vector<float> out;
    if (n <= 2 * h) return out;
    out.reserve(n - 2 * h);
    for (int i = h; i + h < n; ++i)
        out.push_back((v[i + h] - v[i - h]) / (2.0f * h * dt));
    return out;
}

} // namespace

CandidateSummary summarise(const Trajectory& t, float residual_rad, const Options& options)
{
    CandidateSummary s;
    if (t.time.empty()) return s;

    s.valid              = true;
    s.max_residual_rad   = residual_rad;
    s.worst_conditioning = t.worst_conditioning;

    const float dt = 1.0f / options.resample_rate_hz;

    for (int a = 0; a < 3; ++a) {
        const auto& v = t.angle[a];
        if (v.empty()) continue;

        s.min_value[a] = *std::min_element(v.begin(), v.end());
        s.max_value[a] = *std::max_element(v.begin(), v.end());

        // Derivatives are taken on a uniform grid. Differentiating raw keyframes
        // of a piecewise-linear channel yields impulses whose magnitude scales
        // with keyframe density rather than with motion.
        const std::vector<float> grid = resample(t.time, v, options.resample_rate_hz);
        s.max_velocity[a]     = max_abs_central_difference(grid, dt, options.derivative_window);
        const std::vector<float> vel = central_difference(grid, dt, options.derivative_window);
        s.max_acceleration[a] = max_abs_central_difference(vel, dt, options.derivative_window);
    }
    return s;
}

void union_into(CandidateSummary& dst, const CandidateSummary& src)
{
    if (!src.valid) return;
    if (!dst.valid) { dst = src; return; }

    for (int a = 0; a < 3; ++a) {
        dst.min_value[a]        = std::min(dst.min_value[a], src.min_value[a]);
        dst.max_value[a]        = std::max(dst.max_value[a], src.max_value[a]);
        dst.max_velocity[a]     = std::max(dst.max_velocity[a], src.max_velocity[a]);
        dst.max_acceleration[a] = std::max(dst.max_acceleration[a], src.max_acceleration[a]);
    }
    dst.max_residual_rad   = std::max(dst.max_residual_rad, src.max_residual_rad);
    dst.worst_conditioning = std::min(dst.worst_conditioning, src.worst_conditioning);
}

} // namespace detail
} // namespace ChaCha
```

- [ ] **Step 5: Delete the superseded file**

```bash
git rm src/chacha_segment.cpp
```

Remove the `segment_and_merge` declaration from `src/chacha_internal.h`. `chacha_analyzer.cpp`
still calls it and will not compile; that is expected and is repaired in Task 10. To keep the
tree building until then, temporarily comment out the `segment_and_merge` and `filter_stages`
calls in `analyze()` and return an empty vector.

- [ ] **Step 6: Run to verify it passes**

Run: `cmake --build build && ctest --test-dir build -R Summary --output-on-failure`
Expected: PASS, 4 tests.

- [ ] **Step 7: Commit**

```bash
git add -A src tests/chacha_summary_test.cpp
git commit -m "feat: candidate summary with resampled velocity and acceleration"
```

---

### Task 7: Reduced-DOF candidates

**Files:**
- Create: `src/chacha_reduced.cpp`
- Modify: `src/chacha_internal.h`
- Test: `tests/chacha_reduced_test.cpp`

**Interfaces:**
- Consumes: `axis_quat` (Task 2)
- Produces:
  - `float solve_one_dof(const glm::quat& q, int axis)`
  - `void solve_two_dof(const glm::quat& q, int axis0, int axis1, float seed[2], float out[2])`
  - `float residual_one_dof(const glm::quat& q, int axis, float angle)`
  - `float residual_two_dof(const glm::quat& q, int axis0, int axis1, const float a[2])`

- [ ] **Step 1: Write the failing test**

Create `tests/chacha_reduced_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>

using namespace ChaCha::detail;

TEST(Reduced, OneDofRecoversAPureHingeExactly)
{
    for (int axis = 0; axis < 3; ++axis) {
        glm::vec3 v(0.0f); v[axis] = 1.0f;
        glm::quat q = glm::angleAxis(glm::radians(37.0f), v);
        float a = solve_one_dof(q, axis);
        EXPECT_NEAR(glm::degrees(a), 37.0f, 1e-3f);
        EXPECT_LT(residual_one_dof(q, axis, a), 1e-5f);
    }
}

TEST(Reduced, OneDofReportsLargeResidualForOffAxisMotion)
{
    glm::quat q = glm::angleAxis(glm::radians(60.0f), glm::normalize(glm::vec3(1, 0, 1)));
    float a = solve_one_dof(q, 0);
    EXPECT_GT(residual_one_dof(q, 0, a), glm::radians(10.0f));
}

TEST(Reduced, TwoDofRecoversATwoAxisComposition)
{
    glm::quat q = glm::angleAxis(glm::radians(40.0f), glm::vec3(1, 0, 0))
                * glm::angleAxis(glm::radians(-25.0f), glm::vec3(0, 0, 1));
    float seed[2] = {0.0f, 0.0f};
    float out[2]  = {0.0f, 0.0f};
    solve_two_dof(q, 0, 2, seed, out);
    EXPECT_LT(residual_two_dof(q, 0, 2, out), 1e-3f);
    EXPECT_NEAR(glm::degrees(out[0]),  40.0f, 0.5f);
    EXPECT_NEAR(glm::degrees(out[1]), -25.0f, 0.5f);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ctest --test-dir build -R Reduced --output-on-failure`
Expected: FAIL — `solve_one_dof` undeclared.

- [ ] **Step 3: Declare**

Add to `src/chacha_internal.h`:

```cpp
float solve_one_dof(const glm::quat& q, int axis);
float residual_one_dof(const glm::quat& q, int axis, float angle);
void  solve_two_dof(const glm::quat& q, int axis0, int axis1,
                    const float seed[2], float out[2]);
float residual_two_dof(const glm::quat& q, int axis0, int axis1, const float a[2]);
```

- [ ] **Step 4: Implement**

Create `src/chacha_reduced.cpp`:

```cpp
#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>
#include <cmath>

namespace ChaCha {
namespace detail {

namespace {

float angular_distance(const glm::quat& a, const glm::quat& b)
{
    float d = std::fabs(glm::dot(glm::normalize(a), glm::normalize(b)));
    if (d > 1.0f) d = 1.0f;
    return 2.0f * std::acos(d);
}

// Log map of a unit quaternion into the tangent space at identity.
glm::vec3 log_map(glm::quat q)
{
    if (q.w < 0.0f) q = -q;             // shortest arc
    const glm::vec3 v(q.x, q.y, q.z);
    const float vn = glm::length(v);
    if (vn < 1e-8f) return glm::vec3(0.0f);
    const float angle = 2.0f * std::atan2(vn, q.w);
    return v * (angle / vn);
}

} // namespace

// The nearest rotation about `axis` to q is its projection onto that axis.
float solve_one_dof(const glm::quat& q, int axis)
{
    const float qv[3] = {q.x, q.y, q.z};
    return 2.0f * std::atan2(qv[axis], q.w);
}

float residual_one_dof(const glm::quat& q, int axis, float angle)
{
    return angular_distance(axis_quat(axis, angle), q);
}

float residual_two_dof(const glm::quat& q, int axis0, int axis1, const float a[2])
{
    return angular_distance(axis_quat(axis0, a[0]) * axis_quat(axis1, a[1]), q);
}

// Gauss-Newton on the 3-vector residual r(t) = log( R(t)^-1 * q ), two unknowns.
void solve_two_dof(const glm::quat& q, int axis0, int axis1,
                   const float seed[2], float out[2])
{
    out[0] = seed[0];
    out[1] = seed[1];

    const float kStep = 1e-3f;

    for (int iter = 0; iter < 24; ++iter) {
        auto residual_at = [&](float a0, float a1) {
            const glm::quat r = axis_quat(axis0, a0) * axis_quat(axis1, a1);
            return log_map(glm::inverse(r) * q);
        };

        const glm::vec3 r0 = residual_at(out[0], out[1]);
        if (glm::length(r0) < 1e-7f) break;

        // Numerical Jacobian, 3x2.
        const glm::vec3 c0 = (residual_at(out[0] + kStep, out[1]) - r0) / kStep;
        const glm::vec3 c1 = (residual_at(out[0], out[1] + kStep) - r0) / kStep;

        // Normal equations for a 2x2 system, with light damping for stability.
        const float a = glm::dot(c0, c0) + 1e-6f;
        const float b = glm::dot(c0, c1);
        const float d = glm::dot(c1, c1) + 1e-6f;
        const float g0 = glm::dot(c0, r0);
        const float g1 = glm::dot(c1, r0);

        const float det = a * d - b * b;
        if (std::fabs(det) < 1e-12f) break;

        const float s0 = ( d * g0 - b * g1) / det;
        const float s1 = (-b * g0 + a * g1) / det;

        out[0] += s0;
        out[1] += s1;

        if (std::fabs(s0) < 1e-7f && std::fabs(s1) < 1e-7f) break;
    }
}

} // namespace detail
} // namespace ChaCha
```

- [ ] **Step 5: Run to verify it passes**

Run: `ctest --test-dir build -R Reduced --output-on-failure`
Expected: PASS, 3 tests.

- [ ] **Step 6: Commit**

```bash
git add src/chacha_reduced.cpp src/chacha_internal.h tests/chacha_reduced_test.cpp
git commit -m "feat: reduced-DOF candidate solvers"
```

---

### Task 8: Candidate enumeration, union, and scoring

**Files:**
- Create: `src/chacha_search.cpp`
- Modify: `src/chacha_internal.h`
- Test: `tests/chacha_search_test.cpp`

**Interfaces:**
- Consumes: everything from Tasks 2–7
- Produces:
  - `struct JointMotion { std::vector<std::vector<glm::quat>> rel_by_animation; std::vector<std::vector<float>> times_by_animation; }`
  - `struct Candidate { StageType stage[3]; int axis[3]; int dof; bool proper; CandidateSummary summary; }`
  - `Candidate select_candidate(const JointMotion& m, const Options& o)`

- [ ] **Step 1: Write the failing test**

Create `tests/chacha_search_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include "chacha_internal.h"
#include "chacha_types.h"
#include <glm/gtc/quaternion.hpp>

using namespace ChaCha;
using namespace ChaCha::detail;

static JointMotion motion_from(const std::vector<std::vector<glm::quat>>& anims)
{
    JointMotion m;
    for (const auto& a : anims) {
        m.rel_by_animation.push_back(a);
        std::vector<float> t(a.size());
        for (size_t i = 0; i < a.size(); ++i) t[i] = static_cast<float>(i) / 30.0f;
        m.times_by_animation.push_back(std::move(t));
    }
    return m;
}

TEST(Search, PureHingeResolvesToOneDof)
{
    std::vector<glm::quat> a;
    for (int i = 0; i <= 30; ++i)
        a.push_back(glm::angleAxis(glm::radians(i * 2.0f), glm::vec3(1, 0, 0)));
    Candidate c = select_candidate(motion_from({a}), Options{});
    EXPECT_EQ(c.dof, 1);
    EXPECT_EQ(c.stage[0], StageType::xRotate);
}

// The regression that motivated this rework: a pure two-axis swing must not
// manufacture a third rotational degree of freedom.
TEST(Search, TwoAxisSwingProducesNoPhantomThirdDof)
{
    std::vector<glm::quat> a;
    for (int i = 0; i <= 30; ++i)
        a.push_back(glm::angleAxis(glm::radians(45.0f), glm::vec3(1, 0, 0))
                  * glm::angleAxis(glm::radians(i * 3.0f), glm::vec3(0, 0, 1)));
    Candidate c = select_candidate(motion_from({a}), Options{});
    EXPECT_LE(c.dof, 2);
    for (int s = 0; s < c.dof; ++s)
        EXPECT_NE(c.stage[s], StageType::yRotate) << "phantom axial twist";
}

TEST(Search, GeneralMotionUsesThreeDofWithNegligibleResidual)
{
    std::vector<glm::quat> a;
    for (int i = 0; i <= 40; ++i) {
        float f = static_cast<float>(i);
        a.push_back(glm::angleAxis(glm::radians(f * 1.5f), glm::vec3(1, 0, 0))
                  * glm::angleAxis(glm::radians(f * 2.0f), glm::vec3(0, 1, 0))
                  * glm::angleAxis(glm::radians(f * 1.1f), glm::vec3(0, 0, 1)));
    }
    Candidate c = select_candidate(motion_from({a}), Options{});
    EXPECT_EQ(c.dof, 3);
    EXPECT_LT(c.summary.max_residual_rad, 1e-3f);
}

// A single chart must be chosen for the joint as a whole; per-animation chart
// selection would produce angles that cannot be unioned.
TEST(Search, RangeIsUnionedAcrossAnimations)
{
    std::vector<glm::quat> a, b;
    for (int i = 0; i <= 20; ++i)
        a.push_back(glm::angleAxis(glm::radians(i * 1.0f), glm::vec3(1, 0, 0)));
    for (int i = 0; i <= 20; ++i)
        b.push_back(glm::angleAxis(glm::radians(-i * 2.0f), glm::vec3(1, 0, 0)));

    Candidate c = select_candidate(motion_from({a, b}), Options{});
    EXPECT_EQ(c.dof, 1);
    EXPECT_NEAR(glm::degrees(c.summary.min_value[0]), -40.0f, 1.0f);
    EXPECT_NEAR(glm::degrees(c.summary.max_value[0]),  20.0f, 1.0f);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ctest --test-dir build -R Search --output-on-failure`
Expected: FAIL — `select_candidate` undeclared.

- [ ] **Step 3: Declare**

Add to `src/chacha_internal.h`:

```cpp
struct JointMotion {
    std::vector<std::vector<glm::quat>> rel_by_animation;
    std::vector<std::vector<float>>     times_by_animation;
};

struct Candidate {
    StageType        stage[3]{};
    int              axis[3]{};
    int              dof{0};
    bool             proper{false};
    CandidateSummary summary;
};

Candidate select_candidate(const JointMotion& motion, const Options& options);
```

- [ ] **Step 4: Implement**

Create `src/chacha_search.cpp`:

```cpp
#include "chacha_internal.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace ChaCha {
namespace detail {

namespace {

StageType rotate_stage_for(int axis)
{
    switch (axis) {
    case 0:  return StageType::xRotate;
    case 1:  return StageType::yRotate;
    default: return StageType::zRotate;
    }
}

// Full-rank candidate: exact solve, Viterbi, anchor, summarise.
Candidate evaluate_chart(const Chart& chart, const JointMotion& m, const Options& o)
{
    Candidate c;
    c.dof    = 3;
    c.proper = chart.proper;
    for (int s = 0; s < 3; ++s) { c.stage[s] = chart.stage[s]; c.axis[s] = chart.axis[s]; }

    for (size_t ai = 0; ai < m.rel_by_animation.size(); ++ai) {
        Trajectory t = resolve_branches(m.rel_by_animation[ai], m.times_by_animation[ai], chart);
        if (t.time.empty()) continue;
        anchor_trajectory(t);
        // Three distinct axes reconstruct exactly, so residual is zero by construction.
        union_into(c.summary, summarise(t, 0.0f, o));
    }
    return c;
}

Candidate evaluate_one_dof(int axis, const JointMotion& m, const Options& o)
{
    Candidate c;
    c.dof      = 1;
    c.stage[0] = rotate_stage_for(axis);
    c.axis[0]  = axis;

    for (size_t ai = 0; ai < m.rel_by_animation.size(); ++ai) {
        const auto& rel   = m.rel_by_animation[ai];
        const auto& times = m.times_by_animation[ai];
        if (rel.empty()) continue;

        Trajectory t;
        t.time.assign(times.begin(), times.end());
        for (int a = 0; a < 3; ++a) t.angle[a].assign(rel.size(), 0.0f);

        float worst = 0.0f;
        float prev  = 0.0f;
        for (size_t i = 0; i < rel.size(); ++i) {
            const float raw = solve_one_dof(rel[i], axis);
            const float ang = (i == 0) ? raw : prev + principal_difference(raw, prev);
            t.angle[0][i] = ang;
            prev = ang;
            worst = std::max(worst, residual_one_dof(rel[i], axis, ang));
        }
        anchor_trajectory(t);
        union_into(c.summary, summarise(t, worst, o));
    }
    return c;
}

Candidate evaluate_two_dof(int axis0, int axis1, const JointMotion& m, const Options& o)
{
    Candidate c;
    c.dof      = 2;
    c.stage[0] = rotate_stage_for(axis0);  c.axis[0] = axis0;
    c.stage[1] = rotate_stage_for(axis1);  c.axis[1] = axis1;

    for (size_t ai = 0; ai < m.rel_by_animation.size(); ++ai) {
        const auto& rel   = m.rel_by_animation[ai];
        const auto& times = m.times_by_animation[ai];
        if (rel.empty()) continue;

        Trajectory t;
        t.time.assign(times.begin(), times.end());
        for (int a = 0; a < 3; ++a) t.angle[a].assign(rel.size(), 0.0f);

        float worst   = 0.0f;
        float seed[2] = {0.0f, 0.0f};
        float prev[2] = {0.0f, 0.0f};

        for (size_t i = 0; i < rel.size(); ++i) {
            float sol[2];
            // Seeding from the previous frame keeps the solve in the same branch.
            solve_two_dof(rel[i], axis0, axis1, seed, sol);
            seed[0] = sol[0];
            seed[1] = sol[1];

            for (int a = 0; a < 2; ++a) {
                const float ang = (i == 0) ? sol[a] : prev[a] + principal_difference(sol[a], prev[a]);
                t.angle[a][i] = ang;
                prev[a] = ang;
            }
            worst = std::max(worst, residual_two_dof(rel[i], axis0, axis1, sol));
        }
        anchor_trajectory(t);
        union_into(c.summary, summarise(t, worst, o));
    }
    return c;
}

float total_range(const Candidate& c)
{
    float sum = 0.0f;
    for (int a = 0; a < c.dof; ++a)
        sum += c.summary.max_value[a] - c.summary.min_value[a];
    return sum;
}

// Lexicographic: fewer DOF, then tighter total range, then better conditioning.
bool better(const Candidate& lhs, const Candidate& rhs)
{
    if (!rhs.summary.valid) return true;
    if (!lhs.summary.valid) return false;
    if (lhs.dof != rhs.dof) return lhs.dof < rhs.dof;

    const float lr = total_range(lhs), rr = total_range(rhs);
    if (std::fabs(lr - rr) > 1e-4f) return lr < rr;

    return lhs.summary.worst_conditioning > rhs.summary.worst_conditioning;
}

} // namespace

Candidate select_candidate(const JointMotion& motion, const Options& options)
{
    Candidate best;

    // Reduced candidates are only admissible if they explain every observed pose.
    for (int axis = 0; axis < 3; ++axis) {
        Candidate c = evaluate_one_dof(axis, motion, options);
        if (c.summary.valid && c.summary.max_residual_rad <= options.max_fit_residual_rad
            && better(c, best))
            best = c;
    }

    for (int a0 = 0; a0 < 3; ++a0)
        for (int a1 = 0; a1 < 3; ++a1) {
            if (a0 == a1) continue;
            Candidate c = evaluate_two_dof(a0, a1, motion, options);
            if (c.summary.valid && c.summary.max_residual_rad <= options.max_fit_residual_rad
                && better(c, best))
                best = c;
        }

    for (const auto& chart : all_charts()) {
        Candidate c = evaluate_chart(chart, motion, options);
        if (c.summary.valid && better(c, best)) best = c;
    }

    return best;
}

} // namespace detail
} // namespace ChaCha
```

- [ ] **Step 5: Run to verify it passes**

Run: `ctest --test-dir build -R Search --output-on-failure`
Expected: PASS, 4 tests. In particular `TwoAxisSwingProducesNoPhantomThirdDof` — the
headline defect — must pass.

- [ ] **Step 6: Commit**

```bash
git add src/chacha_search.cpp src/chacha_internal.h tests/chacha_search_test.cpp
git commit -m "feat: candidate search with union-then-score selection"
```

---

### Task 9: Public type changes

**Files:**
- Modify: `include/chacha_stage.h`, `include/chacha_types.h`, `include/chacha.h`
- Test: `tests/chacha_api_test.cpp`

**Interfaces:**
- Consumes: nothing
- Produces: `Animation`, `Diagnostic`, extended `AnimationChannel` / `Skeleton` / `Articulation` / `Options` / `Stage`, new `analyze()` signature

- [ ] **Step 1: Write the failing test**

Create `tests/chacha_api_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include "chacha.h"

using namespace ChaCha;

TEST(Api, DefaultsMatchTheSpec)
{
    Options o;
    EXPECT_FLOAT_EQ(o.rotation_threshold_rad,  0.01f);
    EXPECT_FLOAT_EQ(o.translation_threshold_m, 0.001f);
    EXPECT_FLOAT_EQ(o.scale_threshold,         0.01f);
    EXPECT_FLOAT_EQ(o.max_fit_residual_rad,    0.02f);
    EXPECT_FLOAT_EQ(o.resample_rate_hz,        60.0f);
    EXPECT_EQ(o.derivative_window,             5);
    EXPECT_TRUE(o.prioritize_rom_animations);
}

TEST(Api, StageCarriesAccelerationNotEffort)
{
    Stage s;
    s.max_acceleration = 1.5f;
    EXPECT_FLOAT_EQ(s.max_acceleration, 1.5f);
}

TEST(Api, AnalyzeAcceptsEmptyInputWithoutCrashing)
{
    Skeleton sk;
    std::vector<Diagnostic> diags;
    auto out = analyze({}, {}, sk, Options{}, {}, &diags);
    EXPECT_TRUE(out.empty());
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ctest --test-dir build -R Api --output-on-failure`
Expected: FAIL — `max_acceleration` and the new `analyze` overload do not exist.

- [ ] **Step 3: Update the headers**

`include/chacha_stage.h` — replace the `Stage` struct:

```cpp
struct Stage {
    StageType type;
    float min_value{};
    float max_value{};
    float initial_value{};
    float max_velocity{};
    float max_acceleration{};   // was max_effort; effort needs inertia, out of scope
};
```

`include/chacha_types.h` — add and extend:

```cpp
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
    std::span<const glm::vec3> rest_scales;   // empty means all-ones
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
    float max_fit_residual_rad{0.02f};
    float resample_rate_hz{60.0f};
    int   derivative_window{5};
    bool  prioritize_rom_animations{true};
};

struct Diagnostic {
    int node{-1};
    int animation{-1};
    enum Kind { NonUnitQuaternion, EmptyChannel, MalformedValues } kind{EmptyChannel};
};
```

Add `#include <string_view>` to `chacha_types.h`.

`include/chacha.h` — replace the declaration:

```cpp
/// Deduce joint articulation constraints from animation data.
///
/// INDEX SPACE: `AnimationChannel::node` indexes the same array as
/// `Skeleton::parents`. ChaCha works in glTF *node* space. Skin-joint space is
/// incorrect: AGI articulations are per node, and glTF animation channels
/// target nodes rather than skin joints.
///
/// `scan` empty means "all animations". In that case, if
/// `options.prioritize_rom_animations` is true and any animation name begins
/// with "AGI " (case-insensitive), the scan narrows to those animations and all
/// others are ignored — an artist-authored configuration animation is a direct
/// specification of the intended constraints. A non-empty `scan` is honoured
/// verbatim with no auto-narrowing.
std::vector<Articulation> analyze(
    std::span<const AnimationChannel> channels,
    std::span<const Animation>        animations,
    const Skeleton&                   skeleton,
    const Options&                    options = {},
    std::span<const int>              scan = {},
    std::vector<Diagnostic>*          diagnostics = nullptr);
```

- [ ] **Step 4: Stub the new signature**

In `src/chacha_analyzer.cpp`, replace the old `analyze` definition with one matching the new
signature that returns an empty vector. Tasks 10 and 11 fill it in.

- [ ] **Step 5: Run to verify it passes**

Run: `cmake --build build && ctest --test-dir build -R Api --output-on-failure`
Expected: PASS, 3 tests.

- [ ] **Step 6: Commit**

```bash
git add include src/chacha_analyzer.cpp tests/chacha_api_test.cpp
git commit -m "feat: public API for animations, diagnostics and acceleration"
```

---

### Task 10: Orchestration — extraction, filtering, and stage emission

**Files:**
- Modify: `src/chacha_analyzer.cpp`, `src/chacha_filter.cpp`, `src/chacha_internal.h`
- Delete: `src/chacha_decompose.cpp`
- Test: `tests/chacha_analyze_test.cpp`

**Interfaces:**
- Consumes: `select_candidate` (Task 8), `Options`, `Skeleton`, `Animation`
- Produces: a working `analyze()` for rotation, translation and scale

- [ ] **Step 1: Write the failing test**

Create `tests/chacha_analyze_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include "chacha.h"
#include <glm/gtc/quaternion.hpp>
#include <vector>

using namespace ChaCha;

namespace {
struct Rig {
    std::vector<int>       parents{-1};
    std::vector<glm::quat> rest{glm::quat(1, 0, 0, 0)};
    std::vector<glm::vec3> trans{glm::vec3(0)};
    std::vector<glm::vec3> scale{glm::vec3(1)};
    Skeleton skeleton() const { return Skeleton{parents, rest, trans, scale}; }
};
} // namespace

// The second headline regression: observed range must not be widened to include
// the rest pose.
TEST(Analyze, RangeIsNotSeededToZero)
{
    Rig rig;
    std::vector<float> times{0.0f, 0.5f, 1.0f};
    std::vector<float> values;
    for (float deg : {30.0f, 45.0f, 60.0f}) {
        glm::quat q = glm::angleAxis(glm::radians(deg), glm::vec3(1, 0, 0));
        values.insert(values.end(), {q.x, q.y, q.z, q.w});
    }
    AnimationChannel ch;
    ch.node = 0; ch.animation = 0; ch.property = Property::Rotation;
    ch.times = times; ch.values = values;

    std::vector<AnimationChannel> chans{ch};
    std::vector<Animation> anims{Animation{"walk"}};

    auto out = analyze(chans, anims, rig.skeleton());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].stages.size(), 1u);
    EXPECT_EQ(out[0].stages[0].type, StageType::xRotate);
    EXPECT_NEAR(glm::degrees(out[0].stages[0].min_value), 30.0f, 1.0f);
    EXPECT_NEAR(glm::degrees(out[0].stages[0].max_value), 60.0f, 1.0f);
    EXPECT_GE(out[0].stages[0].initial_value, out[0].stages[0].min_value);
    EXPECT_LE(out[0].stages[0].initial_value, out[0].stages[0].max_value);
}

TEST(Analyze, MotionAtRestProducesNoArticulation)
{
    Rig rig;
    rig.rest[0] = glm::angleAxis(glm::radians(30.0f), glm::vec3(1, 0, 0));

    std::vector<float> times{0.0f, 0.5f, 1.0f};
    std::vector<float> values;
    for (int i = 0; i < 3; ++i) {
        glm::quat q = rig.rest[0];
        values.insert(values.end(), {q.x, q.y, q.z, q.w});
    }
    AnimationChannel ch;
    ch.node = 0; ch.animation = 0; ch.property = Property::Rotation;
    ch.times = times; ch.values = values;

    std::vector<AnimationChannel> chans{ch};
    std::vector<Animation> anims{Animation{"idle"}};
    EXPECT_TRUE(analyze(chans, anims, rig.skeleton()).empty());
}

// Scale stages are multiplicative factors, not additive deltas.
TEST(Analyze, ScaleIsAMultiplicativeFactor)
{
    Rig rig;
    std::vector<float> times{0.0f, 1.0f};
    std::vector<float> values{1.0f, 1.0f, 1.0f,  2.0f, 1.0f, 1.0f};
    AnimationChannel ch;
    ch.node = 0; ch.animation = 0; ch.property = Property::Scale;
    ch.times = times; ch.values = values;

    std::vector<AnimationChannel> chans{ch};
    std::vector<Animation> anims{Animation{"grow"}};

    auto out = analyze(chans, anims, rig.skeleton());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].stages.size(), 1u);
    EXPECT_EQ(out[0].stages[0].type, StageType::xScale);
    EXPECT_NEAR(out[0].stages[0].min_value, 1.0f, 1e-4f);
    EXPECT_NEAR(out[0].stages[0].max_value, 2.0f, 1e-4f);
    EXPECT_NEAR(out[0].stages[0].initial_value, 1.0f, 1e-4f);
}

TEST(Analyze, NonUnitQuaternionIsReportedAsADiagnostic)
{
    Rig rig;
    std::vector<float> times{0.0f, 1.0f};
    std::vector<float> values{0, 0, 0, 0,   0, 0, 0, 0};   // zero quaternions
    AnimationChannel ch;
    ch.node = 0; ch.animation = 0; ch.property = Property::Rotation;
    ch.times = times; ch.values = values;

    std::vector<AnimationChannel> chans{ch};
    std::vector<Animation> anims{Animation{"broken"}};
    std::vector<Diagnostic> diags;
    analyze(chans, anims, rig.skeleton(), Options{}, {}, &diags);

    ASSERT_FALSE(diags.empty());
    EXPECT_EQ(diags[0].kind, Diagnostic::NonUnitQuaternion);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ctest --test-dir build -R Analyze --output-on-failure`
Expected: FAIL — `analyze` currently returns an empty vector.

- [ ] **Step 3: Update the filter for scale**

Replace `filter_stages` in `src/chacha_filter.cpp` so scale is measured as a deviation
from 1.0 rather than from 0.0:

```cpp
std::vector<Stage> filter_stages(
    std::span<const RawStage> raw_stages,
    const Options& options)
{
    std::vector<Stage> result;
    result.reserve(raw_stages.size());

    for (const auto& raw : raw_stages) {
        const float threshold = threshold_for_type(raw.type, options);
        if (raw.range() < threshold) continue;

        Stage stage;
        stage.type             = raw.type;
        stage.min_value        = raw.min_value;
        stage.max_value        = raw.max_value;
        stage.initial_value    = std::clamp(raw.initial_value, raw.min_value, raw.max_value);
        stage.max_velocity     = raw.max_velocity;
        stage.max_acceleration = raw.max_acceleration;
        result.push_back(stage);
    }
    return result;
}
```

Update `RawStage` in `src/chacha_internal.h` to carry `max_acceleration` instead of
`max_effort`. Add `#include <algorithm>` to `chacha_filter.cpp`.

- [ ] **Step 4: Implement orchestration**

Replace the body of `src/chacha_analyzer.cpp` with:

```cpp
#include "chacha_internal.h"
#include "chacha.h"
#include <algorithm>
#include <cctype>
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
namespace {

int stride_for(Property p, InterpolationType i)
{
    const int vpk = (p == Property::Rotation) ? 4 : 3;
    return (i == InterpolationType::CubicSpline) ? vpk * 3 : vpk;
}

const float* key_at(const AnimationChannel& ch, int i)
{
    const int vpk    = (ch.property == Property::Rotation) ? 4 : 3;
    const int stride = stride_for(ch.property, ch.interp);
    const int offset = (ch.interp == InterpolationType::CubicSpline) ? vpk : 0;
    return ch.values.data() + static_cast<size_t>(i) * stride + offset;
}

bool channel_is_well_formed(const AnimationChannel& ch)
{
    const int n = static_cast<int>(ch.times.size());
    if (n == 0) return false;
    return ch.values.size() >= static_cast<size_t>(n) * stride_for(ch.property, ch.interp);
}

bool starts_with_agi(std::string_view name)
{
    if (name.size() < 4) return false;
    auto lower = [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); };
    return lower(name[0]) == 'a' && lower(name[1]) == 'g' && lower(name[2]) == 'i' && name[3] == ' ';
}

} // namespace

std::vector<int> resolve_scan(
    std::span<const Animation> animations,
    std::span<const int>       scan,
    const Options&             options)
{
    std::vector<int> out;
    if (!scan.empty()) {                       // explicit request is honoured verbatim
        out.assign(scan.begin(), scan.end());
        return out;
    }

    if (options.prioritize_rom_animations) {
        for (int i = 0; i < static_cast<int>(animations.size()); ++i)
            if (starts_with_agi(animations[i].name)) out.push_back(i);
        if (!out.empty()) return out;          // artist-authored spec wins outright
    }

    for (int i = 0; i < static_cast<int>(animations.size()); ++i) out.push_back(i);
    return out;
}

} // namespace detail

std::vector<Articulation> analyze(
    std::span<const AnimationChannel> channels,
    std::span<const Animation>        animations,
    const Skeleton&                   skeleton,
    const Options&                    options,
    std::span<const int>              scan,
    std::vector<Diagnostic>*          diagnostics)
{
    using namespace detail;

    const int num_nodes = static_cast<int>(skeleton.parents.size());
    if (num_nodes == 0) return {};

    const std::vector<int> scanned = resolve_scan(animations, scan, options);
    std::vector<bool> in_scan(animations.size(), false);
    for (int a : scanned)
        if (a >= 0 && a < static_cast<int>(animations.size())) in_scan[a] = true;

    auto note = [&](int node, int anim, Diagnostic::Kind k) {
        if (diagnostics) diagnostics->push_back(Diagnostic{node, anim, k});
    };

    // Per node: rotation trajectories grouped by animation, plus vec3 accumulators.
    std::map<int, JointMotion> rotation_by_node;
    struct Vec3Acc {
        bool  seen[3]{};
        float lo[3]{}, hi[3]{}, vel[3]{}, acc[3]{}, initial[3]{};
    };
    std::map<int, Vec3Acc> translation_by_node, scale_by_node;

    for (const auto& ch : channels) {
        if (ch.node < 0 || ch.node >= num_nodes) continue;
        if (ch.animation < 0 || ch.animation >= static_cast<int>(animations.size())) continue;
        if (!in_scan[ch.animation]) continue;
        if (!channel_is_well_formed(ch)) {
            note(ch.node, ch.animation,
                 ch.times.empty() ? Diagnostic::EmptyChannel : Diagnostic::MalformedValues);
            continue;
        }

        const int n = static_cast<int>(ch.times.size());

        if (ch.property == Property::Rotation) {
            const glm::quat rest_inv = glm::inverse(glm::normalize(skeleton.rest_rotations[ch.node]));
            std::vector<glm::quat> rel;
            std::vector<float>     times;
            rel.reserve(n); times.reserve(n);

            bool degenerate = false;
            for (int i = 0; i < n; ++i) {
                const float* v = key_at(ch, i);
                glm::quat key(v[3], v[0], v[1], v[2]);
                const float len = std::sqrt(glm::dot(key, key));
                if (!(len > 1e-6f)) { degenerate = true; break; }
                if (std::fabs(len - 1.0f) > 1e-3f) degenerate = true;   // still usable
                rel.push_back(rest_inv * (key / len));
                times.push_back(ch.times[i]);
            }
            if (degenerate) note(ch.node, ch.animation, Diagnostic::NonUnitQuaternion);
            if (rel.empty()) continue;

            JointMotion& m = rotation_by_node[ch.node];
            m.rel_by_animation.push_back(std::move(rel));
            m.times_by_animation.push_back(std::move(times));
            continue;
        }

        // Translation and scale.
        const bool is_scale = (ch.property == Property::Scale);
        glm::vec3 rest(0.0f);
        if (is_scale) {
            rest = skeleton.rest_scales.empty() ? glm::vec3(1.0f)
                                                : skeleton.rest_scales[ch.node];
        } else {
            rest = skeleton.rest_translations[ch.node];
        }

        Vec3Acc& acc = is_scale ? scale_by_node[ch.node] : translation_by_node[ch.node];

        std::vector<float> track[3];
        for (int a = 0; a < 3; ++a) track[a].reserve(n);

        for (int i = 0; i < n; ++i) {
            const float* v = key_at(ch, i);
            for (int a = 0; a < 3; ++a) {
                // Scale is a multiplicative factor relative to rest; translation is a delta.
                const float rest_a = rest[a];
                const float value  = is_scale
                    ? ((std::fabs(rest_a) > 1e-8f) ? v[a] / rest_a : v[a])
                    : (v[a] - rest_a);
                track[a].push_back(value);
            }
        }

        Trajectory t;
        t.time.assign(ch.times.begin(), ch.times.begin() + n);
        for (int a = 0; a < 3; ++a) t.angle[a] = track[a];
        const CandidateSummary s = summarise(t, 0.0f, options);

        for (int a = 0; a < 3; ++a) {
            if (!acc.seen[a]) {
                acc.seen[a]    = true;
                acc.lo[a]      = s.min_value[a];
                acc.hi[a]      = s.max_value[a];
                acc.initial[a] = track[a].front();
            } else {
                acc.lo[a] = std::min(acc.lo[a], s.min_value[a]);
                acc.hi[a] = std::max(acc.hi[a], s.max_value[a]);
            }
            acc.vel[a] = std::max(acc.vel[a], s.max_velocity[a]);
            acc.acc[a] = std::max(acc.acc[a], s.max_acceleration[a]);
        }
    }

    // Assemble per-node articulations.
    std::map<int, std::vector<RawStage>> raw_by_node;
    std::map<int, Candidate>             candidate_by_node;

    auto push_vec3 = [&](const std::map<int, Vec3Acc>& src, StageType base) {
        for (const auto& [node, acc] : src) {
            for (int a = 0; a < 3; ++a) {
                if (!acc.seen[a]) continue;
                RawStage r;
                r.type             = static_cast<StageType>(static_cast<int>(base) + a);
                r.min_value        = acc.lo[a];
                r.max_value        = acc.hi[a];
                r.initial_value    = acc.initial[a];
                r.max_velocity     = acc.vel[a];
                r.max_acceleration = acc.acc[a];
                raw_by_node[node].push_back(r);
            }
        }
    };

    push_vec3(translation_by_node, StageType::xTranslate);
    push_vec3(scale_by_node,       StageType::xScale);

    for (const auto& [node, motion] : rotation_by_node) {
        Candidate c = select_candidate(motion, options);
        if (!c.summary.valid) continue;
        candidate_by_node[node] = c;
        for (int s = 0; s < c.dof; ++s) {
            RawStage r;
            r.type             = c.stage[s];
            r.min_value        = c.summary.min_value[s];
            r.max_value        = c.summary.max_value[s];
            r.initial_value    = 0.0f;              // rest pose is the origin
            r.max_velocity     = c.summary.max_velocity[s];
            r.max_acceleration = c.summary.max_acceleration[s];
            raw_by_node[node].push_back(r);
        }
    }

    std::vector<Articulation> result;
    for (auto& [node, raws] : raw_by_node) {
        std::vector<Stage> stages = filter_stages(raws, options);
        if (stages.empty()) continue;

        Articulation art;
        art.node = node;
        art.stages = std::move(stages);

        auto it = candidate_by_node.find(node);
        if (it != candidate_by_node.end()) {
            art.dof_count        = static_cast<uint8_t>(it->second.dof);
            art.fit_residual_rad = it->second.summary.max_residual_rad;
        }
        result.push_back(std::move(art));
    }

    std::sort(result.begin(), result.end(),
              [](const Articulation& a, const Articulation& b) { return a.node < b.node; });

    detail::infer_pointing_vectors(result, skeleton);
    return result;
}

} // namespace ChaCha
```

Delete `src/chacha_decompose.cpp` and remove the `SwingTwist` struct,
`decompose_swing_twist`, `swing_to_angles`, `twist_to_angle`, `extract_dof_tracks`,
`segment_and_merge`, `optimize_stage_order` and the `DofTrack` / `Sample` declarations
from `src/chacha_internal.h`.

```bash
git rm src/chacha_decompose.cpp
```

- [ ] **Step 5: Run to verify it passes**

Run: `cmake --build build && ctest --test-dir build --output-on-failure`
Expected: PASS, all tests including the four `Analyze` cases.

- [ ] **Step 6: Commit**

```bash
git add -A src include tests
git commit -m "feat: rebuild analyze() on the candidate search; drop swing-twist"
```

---

### Task 11: `AGI ` configuration animation fast path

**Files:**
- Create: `src/chacha_config.cpp`
- Modify: `src/chacha_internal.h`, `src/chacha_analyzer.cpp`
- Test: `tests/chacha_config_test.cpp`

**Interfaces:**
- Consumes: `JointMotion`, `Options`
- Produces:
  - `bool is_configuration_motion(const JointMotion& m, const Options& o)`
  - `Candidate solve_configuration(const JointMotion& m, const Options& o)`

- [ ] **Step 1: Write the failing test**

Create `tests/chacha_config_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include "chacha_internal.h"
#include "chacha_types.h"
#include <glm/gtc/quaternion.hpp>

using namespace ChaCha;
using namespace ChaCha::detail;

// Mirrors the treefrog layout: X sweep, rest, Z sweep, rest, Y sweep, rest.
static JointMotion configuration_motion()
{
    const glm::vec3 axes[3] = {glm::vec3(1,0,0), glm::vec3(0,0,1), glm::vec3(0,1,0)};
    std::vector<glm::quat> rel;
    std::vector<float> times;
    int frame = 0;
    for (int phase = 0; phase < 3; ++phase) {
        for (int i = 0; i <= 40; ++i) {
            float f = static_cast<float>(i) / 40.0f;
            float deg = 13.4f * std::sin(f * 2.0f * 3.14159265f);
            rel.push_back(glm::angleAxis(glm::radians(deg), axes[phase]));
            times.push_back(frame++ / 24.0f);
        }
    }
    JointMotion m;
    m.rel_by_animation.push_back(std::move(rel));
    m.times_by_animation.push_back(std::move(times));
    return m;
}

TEST(Config, DetectsSequentialSingleAxisPhases)
{
    EXPECT_TRUE(is_configuration_motion(configuration_motion(), Options{}));
}

TEST(Config, RecoversStageOrderAndRange)
{
    Candidate c = solve_configuration(configuration_motion(), Options{});
    ASSERT_EQ(c.dof, 3);
    EXPECT_EQ(c.stage[0], StageType::xRotate);
    EXPECT_EQ(c.stage[1], StageType::zRotate);
    EXPECT_EQ(c.stage[2], StageType::yRotate);
    for (int s = 0; s < 3; ++s) {
        EXPECT_NEAR(glm::degrees(c.summary.min_value[s]), -13.4f, 1.0f);
        EXPECT_NEAR(glm::degrees(c.summary.max_value[s]),  13.4f, 1.0f);
    }
}

TEST(Config, RejectsOrdinaryMultiAxisMotion)
{
    std::vector<glm::quat> rel; std::vector<float> times;
    for (int i = 0; i <= 60; ++i) {
        float f = static_cast<float>(i);
        rel.push_back(glm::angleAxis(glm::radians(f), glm::vec3(1,0,0))
                    * glm::angleAxis(glm::radians(f * 0.7f), glm::vec3(0,0,1)));
        times.push_back(i / 30.0f);
    }
    JointMotion m;
    m.rel_by_animation.push_back(rel);
    m.times_by_animation.push_back(times);
    EXPECT_FALSE(is_configuration_motion(m, Options{}));
}

TEST(Config, JointHeldAtRestYieldsNoStages)
{
    JointMotion m;
    m.rel_by_animation.push_back({glm::quat(1,0,0,0), glm::quat(1,0,0,0)});
    m.times_by_animation.push_back({0.0f, 5.0f});
    Candidate c = solve_configuration(m, Options{});
    EXPECT_EQ(c.dof, 0);
}
```

Add `#include <cmath>` to the test file.

- [ ] **Step 2: Run to verify it fails**

Run: `ctest --test-dir build -R Config --output-on-failure`
Expected: FAIL — `is_configuration_motion` undeclared.

- [ ] **Step 3: Declare**

Add to `src/chacha_internal.h`:

```cpp
// An artist-authored configuration animation exercises one axis at a time,
// returning to rest between phases, in the order the stages should apply.
bool      is_configuration_motion(const JointMotion& motion, const Options& options);
Candidate solve_configuration(const JointMotion& motion, const Options& options);
```

- [ ] **Step 4: Implement**

Create `src/chacha_config.cpp`:

```cpp
#include "chacha_internal.h"
#include <algorithm>
#include <cmath>

namespace ChaCha {
namespace detail {

namespace {

constexpr float kRestEpsilonRad = 0.5f * 3.14159265358979323846f / 180.0f;  // 0.5 degrees

StageType rotate_stage_for(int axis)
{
    switch (axis) {
    case 0:  return StageType::xRotate;
    case 1:  return StageType::yRotate;
    default: return StageType::zRotate;
    }
}

struct Phase {
    int   axis{-1};
    float lo{0.0f};
    float hi{0.0f};
};

// Split a trajectory at returns-to-rest, and identify each phase's single axis.
// Returns false if any phase is not dominated by one principal axis.
bool extract_phases(const std::vector<glm::quat>& rel, std::vector<Phase>& out)
{
    out.clear();
    bool  in_phase = false;
    Phase current;

    for (const glm::quat& q : rel) {
        const glm::quat n = glm::normalize(q);
        const float v[3]  = {n.x, n.y, n.z};
        const float mag2  = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
        const float angle = 2.0f * std::atan2(std::sqrt(mag2), std::fabs(n.w));

        if (angle < kRestEpsilonRad) {           // at rest: close any open phase
            if (in_phase) { out.push_back(current); in_phase = false; }
            continue;
        }

        int   axis = 0;
        float best = std::fabs(v[0]);
        for (int a = 1; a < 3; ++a)
            if (std::fabs(v[a]) > best) { best = std::fabs(v[a]); axis = a; }

        // The rotation must lie essentially on one principal axis.
        if (best * best < 0.98f * mag2) return false;

        const float signed_angle = (v[axis] < 0.0f) ? -angle : angle;

        if (!in_phase) {
            in_phase     = true;
            current      = Phase{axis, signed_angle, signed_angle};
        } else if (current.axis != axis) {
            return false;                        // two axes inside one phase
        } else {
            current.lo = std::min(current.lo, signed_angle);
            current.hi = std::max(current.hi, signed_angle);
        }
    }
    if (in_phase) out.push_back(current);

    // Each axis may appear at most once.
    for (size_t i = 0; i < out.size(); ++i)
        for (size_t j = i + 1; j < out.size(); ++j)
            if (out[i].axis == out[j].axis) return false;

    return true;
}

} // namespace

bool is_configuration_motion(const JointMotion& motion, const Options&)
{
    if (motion.rel_by_animation.empty()) return false;

    for (const auto& rel : motion.rel_by_animation) {
        std::vector<Phase> phases;
        if (!extract_phases(rel, phases)) return false;
        if (phases.empty()) continue;            // joint locked in this animation
        return true;
    }
    return false;
}

Candidate solve_configuration(const JointMotion& motion, const Options&)
{
    Candidate c;
    std::vector<Phase> merged;

    for (const auto& rel : motion.rel_by_animation) {
        std::vector<Phase> phases;
        if (!extract_phases(rel, phases)) continue;
        for (const Phase& p : phases) {
            auto it = std::find_if(merged.begin(), merged.end(),
                                   [&](const Phase& m) { return m.axis == p.axis; });
            if (it == merged.end()) merged.push_back(p);
            else { it->lo = std::min(it->lo, p.lo); it->hi = std::max(it->hi, p.hi); }
        }
    }

    if (merged.empty()) return c;                // locked joint: no stages

    c.dof = static_cast<int>(std::min<size_t>(merged.size(), 3));
    c.summary.valid = true;
    for (int s = 0; s < c.dof; ++s) {
        c.stage[s]               = rotate_stage_for(merged[s].axis);
        c.axis[s]                = merged[s].axis;
        c.summary.min_value[s]   = merged[s].lo;
        c.summary.max_value[s]   = merged[s].hi;
    }
    return c;
}

} // namespace detail
} // namespace ChaCha
```

- [ ] **Step 5: Wire it into `analyze()`**

In `src/chacha_analyzer.cpp`, replace the rotation candidate selection loop:

```cpp
    for (const auto& [node, motion] : rotation_by_node) {
        // An artist-authored configuration animation is a direct specification:
        // it declares the stage set, order and range, so no search is run.
        Candidate c = is_configuration_motion(motion, options)
                    ? solve_configuration(motion, options)
                    : select_candidate(motion, options);
        if (!c.summary.valid || c.dof == 0) continue;
        candidate_by_node[node] = c;
        for (int s = 0; s < c.dof; ++s) {
            RawStage r;
            r.type             = c.stage[s];
            r.min_value        = c.summary.min_value[s];
            r.max_value        = c.summary.max_value[s];
            r.initial_value    = 0.0f;
            r.max_velocity     = c.summary.max_velocity[s];
            r.max_acceleration = c.summary.max_acceleration[s];
            raw_by_node[node].push_back(r);
        }
    }
```

- [ ] **Step 6: Run to verify it passes**

Run: `cmake --build build && ctest --test-dir build --output-on-failure`
Expected: PASS, all tests.

- [ ] **Step 7: Commit**

```bash
git add src/chacha_config.cpp src/chacha_internal.h src/chacha_analyzer.cpp tests/chacha_config_test.cpp
git commit -m "feat: honour AGI configuration animations as authoritative"
```

---

### Task 12: Shared naming helpers

Both consumers need identical AGI name sanitisation, so it lives in the library.

**Files:**
- Create: `include/chacha_naming.h`, `src/chacha_naming.cpp`
- Modify: `CMakeLists.txt` (PUBLIC_HEADER list)
- Test: `tests/chacha_naming_test.cpp`

**Interfaces:**
- Consumes: `StageType`
- Produces:
  - `std::string sanitize_articulation_name(std::string_view raw, std::vector<std::string>& taken)`
  - `std::string stage_name_for(StageType type, int occurrence)`

- [ ] **Step 1: Write the failing test**

Create `tests/chacha_naming_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include "chacha_naming.h"

using namespace ChaCha;

TEST(Naming, WhitespaceIsRemovedFromArticulationNames)
{
    std::vector<std::string> taken;
    EXPECT_EQ(sanitize_articulation_name("Left Arm", taken), "Left_Arm");
    EXPECT_EQ(sanitize_articulation_name("  Spine 03 ", taken), "Spine_03");
}

TEST(Naming, DuplicateArticulationNamesGetSuffixes)
{
    std::vector<std::string> taken;
    EXPECT_EQ(sanitize_articulation_name("Bone", taken), "Bone");
    EXPECT_EQ(sanitize_articulation_name("Bone", taken), "Bone_2");
    EXPECT_EQ(sanitize_articulation_name("Bone", taken), "Bone_3");
}

TEST(Naming, EmptyNameFallsBackToAPlaceholder)
{
    std::vector<std::string> taken;
    EXPECT_FALSE(sanitize_articulation_name("", taken).empty());
}

// Proper-Euler charts repeat an axis, so stage names must disambiguate.
TEST(Naming, RepeatedStageTypesGetDistinctNames)
{
    EXPECT_EQ(stage_name_for(StageType::zRotate, 0), "zRotate");
    EXPECT_EQ(stage_name_for(StageType::zRotate, 1), "zRotate2");
    EXPECT_EQ(stage_name_for(StageType::xTranslate, 2), "xTranslate3");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ctest --test-dir build -R Naming --output-on-failure`
Expected: FAIL — `chacha_naming.h` does not exist.

- [ ] **Step 3: Implement**

Create `include/chacha_naming.h`:

```cpp
#ifndef CHACHA_NAMING_H
#define CHACHA_NAMING_H

#include "chacha_stage.h"
#include <string>
#include <string_view>
#include <vector>

namespace ChaCha {

/// AGI requires articulation names to match ^[^\s]+$ and be unique per model.
/// Collapses whitespace to '_' and appends _2, _3, ... on collision. The name
/// chosen is appended to `taken`.
std::string sanitize_articulation_name(std::string_view raw, std::vector<std::string>& taken);

/// AGI requires stage names to be unique within an articulation. Proper-Euler
/// charts repeat an axis, so `occurrence` disambiguates: 0 -> "zRotate",
/// 1 -> "zRotate2".
std::string stage_name_for(StageType type, int occurrence);

} // namespace ChaCha

#endif // CHACHA_NAMING_H
```

Create `src/chacha_naming.cpp`:

```cpp
#include "chacha_naming.h"
#include <algorithm>
#include <cctype>

namespace ChaCha {

std::string sanitize_articulation_name(std::string_view raw, std::vector<std::string>& taken)
{
    std::string base;
    base.reserve(raw.size());
    for (char c : raw) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!base.empty() && base.back() != '_') base.push_back('_');
        } else {
            base.push_back(c);
        }
    }
    while (!base.empty() && base.back() == '_') base.pop_back();
    if (base.empty()) base = "articulation";

    std::string candidate = base;
    int suffix = 1;
    while (std::find(taken.begin(), taken.end(), candidate) != taken.end())
        candidate = base + "_" + std::to_string(++suffix);

    taken.push_back(candidate);
    return candidate;
}

std::string stage_name_for(StageType type, int occurrence)
{
    std::string name = stage_type_name(type);
    if (occurrence > 0) name += std::to_string(occurrence + 1);
    return name;
}

} // namespace ChaCha
```

Add `chacha_naming.h` to the `PUBLIC_HEADER` property in `CMakeLists.txt`.

- [ ] **Step 4: Run to verify it passes**

Run: `cmake --build build && ctest --test-dir build -R Naming --output-on-failure`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add include/chacha_naming.h src/chacha_naming.cpp CMakeLists.txt tests/chacha_naming_test.cpp
git commit -m "feat: shared AGI name sanitisation helpers"
```

---

### Task 13: gltfRepackager adoption and conformance

**Files:**
- Modify: `gltfRepackager/gltfRepackager/src/steps/infer_articulations.cpp`

**Interfaces:**
- Consumes: new `analyze()` (Task 9), `chacha_naming.h` (Task 12)
- Produces: schema-conformant AGI output in degrees

This task has no unit test of its own; Task 15 covers it end to end. Verification here is a
build plus a manual run.

- [ ] **Step 1: Populate `channel.animation` and build the animations span**

In `InferArticulations`, replace the channel collection loop so it records the animation
index and builds the parallel `Animation` array:

```cpp
    std::vector<ChaCha::Animation> chacha_animations;
    chacha_animations.reserve(doc.animations.size());
    for (auto const& a : doc.animations)
        chacha_animations.push_back(ChaCha::Animation{a.name});

    // ... inside the loop over animations, tracking the index:
    for (uint32_t anim_index = 0; anim_index < doc.animations.size(); ++anim_index)
    {
        auto& animation = doc.animations[anim_index];
        for (auto const& channel : animation.channels)
        {
            // ... existing decoding ...
            ac.node      = node;
            ac.animation = static_cast<int>(anim_index);
            // ...
        }
    }
```

Delete the `drop_anim` pre-pass that matched `agiconfiguration` / `animconfiguration`:
ChaCha now selects configuration animations itself from the names. Keep the removal of
those animations from the document, but derive the set from a case-insensitive `"agi "`
prefix test so it matches the actual authored names (`AGI Configuration`,
`AGI configuration`, `AGI Configuration.001`).

- [ ] **Step 2: Call the new signature**

```cpp
    std::vector<ChaCha::Diagnostic> diagnostics;
    auto articulations = ChaCha::analyze(
        std::span<const ChaCha::AnimationChannel>(chacha_channels),
        std::span<const ChaCha::Animation>(chacha_animations),
        skeleton,
        ChaCha::Options{},
        {},
        &diagnostics);

    for (auto const& d : diagnostics)
        std::fprintf(stderr, "chacha: node %d animation %d reported issue %d\n",
                     d.node, d.animation, static_cast<int>(d.kind));
```

- [ ] **Step 3: Populate `rest_scales`**

`GetNodeRest` already decomposes scale but discards it. Add a `glm::vec3 scale{1,1,1}`
field to `RestTRS`, populate it in all three branches, collect into a
`std::vector<glm::vec3> rest_scales(N, glm::vec3(1))`, and add it to the `ChaCha::Skeleton`
aggregate initialiser.

- [ ] **Step 4: Fix units and naming at emission**

Replace the articulation emission loop:

```cpp
    static constexpr float kRadToDeg = 57.29577951308232f;

    auto is_rotation = [](ChaCha::StageType t) {
        return t == ChaCha::StageType::xRotate
            || t == ChaCha::StageType::yRotate
            || t == ChaCha::StageType::zRotate;
    };

    std::vector<std::string> taken_names;
    std::vector<int>         type_occurrence;

    for (auto& ca : articulations)
    {
        if (ca.node < 0 || ca.node >= N) continue;

        AGIArticulation art;
        art.name = ChaCha::sanitize_articulation_name(doc.nodes[ca.node].name, taken_names);

        art.pointingVector = {
            ca.pointing_vector.x, ca.pointing_vector.y, ca.pointing_vector.z,
        };

        type_occurrence.assign(9, 0);
        art.stages.reserve(ca.stages.size());
        for (auto const& cs : ca.stages)
        {
            const int slot  = static_cast<int>(cs.type);
            const int occur = type_occurrence[slot]++;
            // AGI defines rotation stages in degrees; translation in metres;
            // scale as a bare multiplicative factor.
            const float conv = is_rotation(cs.type) ? kRadToDeg : 1.0f;

            AGIStage s;
            s.name         = ChaCha::stage_name_for(cs.type, occur);
            s.type         = StageEnum(cs.type);
            s.minimumValue = cs.min_value     * conv;
            s.maximumValue = cs.max_value     * conv;
            s.initialValue = cs.initial_value * conv;

            if (cs.max_velocity > 0.0f)
                s.extensionsAndExtras["extras"]["chachaMaximumSpeed"] =
                    cs.max_velocity * conv;
            if (cs.max_acceleration > 0.0f)
                s.extensionsAndExtras["extras"]["chachaMaximumAcceleration"] =
                    cs.max_acceleration * conv;

            art.stages.push_back(std::move(s));
        }

        extensions.nodes[ca.node].extensions.AGI_articulations.articulationName = art.name;
        store.articulations.push_back(std::move(art));
    }
```

Add `#include <chacha_naming.h>`. If `AGIStage` has no `extensionsAndExtras` member,
add one following the pattern used elsewhere in `fx/extensions/agi_articulation.h`.

- [ ] **Step 5: Build and spot-check**

```bash
cmake --build build
./build/gltfRepackager "/mnt/Passport/Libraries/Spehleon/chacha/testdata/treefrog.glb" -o /tmp/treefrog-out.glb
python3 -c "
import struct, json
f = open('/tmp/treefrog-out.glb','rb'); f.read(12)
ln, ty = struct.unpack('<II', f.read(8)); j = json.loads(f.read(ln))
arts = j['extensions']['AGI_articulations']['articulations']
print('articulations:', len(arts))
print(json.dumps(arts[0], indent=2)[:600])
"
```

Expected: every stage has a non-empty `name`; rotation `minimumValue` / `maximumValue` are
in the tens of degrees, not hundredths of a radian; no articulation name contains a space.

- [ ] **Step 6: Commit**

```bash
cd /mnt/Passport/Libraries/gltfRepackager
git add gltfRepackager/src/steps/infer_articulations.cpp
git commit -m "fix: AGI conformance and degree conversion in infer_articulations"
```

---

### Task 14: tonton-example adoption, node space, and conformance

**Files:**
- Modify: `tonton-example/src/chacha_fxgltf_bridge.h`, `chacha_fxgltf_bridge.cpp`, `chacha_main.cpp`

**Interfaces:**
- Consumes: new `analyze()` (Task 9), `chacha_naming.h` (Task 12)
- Produces: node-space extraction and conformant output

- [ ] **Step 1: Move `extract_skeleton` to node space**

AGI articulations are per node and glTF animation channels target nodes, so skin-joint
space is wrong. Replace `extract_skeleton` with a signature that takes no skin index and
builds arrays over all `doc.nodes`:

```cpp
ExtractedSkeleton extract_skeleton(const fx::gltf::Document& doc)
{
    ExtractedSkeleton result;
    const size_t n = doc.nodes.size();

    result.parents.assign(n, -1);
    result.rest_rotations.assign(n, glm::quat(1, 0, 0, 0));
    result.rest_translations.assign(n, glm::vec3(0));
    result.rest_scales.assign(n, glm::vec3(1));

    for (size_t i = 0; i < n; ++i)
        for (auto child : doc.nodes[i].children)
            if (child < n) result.parents[child] = static_cast<int>(i);

    for (size_t i = 0; i < n; ++i) {
        const auto& node = doc.nodes[i];
        if (node.matrix != fx::gltf::defaults::IdentityMatrix) {
            glm::mat4 mat;
            std::memcpy(&mat[0].x, node.matrix.data(), sizeof(glm::mat4));
            glm::vec3 scale, translation, skew;
            glm::quat rotation;
            glm::vec4 perspective;
            glm::decompose(mat, scale, rotation, translation, skew, perspective);
            result.rest_rotations[i]    = rotation;
            result.rest_translations[i] = translation;
            result.rest_scales[i]       = scale;
        } else {
            result.rest_rotations[i] = glm::quat(
                node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
            result.rest_translations[i] = glm::vec3(
                node.translation[0], node.translation[1], node.translation[2]);
            result.rest_scales[i] = glm::vec3(
                node.scale[0], node.scale[1], node.scale[2]);
        }
    }
    return result;
}
```

Add `std::vector<glm::vec3> rest_scales;` to `ExtractedSkeleton` in the header, drop
`joint_nodes`, and include `rest_scales` in `as_skeleton()`.

- [ ] **Step 2: Record the animation index during extraction**

In `extract_animation_channels`, set `ch.animation = static_cast<int>(anim_idx);` alongside
`ch.node`, and build a parallel `std::vector<ChaCha::Animation>` plus owned name storage:

```cpp
    // ExtractedAnimations gains:
    //   std::vector<std::string>        name_storage;
    //   std::vector<ChaCha::Animation>  animations;
    result.name_storage.reserve(doc.animations.size());
    for (const auto& a : doc.animations) result.name_storage.push_back(a.name);
    for (const auto& s : result.name_storage)
        result.animations.push_back(ChaCha::Animation{s});
```

`name_storage` must be filled completely before any `string_view` is taken, or reallocation
invalidates the views.

- [ ] **Step 3: Delete the skin remap in `chacha_main.cpp`**

Remove the `node_to_joint` map and the `remapped_channels` loop entirely (currently lines
161–178), along with the skin discovery block that computes `skin_index`. Call:

```cpp
        auto extracted_anims = ChaChaFxGltf::extract_animation_channels(doc);
        auto extracted_skel  = ChaChaFxGltf::extract_skeleton(doc);

        std::vector<ChaCha::Diagnostic> diagnostics;
        auto articulations = ChaCha::analyze(
            extracted_anims.channels,
            extracted_anims.animations,
            extracted_skel.as_skeleton(),
            chacha_opts,
            {},
            &diagnostics);

        for (const auto& d : diagnostics)
            std::cerr << "warning: node " << d.node << " animation " << d.animation
                      << " kind " << static_cast<int>(d.kind) << "\n";
```

Update the two `write_agi_articulations` / `print_articulations_json` call sites to drop the
`joint_nodes` argument.

- [ ] **Step 4: Fix emission**

In `write_agi_articulations`, drop the `joint_nodes` parameter (node indices are now direct),
use the real pointing vector, add stage names, and move speed and acceleration into `extras`:

```cpp
    std::vector<std::string> taken_names;

    for (const auto& artic : articulations) {
        const uint32_t node_idx = static_cast<uint32_t>(artic.node);
        std::string source = (node_idx < doc.nodes.size() && !doc.nodes[node_idx].name.empty())
            ? doc.nodes[node_idx].name
            : ("joint_" + std::to_string(artic.node));
        const std::string name = ChaCha::sanitize_articulation_name(source, taken_names);

        nlohmann::json stages_json = nlohmann::json::array();
        int occurrence[9] = {0,0,0,0,0,0,0,0,0};

        for (const auto& stage : artic.stages) {
            const int   slot  = static_cast<int>(stage.type);
            const int   occur = occurrence[slot]++;
            const float conv  = is_rotation_stage(stage.type) ? rad_to_deg : 1.0f;

            nlohmann::json sj;
            sj["name"]         = ChaCha::stage_name_for(stage.type, occur);
            sj["type"]         = stage_type_to_agi_type(stage.type);
            sj["minimumValue"] = stage.min_value     * conv;
            sj["maximumValue"] = stage.max_value     * conv;
            sj["initialValue"] = stage.initial_value * conv;
            if (stage.max_velocity > 0)
                sj["extras"]["chachaMaximumSpeed"] = stage.max_velocity * conv;
            if (stage.max_acceleration > 0)
                sj["extras"]["chachaMaximumAcceleration"] = stage.max_acceleration * conv;
            stages_json.push_back(sj);
        }

        nlohmann::json entry;
        entry["name"]           = name;
        entry["stages"]         = stages_json;
        entry["pointingVector"] = {artic.pointing_vector.x,
                                   artic.pointing_vector.y,
                                   artic.pointing_vector.z};
        agi_array.push_back(entry);

        if (node_idx < doc.nodes.size())
            doc.nodes[node_idx].extensionsAndExtras["extensions"]
                ["AGI_articulations"]["articulationName"] = name;
    }
```

Apply the same stage-name and unit treatment to `print_articulations_json`, and add
`#include "chacha_naming.h"`.

Change `remove_agi_animations` to match a case-insensitive `"agi "` prefix rather than
`"agi_"`, so it matches the authored names.

- [ ] **Step 5: Build and spot-check**

```bash
cmake --build build
./build/chacha-example "/mnt/Passport/Libraries/Spehleon/chacha/testdata/treefrog.glb" --stdout
```

Expected: articulations for the joints the configuration animation exercises, with ranges
near ±13.4 degrees, and no articulations for joints held at rest.

- [ ] **Step 6: Commit**

```bash
cd /mnt/Passport/Libraries/Spehleon/tonton-example
git add src/chacha_fxgltf_bridge.h src/chacha_fxgltf_bridge.cpp src/chacha_main.cpp
git commit -m "fix: node-space extraction and AGI conformance in chacha bridge"
```

---

### Task 15: Integration tests against real models

Unit tests use synthetic data; this task exercises the real corpus. It lives in
tonton-example because `fx-gltf` is already wired up there and ChaCha must stay
parser-free.

**Files:**
- Create: `tonton-example/tests/chacha_integration_tests.cpp`
- Modify: `tonton-example/CMakeLists.txt`

**Interfaces:**
- Consumes: `ChaChaFxGltf::extract_animation_channels`, `extract_skeleton` (Task 14)
- Produces: nothing consumed downstream

- [ ] **Step 1: Write the failing test**

Create `tonton-example/tests/chacha_integration_tests.cpp`:

```cpp
#include <gtest/gtest.h>
#include "fx/gltf.h"
#include "chacha.h"
#include "chacha_fxgltf_bridge.h"
#include <glm/gtc/quaternion.hpp>
#include <filesystem>
#include <map>
#include <cmath>

namespace {

fx::gltf::Document load(const char* name)
{
    std::filesystem::path p = std::filesystem::path(CHACHA_TESTDATA_DIR) / name;
    return fx::gltf::LoadFromBinary(p);
}

struct Analysis {
    ChaChaFxGltf::ExtractedAnimations anims;
    ChaChaFxGltf::ExtractedSkeleton   skel;
    std::vector<ChaCha::Articulation> articulations;
};

Analysis run(const fx::gltf::Document& doc)
{
    Analysis a;
    a.anims = ChaChaFxGltf::extract_animation_channels(doc);
    a.skel  = ChaChaFxGltf::extract_skeleton(doc);
    a.articulations = ChaCha::analyze(
        a.anims.channels, a.anims.animations, a.skel.as_skeleton());
    return a;
}

const ChaCha::Articulation* find_named(
    const fx::gltf::Document& doc,
    const std::vector<ChaCha::Articulation>& arts,
    const std::string& node_name)
{
    for (const auto& a : arts)
        if (a.node >= 0 && a.node < static_cast<int>(doc.nodes.size())
            && doc.nodes[a.node].name == node_name)
            return &a;
    return nullptr;
}

} // namespace

// The configuration animation is ground truth: it declares the stage set,
// the order, and the range.
TEST(ChaChaIntegration, TreefrogConfigurationIsHonoured)
{
    auto doc = load("treefrog.glb");
    auto a   = run(doc);

    for (const char* joint : {"Head", "Spine.003"}) {
        const auto* art = find_named(doc, a.articulations, joint);
        ASSERT_NE(art, nullptr) << joint;
        ASSERT_EQ(art->stages.size(), 3u) << joint;
        EXPECT_EQ(art->stages[0].type, ChaCha::StageType::xRotate) << joint;
        EXPECT_EQ(art->stages[1].type, ChaCha::StageType::zRotate) << joint;
        EXPECT_EQ(art->stages[2].type, ChaCha::StageType::yRotate) << joint;
        for (const auto& s : art->stages) {
            EXPECT_NEAR(glm::degrees(s.min_value), -13.4f, 2.0f) << joint;
            EXPECT_NEAR(glm::degrees(s.max_value),  13.4f, 2.0f) << joint;
        }
    }
}

TEST(ChaChaIntegration, TreefrogRestingJointsAreLocked)
{
    auto doc = load("treefrog.glb");
    auto a   = run(doc);
    // Most of the 84 nodes are held at rest for the whole configuration clip.
    EXPECT_LT(a.articulations.size(), doc.nodes.size() / 2);
}

TEST(ChaChaIntegration, ScorpionProducesArticulations)
{
    auto doc = load("emporer scorpion.glb");
    auto a   = run(doc);
    EXPECT_FALSE(a.articulations.empty());
}

// The objective correctness measure: applying the emitted stages at their
// solved angles must reproduce the original keyframes. Requires no
// anatomical ground truth.
TEST(ChaChaIntegration, SophiaRoundTripsWithinTolerance)
{
    auto doc = load("sophia-2_9.glb");
    auto a   = run(doc);
    ASSERT_FALSE(a.articulations.empty());

    std::map<int, const ChaCha::Articulation*> by_node;
    for (const auto& art : a.articulations) by_node[art.node] = &art;

    auto axis_of = [](ChaCha::StageType t) -> int {
        switch (t) {
        case ChaCha::StageType::xRotate: return 0;
        case ChaCha::StageType::yRotate: return 1;
        case ChaCha::StageType::zRotate: return 2;
        default: return -1;
        }
    };

    int    sampled = 0;
    double worst   = 0.0;

    for (const auto& ch : a.anims.channels) {
        if (ch.property != ChaCha::Property::Rotation) continue;
        auto it = by_node.find(ch.node);
        if (it == by_node.end()) continue;

        // Reachability check: the emitted axes must span the observed motion.
        // A joint constrained to fewer axes than it moves in would show up here.
        const glm::quat rest_inv =
            glm::inverse(glm::normalize(a.skel.rest_rotations[ch.node]));

        const int n = static_cast<int>(ch.times.size());
        for (int i = 0; i < n; i += 7) {
            const float* v = ch.values.data() + static_cast<size_t>(i) * 4;
            glm::quat key(v[3], v[0], v[1], v[2]);
            if (glm::dot(key, key) < 1e-8f) continue;
            glm::quat rel = rest_inv * glm::normalize(key);

            // Project onto the emitted axes and measure what is left over.
            glm::quat residual = rel;
            for (const auto& s : it->second->stages) {
                const int ax = axis_of(s.type);
                if (ax < 0) continue;
                glm::vec3 axis(0.0f); axis[ax] = 1.0f;
                const float qv[3] = {residual.x, residual.y, residual.z};
                const float angle = 2.0f * std::atan2(qv[ax], residual.w);
                residual = glm::inverse(glm::angleAxis(angle, axis)) * residual;
            }
            float d = std::fabs(residual.w);
            if (d > 1.0f) d = 1.0f;
            worst = std::max(worst, static_cast<double>(2.0f * std::acos(d)));
            ++sampled;
        }
    }

    EXPECT_GT(sampled, 1000);
    // Reported rather than tightly asserted: mixamo rigs contain genuine
    // three-axis motion at many joints, so this bounds gross failure only.
    std::printf("[sophia] sampled=%d worst residual=%.2f deg\n",
                sampled, worst * 57.29577951);
    EXPECT_LT(worst, 3.2);   // radians; anything near pi indicates a broken fit
}

TEST(ChaChaIntegration, SophiaConstantScaleChannelsProduceNoScaleStages)
{
    auto doc = load("sophia-2_9.glb");
    auto a   = run(doc);
    for (const auto& art : a.articulations)
        for (const auto& s : art.stages)
            EXPECT_FALSE(s.type == ChaCha::StageType::xScale
                      || s.type == ChaCha::StageType::yScale
                      || s.type == ChaCha::StageType::zScale)
                << "constant unit scale should filter out";
}

TEST(ChaChaIntegration, SophiaDofHistogramIsReported)
{
    auto doc = load("sophia-2_9.glb");
    auto a   = run(doc);
    int hist[4] = {0, 0, 0, 0};
    float worst_conditioning_proxy = 0.0f;
    for (const auto& art : a.articulations) {
        if (art.dof_count < 4) hist[art.dof_count]++;
        worst_conditioning_proxy = std::max(worst_conditioning_proxy, art.fit_residual_rad);
    }
    std::printf("[sophia] dof histogram 0/1/2/3 = %d/%d/%d/%d, worst residual %.3f rad\n",
                hist[0], hist[1], hist[2], hist[3], worst_conditioning_proxy);
    EXPECT_GT(hist[1] + hist[2] + hist[3], 0);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build`
Expected: FAIL — `CHACHA_TESTDATA_DIR` undefined and no test target includes the file.

- [ ] **Step 3: Wire the target**

In `tonton-example/CMakeLists.txt`, add after the `chacha-example` block:

```cmake
if(BUILD_TESTING AND GTest_FOUND)
    add_executable(chacha-tests
        tests/chacha_integration_tests.cpp
        src/chacha_fxgltf_bridge.cpp
    )
    target_include_directories(chacha-tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/modules/fx-gltf/include
        ${CMAKE_SOURCE_DIR}/modules/fx-gltf/test/thirdparty
        ${CMAKE_SOURCE_DIR}/modules/chacha/include
    )
    target_link_libraries(chacha-tests PRIVATE
        chacha fx-gltf GTest::gtest GTest::gtest_main Threads::Threads)
    if(glm_FOUND)
        target_link_libraries(chacha-tests PRIVATE glm::glm)
    endif()
    target_compile_definitions(chacha-tests PRIVATE
        CHACHA_TESTDATA_DIR="${CMAKE_SOURCE_DIR}/modules/chacha/testdata")
    include(GoogleTest)
    gtest_discover_tests(chacha-tests)
endif()
```

- [ ] **Step 4: Run to verify it passes**

Run: `cmake -S . -B build && cmake --build build && ctest --test-dir build -R ChaChaIntegration --output-on-failure`
Expected: PASS, 6 tests. The `[sophia]` report lines appear in the output.

If `TreefrogConfigurationIsHonoured` fails on stage *order*, the phase-ordering logic in
`solve_configuration` is at fault, not the solver — phases must be emitted in the order
they occur in time.

- [ ] **Step 5: Commit**

```bash
git add tests/chacha_integration_tests.cpp CMakeLists.txt
git commit -m "test: integration coverage against treefrog, scorpion and sophia"
```

---

### Task 16: Performance guard and documentation

**Files:**
- Modify: `tonton-example/tests/chacha_integration_tests.cpp`
- Modify: `chacha/CLAUDE.md`, `chacha/README.md`

**Interfaces:**
- Consumes: everything
- Produces: nothing consumed downstream

- [ ] **Step 1: Add the timing test**

Append to `tonton-example/tests/chacha_integration_tests.cpp`:

```cpp
#include <chrono>

// The search is 49 joints x 12 charts x 88 animations x ~157 frames on this
// model. Guards against a silent order-of-magnitude regression, not against
// small drift.
TEST(ChaChaIntegration, SophiaAnalysisCompletesInReasonableTime)
{
    auto doc   = load("sophia-2_9.glb");
    auto anims = ChaChaFxGltf::extract_animation_channels(doc);
    auto skel  = ChaChaFxGltf::extract_skeleton(doc);

    const auto start = std::chrono::steady_clock::now();
    auto arts = ChaCha::analyze(anims.channels, anims.animations, skel.as_skeleton());
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    std::printf("[sophia] analyze() took %lld ms for %zu articulations\n",
                static_cast<long long>(elapsed), arts.size());
    EXPECT_LT(elapsed, 120000);
}
```

- [ ] **Step 2: Run it**

Run: `ctest --test-dir build -R SophiaAnalysisCompletes --output-on-failure`
Expected: PASS. Record the reported time — if it exceeds roughly 30 s, parallelising the
per-joint loop in `analyze()` is the documented remedy.

- [ ] **Step 3: Update the documentation**

In `chacha/CLAUDE.md`, replace the "Processing Pipeline" section with the new pipeline from
the design doc, replace the "Multi-axis rotation decomposition" decision (swing-twist is
gone) with the 12-chart search, and update "File Structure" to match the current `src/`
contents. Add to the "Limitations" section:

```markdown
- ChaCha reports *observed* ROM, not *possible* ROM. If no animation fully extends a
  joint, the range is underestimated.
- When the scan narrows to `AGI ` configuration animations, velocity and acceleration
  reflect the sweep speed the artist authored, which is usually a uniform ramp rather
  than a real speed limit. Ranges and stage order are the trustworthy outputs in that
  mode.
- Effort and torque limits are out of scope: they require mass and inertia and depend on
  the motion of the whole subtree.
- A trajectory that crosses the singular set of all 12 charts cannot be represented by
  three stages. A redundant fourth stage would be required; this is not implemented.
```

Update `README.md` to show the new `analyze()` signature.

- [ ] **Step 4: Commit**

```bash
cd /mnt/Passport/Libraries/Spehleon/tonton-example
git add tests/chacha_integration_tests.cpp
git commit -m "test: performance guard for sophia analysis"

cd /mnt/Passport/Libraries/Spehleon/chacha
git add CLAUDE.md README.md
git commit -m "docs: update pipeline and limitations for the solver rework"
```

---

## Self-Review Notes

**Spec coverage.** Every section of the design doc maps to a task: pipeline (10, 11),
exact solver (2, 3), DP (4), anchoring (5), reduced-DOF (7), scoring (8), configuration
path (11), velocity/acceleration (6), API (9), behavioural changes to ranges, initial
values and scale (6, 10), consumer conformance (13, 14), testing (1, 15, 16), deletions
(6, 10).

**Known gap, carried forward from the design review.** No task asserts the 12-chart search
against a known-correct answer on *real* data, because no available model provides one.
Coverage there rests on the synthetic tests in Task 8 plus the round-trip bound in Task 15.
The treefrog assertions test the configuration path, which bypasses the search entirely.

**Deferred.** The four-stage fallback is not implemented. Task 15's DOF histogram and
Task 16's timing output are the evidence that would justify building it.
