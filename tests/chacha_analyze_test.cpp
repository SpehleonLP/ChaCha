#include <gtest/gtest.h>
#include "chacha.h"
#include "chacha_internal.h"
#include <glm/gtc/quaternion.hpp>
#include <vector>

using namespace ChaCha;
using namespace ChaCha::detail;

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

// --- Review round: Skeleton rest-span validation (Finding 1) ---------------

// A Skeleton whose rest_translations is shorter than parents.size() previously
// segfaulted inside infer_pointing_vectors even for a rotation-only rig with
// no translation channels at all. analyze() must reject the malformed
// Skeleton outright rather than index out of bounds.
TEST(Analyze, EmptyRestTranslationsIsRejectedNotCrashed)
{
    Rig rig;
    rig.trans.clear();

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

    EXPECT_TRUE(analyze(chans, anims, rig.skeleton()).empty());
}

// A Skeleton whose rest_rotations is shorter than parents.size() previously
// segfaulted directly in the rest-relative rotation computation.
TEST(Analyze, EmptyRestRotationsIsRejectedNotCrashed)
{
    Rig rig;
    rig.rest.clear();

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

    EXPECT_TRUE(analyze(chans, anims, rig.skeleton()).empty());
}

// --- Review round: repeated stage type at the public API (Finding 2) -------

// A proper-Euler winner's slot 0 and slot 2 share a StageType by
// construction. analyze() must emit both, in slot order, with independently
// tracked ranges -- never deduped/keyed by StageType, and never reordered.
TEST(Analyze, ProperEulerRepeatedStageSurvivesInSlotOrder)
{
    const Chart* proper = nullptr;
    for (const auto& c : all_charts())
        if (c.proper) { proper = &c; break; }
    ASSERT_NE(proper, nullptr);

    Rig rig;
    std::vector<float> times;
    std::vector<float> values;
    const int n = 40;
    for (int i = 0; i <= n; ++i) {
        const float f = static_cast<float>(i);
        const float angle[3] = {
            glm::radians(f * 1.0f),          // slot 0 (repeated axis, 1st occurrence): 0..40 deg
            glm::radians(f * 0.7f + 5.0f),   // slot 1 (middle, different axis): 5..33 deg
            glm::radians(f * 0.15f),         // slot 2 (repeated axis, 2nd occurrence): 0..6 deg
        };
        glm::quat q = compose_chart(*proper, angle);
        values.insert(values.end(), {q.x, q.y, q.z, q.w});
        times.push_back(f / 30.0f);
    }

    AnimationChannel ch;
    ch.node = 0; ch.animation = 0; ch.property = Property::Rotation;
    ch.times = times; ch.values = values;

    std::vector<AnimationChannel> chans{ch};
    std::vector<Animation> anims{Animation{"rom"}};

    auto out = analyze(chans, anims, rig.skeleton());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].stages.size(), 3u);

    EXPECT_EQ(out[0].stages[0].type, out[0].stages[2].type)
        << "repeated proper-Euler stage type must survive, not be deduped";
    EXPECT_NE(out[0].stages[0].type, out[0].stages[1].type);

    // Distinct, independently tracked ranges per occurrence -- and in slot
    // order, not reversed (which would swap which occurrence carries the
    // wide vs. narrow range and mis-attribute values to the wrong axis).
    const float range0 = out[0].stages[0].max_value - out[0].stages[0].min_value;
    const float range2 = out[0].stages[2].max_value - out[0].stages[2].min_value;
    EXPECT_GT(range0, glm::radians(20.0f));
    EXPECT_LT(range2, glm::radians(15.0f));
}

// --- Review round: initial_value clamp for translation/scale (Finding 3) ---

TEST(Analyze, TranslationInitialValueClampsWhenRangeExcludesRest)
{
    Rig rig;   // rest_translations[0] == vec3(0)
    std::vector<float> times{0.0f, 1.0f};
    std::vector<float> values{0, 1.0f, 0,   0, 2.0f, 0};   // y: 1.0 -> 2.0, never revisits 0
    AnimationChannel ch;
    ch.node = 0; ch.animation = 0; ch.property = Property::Translation;
    ch.times = times; ch.values = values;

    std::vector<AnimationChannel> chans{ch};
    std::vector<Animation> anims{Animation{"lift"}};

    auto out = analyze(chans, anims, rig.skeleton());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].stages.size(), 1u);
    EXPECT_EQ(out[0].stages[0].type, StageType::yTranslate);
    EXPECT_NEAR(out[0].stages[0].min_value, 1.0f, 1e-4f);
    EXPECT_NEAR(out[0].stages[0].max_value, 2.0f, 1e-4f);
    // Rest (0) lies outside [1, 2]; the clamp must pull initial_value to the
    // nearest bound rather than leave it at an out-of-range 0.
    EXPECT_NEAR(out[0].stages[0].initial_value, 1.0f, 1e-4f);
}

TEST(Analyze, ScaleInitialValueClampsWhenRangeExcludesRest)
{
    Rig rig;   // rest_scales[0] == vec3(1)
    std::vector<float> times{0.0f, 1.0f};
    std::vector<float> values{2.0f, 1, 1,   3.0f, 1, 1};   // x: 2.0 -> 3.0, never revisits 1
    AnimationChannel ch;
    ch.node = 0; ch.animation = 0; ch.property = Property::Scale;
    ch.times = times; ch.values = values;

    std::vector<AnimationChannel> chans{ch};
    std::vector<Animation> anims{Animation{"grow"}};

    auto out = analyze(chans, anims, rig.skeleton());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].stages.size(), 1u);
    EXPECT_EQ(out[0].stages[0].type, StageType::xScale);
    EXPECT_NEAR(out[0].stages[0].min_value, 2.0f, 1e-4f);
    EXPECT_NEAR(out[0].stages[0].max_value, 3.0f, 1e-4f);
    // Rest (1) lies outside [2, 3]; the clamp must pull initial_value in.
    EXPECT_NEAR(out[0].stages[0].initial_value, 2.0f, 1e-4f);
}

// --- Review round: previously-untested public behaviours (Finding 4) ------

TEST(Analyze, TranslationPropertyEndToEnd)
{
    Rig rig;
    std::vector<float> times{0.0f, 0.5f, 1.0f};
    std::vector<float> values{0, 0, 0,   0, 0.1f, 0,   0, 0.3f, 0};
    AnimationChannel ch;
    ch.node = 0; ch.animation = 0; ch.property = Property::Translation;
    ch.times = times; ch.values = values;

    std::vector<AnimationChannel> chans{ch};
    std::vector<Animation> anims{Animation{"reach"}};

    auto out = analyze(chans, anims, rig.skeleton());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].stages.size(), 1u);
    EXPECT_EQ(out[0].stages[0].type, StageType::yTranslate);
    EXPECT_NEAR(out[0].stages[0].min_value, 0.0f, 1e-4f);
    EXPECT_NEAR(out[0].stages[0].max_value, 0.3f, 1e-4f);
}

TEST(Analyze, CubicSplineStrideAndOffsetAreCorrect)
{
    Rig rig;
    std::vector<float> times{0.0f, 0.5f, 1.0f};
    std::vector<float> values;
    for (float deg : {30.0f, 45.0f, 60.0f}) {
        glm::quat q = glm::angleAxis(glm::radians(deg), glm::vec3(1, 0, 0));
        values.insert(values.end(), {0, 0, 0, 0});            // in-tangent (unused)
        values.insert(values.end(), {q.x, q.y, q.z, q.w});    // value
        values.insert(values.end(), {0, 0, 0, 0});            // out-tangent (unused)
    }
    AnimationChannel ch;
    ch.node = 0; ch.animation = 0; ch.property = Property::Rotation;
    ch.interp = InterpolationType::CubicSpline;
    ch.times = times; ch.values = values;

    std::vector<AnimationChannel> chans{ch};
    std::vector<Animation> anims{Animation{"walk"}};

    auto out = analyze(chans, anims, rig.skeleton());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].stages.size(), 1u);
    EXPECT_EQ(out[0].stages[0].type, StageType::xRotate);
    EXPECT_NEAR(glm::degrees(out[0].stages[0].min_value), 30.0f, 1.0f);
    EXPECT_NEAR(glm::degrees(out[0].stages[0].max_value), 60.0f, 1.0f);
}

TEST(Analyze, TruncatedCubicSplineBufferYieldsMalformedValuesDiagnostic)
{
    Rig rig;
    std::vector<float> times{0.0f, 0.5f, 1.0f};
    // CubicSpline rotation needs 3 keys * 12 floats/key; supply values for
    // only 2 keys so the buffer is short of what `times` promises.
    std::vector<float> values(2 * 12, 0.0f);
    AnimationChannel ch;
    ch.node = 0; ch.animation = 0; ch.property = Property::Rotation;
    ch.interp = InterpolationType::CubicSpline;
    ch.times = times; ch.values = values;

    std::vector<AnimationChannel> chans{ch};
    std::vector<Animation> anims{Animation{"walk"}};
    std::vector<Diagnostic> diags;
    auto out = analyze(chans, anims, rig.skeleton(), Options{}, {}, &diags);

    EXPECT_TRUE(out.empty());
    ASSERT_FALSE(diags.empty());
    EXPECT_EQ(diags[0].kind, Diagnostic::MalformedValues);
}

namespace {
std::vector<float> agi_test_rotation_values(float max_deg)
{
    std::vector<float> v;
    for (float deg : {0.0f, max_deg}) {
        glm::quat q = glm::angleAxis(glm::radians(deg), glm::vec3(1, 0, 0));
        v.insert(v.end(), {q.x, q.y, q.z, q.w});
    }
    return v;
}
} // namespace

// "walk" swings 0..-20 deg; "AGI rom" swings 0..50 deg -- deliberately
// disjoint (not merely different-magnitude) ranges. If narrowing failed
// silently and both animations were unioned, min_value would be pulled down
// to roughly -20 deg; only correct narrowing keeps min_value at ~0.
// (A same-signed pair like 0..10 vs 0..50 would let a "narrowing disabled"
// bug hide behind max_value alone, since the wider range's max already
// dominates a naive union -- this is deliberately NOT that shape.)
TEST(Analyze, PrioritizeRomAnimationsNarrowsToAgiPrefixedAnimations)
{
    Rig rig;
    std::vector<float> times{0.0f, 1.0f};
    std::vector<float> walk_values = agi_test_rotation_values(-20.0f);
    std::vector<float> agi_values  = agi_test_rotation_values(50.0f);

    AnimationChannel walk;
    walk.node = 0; walk.animation = 0; walk.property = Property::Rotation;
    walk.times = times; walk.values = walk_values;

    AnimationChannel agi;
    agi.node = 0; agi.animation = 1; agi.property = Property::Rotation;
    agi.times = times; agi.values = agi_values;

    std::vector<AnimationChannel> chans{walk, agi};
    std::vector<Animation> anims{Animation{"walk"}, Animation{"AGI rom"}};

    // scan empty, options default (prioritize_rom_animations == true): must
    // auto-narrow to the "AGI "-prefixed animation and ignore "walk"
    // entirely, per the documented contract in chacha.h.
    auto out = analyze(chans, anims, rig.skeleton());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].stages.size(), 1u);
    EXPECT_NEAR(glm::degrees(out[0].stages[0].min_value), 0.0f, 1.0f);
    EXPECT_NEAR(glm::degrees(out[0].stages[0].max_value), 50.0f, 1.0f);
}

TEST(Analyze, ExplicitScanOverridesAgiNarrowing)
{
    Rig rig;
    std::vector<float> times{0.0f, 1.0f};
    std::vector<float> walk_values = agi_test_rotation_values(-20.0f);
    std::vector<float> agi_values  = agi_test_rotation_values(50.0f);

    AnimationChannel walk;
    walk.node = 0; walk.animation = 0; walk.property = Property::Rotation;
    walk.times = times; walk.values = walk_values;

    AnimationChannel agi;
    agi.node = 0; agi.animation = 1; agi.property = Property::Rotation;
    agi.times = times; agi.values = agi_values;

    std::vector<AnimationChannel> chans{walk, agi};
    std::vector<Animation> anims{Animation{"walk"}, Animation{"AGI rom"}};

    // An explicit scan is honoured verbatim, overriding the "AGI " auto-
    // narrowing that would otherwise apply.
    std::vector<int> scan{0};
    auto out = analyze(chans, anims, rig.skeleton(), Options{}, scan);
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].stages.size(), 1u);
    EXPECT_NEAR(glm::degrees(out[0].stages[0].min_value), -20.0f, 1.0f);
    EXPECT_NEAR(glm::degrees(out[0].stages[0].max_value), 0.0f, 1.0f);
}

// --- Review round: minor items ---------------------------------------------

// A rest scale near zero must not amplify the reported range to something
// like [1e6, 5e6]; it should fall back to raw values for the affected axis
// and flag it, rather than either blowing up (old epsilon too small) or
// silently substituting raw values with no diagnostic at all (exact zero).
TEST(Analyze, DegenerateRestScaleFallsBackAndFlagsADiagnostic)
{
    Rig rig;
    rig.scale[0] = glm::vec3(1e-7f, 1.0f, 1.0f);

    std::vector<float> times{0.0f, 1.0f};
    std::vector<float> values{0.1f, 1.0f, 1.0f,   0.5f, 1.0f, 1.0f};
    AnimationChannel ch;
    ch.node = 0; ch.animation = 0; ch.property = Property::Scale;
    ch.times = times; ch.values = values;

    std::vector<AnimationChannel> chans{ch};
    std::vector<Animation> anims{Animation{"tiny"}};
    std::vector<Diagnostic> diags;
    auto out = analyze(chans, anims, rig.skeleton(), Options{}, {}, &diags);

    bool found = false;
    for (auto& d : diags) found = found || (d.kind == Diagnostic::DegenerateRestScale);
    EXPECT_TRUE(found);

    ASSERT_EQ(out.size(), 1u);
    bool saw_x_scale = false;
    for (auto& st : out[0].stages) {
        if (st.type != StageType::xScale) continue;
        saw_x_scale = true;
        EXPECT_NEAR(st.min_value, 0.1f, 1e-3f);
        EXPECT_NEAR(st.max_value, 0.5f, 1e-3f);
    }
    EXPECT_TRUE(saw_x_scale);
}

// A channel whose animation index doesn't exist in `animations` -- including
// every channel, when `animations` is empty entirely -- must not be dropped
// silently. Bridge code (Tasks 13/14) building AnimationChannel/Animation
// pairs independently is exactly the caller that could get this wrong.
TEST(Analyze, UnknownAnimationIndexIsFlagged)
{
    Rig rig;
    std::vector<float> times{0.0f, 1.0f};
    std::vector<float> values = agi_test_rotation_values(30.0f);
    AnimationChannel ch;
    ch.node = 0; ch.animation = 5;   // no such animation
    ch.property = Property::Rotation;
    ch.times = times; ch.values = values;

    std::vector<AnimationChannel> chans{ch};
    std::vector<Animation> anims;    // empty
    std::vector<Diagnostic> diags;
    auto out = analyze(chans, anims, rig.skeleton(), Options{}, {}, &diags);

    EXPECT_TRUE(out.empty());
    ASSERT_FALSE(diags.empty());
    EXPECT_EQ(diags[0].kind, Diagnostic::UnknownAnimationIndex);
}

// Stage emission order across properties is a deliberate, fixed contract
// (translate, then scale, then rotate) independent of the order channels
// were supplied in.
TEST(Analyze, StageOrderIsTranslateThenScaleThenRotate)
{
    Rig rig;
    std::vector<float> times{0.0f, 1.0f};

    std::vector<float> trans_values{0, 0, 0,   0, 0.5f, 0};
    AnimationChannel trans_ch;
    trans_ch.node = 0; trans_ch.animation = 0; trans_ch.property = Property::Translation;
    trans_ch.times = times; trans_ch.values = trans_values;

    std::vector<float> scale_values{1, 1, 1,   2, 1, 1};
    AnimationChannel scale_ch;
    scale_ch.node = 0; scale_ch.animation = 0; scale_ch.property = Property::Scale;
    scale_ch.times = times; scale_ch.values = scale_values;

    std::vector<float> rot_values = agi_test_rotation_values(30.0f);
    AnimationChannel rot_ch;
    rot_ch.node = 0; rot_ch.animation = 0; rot_ch.property = Property::Rotation;
    rot_ch.times = times; rot_ch.values = rot_values;

    // Deliberately supplied rotate-then-scale-then-translate, to prove
    // emission order does not follow input channel order.
    std::vector<AnimationChannel> chans{rot_ch, scale_ch, trans_ch};
    std::vector<Animation> anims{Animation{"mixed"}};

    auto out = analyze(chans, anims, rig.skeleton());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].stages.size(), 3u);
    EXPECT_EQ(out[0].stages[0].type, StageType::yTranslate);
    EXPECT_EQ(out[0].stages[1].type, StageType::xScale);
    EXPECT_EQ(out[0].stages[2].type, StageType::xRotate);
}

// resolve_scan's "AGI "-prefix narrowing selects WHICH animations are
// scanned; it says nothing about whether the motion it finds actually has
// configuration shape (one axis moving at a time, returning to rest
// between phases). An animation named "AGI ..." whose rotation genuinely
// mixes two axes at once must fall back to the ordinary chart search
// rather than being forced through solve_configuration, which would
// either reject it outright (dof 0, no articulation at all) or -- if the
// shape-detection gate were ever loosened -- silently emit a fabricated
// single-axis reading for motion that isn't single-axis. Two axes
// simultaneously in motion the whole time is exactly what
// is_configuration_motion's residual gate is supposed to catch and
// reject, forcing select_candidate to run instead.
TEST(Analyze, AgiNamedButNonConfigurationShapedMotionFallsBackToSearch)
{
    Rig rig;
    std::vector<float> times;
    std::vector<float> values;
    for (int i = 0; i <= 40; ++i) {
        const float f = static_cast<float>(i);
        glm::quat q = glm::angleAxis(glm::radians(f), glm::vec3(1, 0, 0))
                    * glm::angleAxis(glm::radians(f * 0.7f), glm::vec3(0, 0, 1));
        values.insert(values.end(), {q.x, q.y, q.z, q.w});
        times.push_back(static_cast<float>(i) / 30.0f);
    }
    AnimationChannel ch;
    ch.node = 0; ch.animation = 0; ch.property = Property::Rotation;
    ch.times = times; ch.values = values;

    std::vector<AnimationChannel> chans{ch};
    std::vector<Animation> anims{Animation{"AGI x-z sweep"}};

    auto out = analyze(chans, anims, rig.skeleton());
    // The chart search finds a real decomposition (at minimum a 2-axis
    // candidate); a locked/empty result here would mean the motion was
    // wrongly forced through the configuration path and rejected instead
    // of falling back.
    ASSERT_EQ(out.size(), 1u);
    EXPECT_FALSE(out[0].stages.empty());
    EXPECT_GE(out[0].dof_count, 2);
}

namespace {
// Builds the values/times for a 3-phase configuration sweep: X, then Z,
// then Y, each a half-sine 0 -> peak -> 0 back to rest. Mirrors
// Config.RecoversStageOrderAndRange's own fixture in chacha_config_test.cpp
// (kept independent rather than shared, since these two files test
// different layers: the pure function there, analyze()'s end-to-end
// wiring and gating here).
void build_xzy_configuration_sweep(std::vector<float>& times, std::vector<float>& values)
{
    const glm::vec3 axes[3] = {glm::vec3(1, 0, 0), glm::vec3(0, 0, 1), glm::vec3(0, 1, 0)};
    int frame = 0;
    for (int phase = 0; phase < 3; ++phase) {
        for (int i = 0; i <= 40; ++i) {
            const float f   = static_cast<float>(i) / 40.0f;
            const float deg = 13.4f * std::sin(f * 2.0f * 3.14159265f);
            glm::quat q = glm::angleAxis(glm::radians(deg), axes[phase]);
            values.insert(values.end(), {q.x, q.y, q.z, q.w});
            times.push_back(frame++ / 24.0f);
        }
    }
}
} // namespace

// The headline claim of this task: an artist-authored configuration
// animation run through the PUBLIC analyze() entry point yields stages in
// the AUTHORED order (X, Z, Y), not the chart search's own order. The
// chart search demonstrably gets this wrong for this exact input (see
// Analyze.OrdinaryAnimationWithConfigurationShapeDoesNotFastPathWithoutAgiNarrowing
// below, which exercises the identical motion without "AGI " narrowing and
// documents what the search actually returns) -- so this is not a
// redundant check against a path that would have gotten it right anyway.
TEST(Analyze, ConfigurationAnimationYieldsAuthoredStageOrderNotSearchOrder)
{
    Rig rig;
    std::vector<float> times, values;
    build_xzy_configuration_sweep(times, values);

    AnimationChannel ch;
    ch.node = 0; ch.animation = 0; ch.property = Property::Rotation;
    ch.times = times; ch.values = values;

    std::vector<AnimationChannel> chans{ch};
    std::vector<Animation> anims{Animation{"AGI configuration"}};

    auto out = analyze(chans, anims, rig.skeleton());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].stages.size(), 3u);
    EXPECT_EQ(out[0].stages[0].type, StageType::xRotate);
    EXPECT_EQ(out[0].stages[1].type, StageType::zRotate);
    EXPECT_EQ(out[0].stages[2].type, StageType::yRotate);
    for (const auto& stage : out[0].stages) {
        EXPECT_NEAR(glm::degrees(stage.min_value), -13.4f, 1.0f);
        EXPECT_NEAR(glm::degrees(stage.max_value),  13.4f, 1.0f);
    }
}

// is_configuration_motion fires on SHAPE ALONE; it must not fire without
// corroborating evidence that the animation is genuinely an artist's
// deliberate specification, because ordinary motion can happen to have
// configuration shape too. This test feeds analyze() the EXACT SAME
// motion as ConfigurationAnimationYieldsAuthoredStageOrderNotSearchOrder
// above, differing only in the animation's name ("walk", not "AGI ..."),
// so resolve_scan never narrows and the fast path's gate
// (agi_narrowed_scan) never opens regardless of shape. The output must
// come from select_candidate's chart search instead, which -- for this
// input -- picks a different (non-authored) stage order: this is observed
// and pinned, not assumed, precisely so this test cannot quietly degrade
// into "any 3-stage output passes."
TEST(Analyze, OrdinaryAnimationWithConfigurationShapeDoesNotFastPathWithoutAgiNarrowing)
{
    Rig rig;
    std::vector<float> times, values;
    build_xzy_configuration_sweep(times, values);

    AnimationChannel ch;
    ch.node = 0; ch.animation = 0; ch.property = Property::Rotation;
    ch.times = times; ch.values = values;

    std::vector<AnimationChannel> chans{ch};
    std::vector<Animation> anims{Animation{"walk"}};   // no "AGI " prefix

    auto out = analyze(chans, anims, rig.skeleton());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].stages.size(), 3u);
    // The chart search's own order for this input, observed directly:
    // axis index order (X, Y, Z), NOT the authored (X, Z, Y) order the
    // "AGI "-narrowed test above reports for byte-identical motion.
    EXPECT_EQ(out[0].stages[0].type, StageType::xRotate);
    EXPECT_EQ(out[0].stages[1].type, StageType::yRotate);
    EXPECT_EQ(out[0].stages[2].type, StageType::zRotate);
}

// Integration-level counterpart to
// Config.MultipleAnimationsUnionRangesButFirstAnimationSetsOrder: a joint
// driven by two separately-authored "AGI "-prefixed clips (as in the
// "AGI Configuration" / "AGI Configuration.001" pair some real models
// use) must report stage order from the first clip and union ranges from
// the second, end to end through the public analyze() entry point and
// resolve_scan's real narrowing -- not just in the pure function tested
// directly in chacha_config_test.cpp.
TEST(Analyze, MultipleAgiAnimationsUnionRangesWithFirstAnimationSettingOrder)
{
    Rig rig;

    auto build_sweep = [](const glm::vec3& axis, float peak_deg,
                           std::vector<float>& times, std::vector<float>& values, int& frame) {
        for (int i = 0; i <= 20; ++i) {
            const float f   = static_cast<float>(i) / 20.0f;
            const float deg = peak_deg * std::sin(f * 3.14159265f);
            glm::quat q = glm::angleAxis(glm::radians(deg), axis);
            values.insert(values.end(), {q.x, q.y, q.z, q.w});
            times.push_back(frame++ / 24.0f);
        }
    };

    std::vector<float> times0, values0;
    int frame0 = 0;
    build_sweep(glm::vec3(0, 0, 1), 5.0f,  times0, values0, frame0);   // Z
    build_sweep(glm::vec3(1, 0, 0), 10.0f, times0, values0, frame0);   // X

    std::vector<float> times1, values1;
    int frame1 = 0;
    build_sweep(glm::vec3(1, 0, 0), 15.0f, times1, values1, frame1);   // X, wider
    build_sweep(glm::vec3(0, 1, 0), 3.0f,  times1, values1, frame1);   // Y, new

    AnimationChannel ch0;
    ch0.node = 0; ch0.animation = 0; ch0.property = Property::Rotation;
    ch0.times = times0; ch0.values = values0;

    AnimationChannel ch1;
    ch1.node = 0; ch1.animation = 1; ch1.property = Property::Rotation;
    ch1.times = times1; ch1.values = values1;

    std::vector<AnimationChannel> chans{ch0, ch1};
    std::vector<Animation> anims{Animation{"AGI Configuration"}, Animation{"AGI Configuration.001"}};

    auto out = analyze(chans, anims, rig.skeleton());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].stages.size(), 3u);
    EXPECT_EQ(out[0].stages[0].type, StageType::zRotate);
    EXPECT_EQ(out[0].stages[1].type, StageType::xRotate);
    EXPECT_EQ(out[0].stages[2].type, StageType::yRotate);
    EXPECT_NEAR(glm::degrees(out[0].stages[1].max_value), 15.0f, 1.0f);
}

// StageType::Invalid is the sentinel Candidate uses for a reduced-DOF
// winner's unused trailing slots (see chacha_internal.h and Task 8). It
// must never reach an emitted Stage: two things enforce that in
// chacha_analyzer.cpp's stage-emission loop --
//   (1) the loop itself only visits slots [0, dof), where dof is exactly
//       how many of a Candidate's slots select_candidate/solve_configuration
//       actually populated, so an unused slot is never even considered, and
//   (2) even if a stray Invalid-typed slot were visited (defence in depth:
//       `if (c.stage[s] == StageType::Invalid) continue;`, plus
//       chacha_filter.cpp's threshold_for_type(Invalid) returning +inf as a
//       last-resort backstop in filter_stages), it is dropped before
//       reaching the returned Articulation.
// This test pins the observable outcome -- no Invalid stage in analyze()'s
// output -- across a 1-DOF, a 2-DOF and a 3-DOF winner, since reduced-DOF
// Candidates are exactly the ones whose trailing slots genuinely hold
// Invalid (a 3-DOF winner never has an unused slot at all).
TEST(Analyze, NeverEmitsInvalidStageType)
{
    auto assert_no_invalid_stage = [](const std::vector<Articulation>& out) {
        for (const auto& art : out)
            for (const auto& stage : art.stages)
                EXPECT_NE(stage.type, StageType::Invalid)
                    << "node " << art.node << " emitted an Invalid-typed stage";
    };

    // 1-DOF winner: pure hinge about X.
    {
        Rig rig;
        std::vector<float> times, values;
        for (int i = 0; i <= 30; ++i) {
            glm::quat q = glm::angleAxis(glm::radians(i * 2.0f), glm::vec3(1, 0, 0));
            values.insert(values.end(), {q.x, q.y, q.z, q.w});
            times.push_back(i / 30.0f);
        }
        AnimationChannel ch;
        ch.node = 0; ch.animation = 0; ch.property = Property::Rotation;
        ch.times = times; ch.values = values;
        std::vector<AnimationChannel> chans{ch};
        std::vector<Animation> anims{Animation{"hinge"}};

        auto out = analyze(chans, anims, rig.skeleton());
        ASSERT_EQ(out.size(), 1u);
        ASSERT_EQ(out[0].dof_count, 1);
        assert_no_invalid_stage(out);
    }

    // 2-DOF winner: two animations whose cross-animation anchoring pins the
    // winner at exactly dof == 2 (same construction as
    // Search.TwoDofCrossAnimationAnchorCollapsesOppositeSidesOfTheWrap).
    {
        Rig rig;
        std::vector<float> times0, values0, times1, values1;
        for (int i = 0; i <= 10; ++i) {
            glm::quat qa = glm::angleAxis(glm::radians(175.0f + i), glm::vec3(1, 0, 0))
                         * glm::angleAxis(glm::radians(static_cast<float>(i)), glm::vec3(0, 0, 1));
            values0.insert(values0.end(), {qa.x, qa.y, qa.z, qa.w});
            times0.push_back(i / 30.0f);

            glm::quat qb = glm::angleAxis(glm::radians(185.0f - i), glm::vec3(1, 0, 0))
                         * glm::angleAxis(glm::radians(static_cast<float>(i)), glm::vec3(0, 0, 1));
            values1.insert(values1.end(), {qb.x, qb.y, qb.z, qb.w});
            times1.push_back(i / 30.0f);
        }
        AnimationChannel ch0;
        ch0.node = 0; ch0.animation = 0; ch0.property = Property::Rotation;
        ch0.times = times0; ch0.values = values0;
        AnimationChannel ch1;
        ch1.node = 0; ch1.animation = 1; ch1.property = Property::Rotation;
        ch1.times = times1; ch1.values = values1;
        std::vector<AnimationChannel> chans{ch0, ch1};
        std::vector<Animation> anims{Animation{"a"}, Animation{"b"}};

        auto out = analyze(chans, anims, rig.skeleton());
        ASSERT_EQ(out.size(), 1u);
        ASSERT_EQ(out[0].dof_count, 2);
        assert_no_invalid_stage(out);
    }

    // 3-DOF winner: general 3-axis motion (no unused slots at all -- included
    // so the test also documents that the full-rank path is unaffected).
    {
        Rig rig;
        std::vector<float> times, values;
        for (int i = 0; i <= 40; ++i) {
            const float f = static_cast<float>(i);
            glm::quat q = glm::angleAxis(glm::radians(f * 1.5f), glm::vec3(1, 0, 0))
                        * glm::angleAxis(glm::radians(f * 2.0f), glm::vec3(0, 1, 0))
                        * glm::angleAxis(glm::radians(f * 1.1f), glm::vec3(0, 0, 1));
            values.insert(values.end(), {q.x, q.y, q.z, q.w});
            times.push_back(f / 30.0f);
        }
        AnimationChannel ch;
        ch.node = 0; ch.animation = 0; ch.property = Property::Rotation;
        ch.times = times; ch.values = values;
        std::vector<AnimationChannel> chans{ch};
        std::vector<Animation> anims{Animation{"general"}};

        auto out = analyze(chans, anims, rig.skeleton());
        ASSERT_EQ(out.size(), 1u);
        ASSERT_EQ(out[0].dof_count, 3);
        assert_no_invalid_stage(out);
    }
}
