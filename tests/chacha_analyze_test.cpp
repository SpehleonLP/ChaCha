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
