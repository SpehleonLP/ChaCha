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

TEST(Api, AnalyzeToleratesNullDiagnostics)
{
    Skeleton sk;
    auto out = analyze({}, {}, sk, Options{}, {}, nullptr);
    EXPECT_TRUE(out.empty());
}

TEST(Api, StageTypeNameNeverReturnsUnknownForInvalid)
{
    // Invalid must be named distinctly and deliberately, not fall through
    // to the generic "unknown" bucket used for genuinely unrecognised values.
    EXPECT_STREQ(stage_type_name(StageType::Invalid), "invalid");
}

TEST(Api, SkeletonRestScalesDefaultsEmpty)
{
    Skeleton sk;
    EXPECT_TRUE(sk.rest_scales.empty());
}

TEST(Api, ArticulationHasDofCountAndFitResidual)
{
    Articulation a;
    a.dof_count = 2;
    a.fit_residual_rad = 0.005f;
    EXPECT_EQ(a.dof_count, 2);
    EXPECT_FLOAT_EQ(a.fit_residual_rad, 0.005f);
}

TEST(Api, DiagnosticDefaultKindIsEmptyChannel)
{
    Diagnostic d;
    EXPECT_EQ(d.kind, Diagnostic::EmptyChannel);
}

TEST(Api, AnimationChannelDefaultsToAnimationZero)
{
    AnimationChannel ch;
    EXPECT_EQ(ch.animation, 0);
}
