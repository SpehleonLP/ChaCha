#include <gtest/gtest.h>
#include "chacha_internal.h"

using namespace ChaCha;
using namespace ChaCha::detail;

// StageType::Invalid marks an unused Candidate slot (see chacha_internal.h,
// Task 8) and must never reach an emitted Stage. chacha_analyzer.cpp's
// stage-emission loop is bounded by `dof` and additionally skips any slot
// whose type is Invalid before ever constructing a RawStage for it -- but
// filter_stages is the last-resort backstop that would still have to catch
// a stray Invalid-typed RawStage if some future bug got one past those
// upstream checks (see chacha_filter.cpp's threshold_for_type comment).
// This test pins that backstop directly: even a RawStage with a huge
// observed range -- large enough to clear every real threshold in
// Options -- must still be dropped, because threshold_for_type(Invalid)
// returns +infinity rather than 0 or a finite value.
TEST(Filter, InvalidTypedRawStageIsDroppedRegardlessOfRange)
{
    RawStage r;
    r.type             = StageType::Invalid;
    r.min_value        = -1000.0f;
    r.max_value        = 1000.0f;   // range = 2000, far past any real threshold
    r.initial_value    = 0.0f;
    r.max_velocity     = 0.0f;
    r.max_acceleration = 0.0f;

    std::vector<RawStage> raws{r};
    auto stages = filter_stages(raws, Options{});
    EXPECT_TRUE(stages.empty())
        << "an Invalid-typed RawStage must never survive filter_stages, "
           "no matter how large its observed range is";
}

// A real stage sitting right alongside a stray Invalid one must still
// survive: the backstop must drop only the Invalid entry, not misbehave
// and swallow (or fail to swallow) its neighbours.
TEST(Filter, InvalidTypedRawStageIsDroppedWithoutAffectingRealStages)
{
    RawStage real;
    real.type      = StageType::xRotate;
    real.min_value = 0.0f;
    real.max_value = 1.0f;   // well past rotation_threshold_rad's default 0.01

    RawStage invalid;
    invalid.type      = StageType::Invalid;
    invalid.min_value = -1000.0f;
    invalid.max_value = 1000.0f;

    std::vector<RawStage> raws{real, invalid};
    auto stages = filter_stages(raws, Options{});
    ASSERT_EQ(stages.size(), 1u);
    EXPECT_EQ(stages[0].type, StageType::xRotate);
}
