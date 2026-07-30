#include <gtest/gtest.h>
#include "chacha_stage.h"

TEST(Smoke, StageTypeNamesAreStable)
{
    EXPECT_STREQ(ChaCha::stage_type_name(ChaCha::StageType::xRotate), "xRotate");
    EXPECT_STREQ(ChaCha::stage_type_name(ChaCha::StageType::zScale), "zScale");
}
