#include <gtest/gtest.h>
#include "chacha_naming.h"
#include <set>

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

// A model can legitimately already contain a node literally named "Bone_2"
// alongside two nodes named "Bone". The suffixing scheme must skip past any
// already-taken suffix rather than emitting a duplicate.
TEST(Naming, SuffixSchemeSkipsPreexistingCollidingNames)
{
    std::vector<std::string> taken;
    EXPECT_EQ(sanitize_articulation_name("Bone_2", taken), "Bone_2");
    EXPECT_EQ(sanitize_articulation_name("Bone", taken), "Bone");
    EXPECT_EQ(sanitize_articulation_name("Bone", taken), "Bone_3");
    // All three names actually assigned must be distinct.
    EXPECT_EQ(taken.size(), 3u);
    EXPECT_EQ(std::set<std::string>(taken.begin(), taken.end()).size(), 3u);
}

// A name that is nothing but whitespace must not merely fail to be empty --
// it must reduce to the same non-empty fallback as a truly empty name, and
// that fallback must still satisfy AGI's ^[^\s]+$ (no embedded whitespace).
TEST(Naming, WhitespaceOnlyNameFallsBackToPlaceholder)
{
    std::vector<std::string> taken;
    std::string result = sanitize_articulation_name("   \t\n  ", taken);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.find_first_of(" \t\n\v\f\r"), std::string::npos);
}

// Non-ASCII UTF-8 bytes (each byte here has the high bit set) must survive
// sanitisation untouched: they are not ASCII whitespace and must not be
// corrupted or collapsed by a locale-sensitive isspace check.
TEST(Naming, NonAsciiUtf8BytesArePreserved)
{
    std::vector<std::string> taken;
    // "Épaule" (French for "shoulder"): 0xC3 0x89 is the UTF-8 encoding of
    // 'E' with acute accent.
    std::string raw = "\xC3\x89paule";
    EXPECT_EQ(sanitize_articulation_name(raw, taken), raw);
}
