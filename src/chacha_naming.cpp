#include "chacha_naming.h"
#include <algorithm>
#include <cassert>
#include <cctype>

namespace ChaCha {

namespace {

// std::isspace is locale-dependent; if the ambient global locale (set via
// std::setlocale elsewhere in the process) treats high bytes as whitespace
// (e.g. Latin-1's 0xA0 non-breaking space), a naive std::isspace check would
// misclassify the continuation bytes of a multi-byte UTF-8 sequence -- every
// continuation byte has its high bit set (0x80-0xBF) -- and corrupt node
// names that are valid UTF-8 but not ASCII. Restricting the whitespace test
// to the ASCII range (< 0x80) keeps the classification locale-independent
// and guarantees UTF-8 encoded bytes above ASCII pass through untouched.
bool is_ascii_space(unsigned char c)
{
    return c < 0x80 && std::isspace(c);
}

} // namespace

std::string sanitize_articulation_name(std::string_view raw, std::vector<std::string>& taken)
{
    std::string base;
    base.reserve(raw.size());
    for (char c : raw) {
        if (is_ascii_space(static_cast<unsigned char>(c))) {
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
    // This documents a caller contract -- stage_name_for is only ever meant
    // to be called on a Stage that has already survived the pipeline, and
    // Invalid never should have -- it is NOT the mechanism that keeps
    // StageType::Invalid out of emitted AGI output. This project's default
    // configure (RelWithDebInfo, see CMakeLists.txt) sets -DNDEBUG, which
    // compiles this assert out entirely, so in the build that actually
    // ships this line does nothing at all. The real guarantee lives in the
    // production pipeline: chacha_analyzer.cpp's stage-emission loop is
    // bounded by Candidate::dof, and chacha_filter.cpp's
    // threshold_for_type(Invalid) returns +infinity as a last-resort
    // backstop even if a stray Invalid-typed RawStage got past that loop.
    // See Analyze.NeverEmitsInvalidStageType and
    // Filter.InvalidTypedRawStageIsDroppedRegardlessOfRange, which pin
    // those two facts with tests rather than relying on this assert.
    assert(type != StageType::Invalid && "stage_name_for: StageType::Invalid must never be emitted");

    std::string name = stage_type_name(type);
    if (occurrence > 0) name += std::to_string(occurrence + 1);
    return name;
}

} // namespace ChaCha
