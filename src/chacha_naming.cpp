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
    // Invalid marks an unused Candidate slot (see chacha_internal.h) and must
    // never reach emitted AGI output. Assert loudly in debug builds rather
    // than silently formatting a plausible-looking "invalid"/"invalid2" name
    // that would slip into a schema-conformant-looking but bogus stage.
    assert(type != StageType::Invalid && "stage_name_for: StageType::Invalid must never be emitted");

    std::string name = stage_type_name(type);
    if (occurrence > 0) name += std::to_string(occurrence + 1);
    return name;
}

} // namespace ChaCha
