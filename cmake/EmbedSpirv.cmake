# Called by compile_shaders() to convert a .spv file into a C++ header.
# Variables: SPV_FILE, HDR_FILE, VAR_NAME
file(READ "${SPV_FILE}" SPV_BYTES HEX)

# Split hex string into pairs, build comma-separated 0xNN list
string(REGEX MATCHALL ".." BYTE_LIST "${SPV_BYTES}")
list(JOIN BYTE_LIST ", 0x" BYTE_CSV)
set(BYTE_CSV "0x${BYTE_CSV}")

# Word count for the span size
string(LENGTH "${SPV_BYTES}" HEX_LEN)
math(EXPR BYTE_COUNT "${HEX_LEN} / 2")
math(EXPR WORD_COUNT "${BYTE_COUNT} / 4")

file(WRITE "${HDR_FILE}"
"#pragma once
#include <cstdint>
#include <span>
// Auto-generated -- do not edit
namespace spv {
alignas(4) inline constexpr uint8_t ${VAR_NAME}_bytes[] = { ${BYTE_CSV} };
inline const std::span<const uint32_t> ${VAR_NAME} {
    reinterpret_cast<const uint32_t*>(${VAR_NAME}_bytes), ${WORD_COUNT}
};
} // namespace spv
")
