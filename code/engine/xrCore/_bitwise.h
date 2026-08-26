#pragma once
#ifndef _BITWISE_
#define _BITWISE_

#include <cmath>
#include <intrin.h>
#include <immintrin.h>

// float values defines
constexpr u32 fdSGN  = 0x080000000;  // mask for sign bit
constexpr u32 fdMABS = 0x07FFFFFFF;  // mask for absolute value (~sgn)
constexpr u32 fdMANT = 0x0007FFFFF;  // mask for mantissa
constexpr u32 fdEXPO = 0x07F800000;  // mask for exponent
constexpr u32 fdONE  = 0x03F800000;  // 1.0f
constexpr u32 fdHALF = 0x03F000000;  // 0.5f
constexpr u32 fdTWO  = 0x040000000;  // 2.0
constexpr u32 fdOOB  = 0x000000000;  // "out of bounds" value
constexpr u32 fdNAN  = 0x07fffffff;  // "Not a number" value
constexpr u32 fdMAX  = 0x07F7FFFFF;  // FLT_MAX
constexpr u32 fdRLE10 = 0x03ede5bdb; // 1/ln10

[[nodiscard]] inline bool negative(const float f) noexcept { return std::signbit(f); }
[[nodiscard]] inline bool positive(const float f) noexcept { return !std::signbit(f); }
inline void set_negative(float& f) noexcept { f = -std::abs(f); }
inline void set_positive(float& f) noexcept { f = std::abs(f); }

// Lowest Bit Mask
[[nodiscard]] constexpr int btwLowestBitMask(int v) noexcept { return (v & -v); }
[[nodiscard]] constexpr u32 btwLowestBitMask(u32 x) noexcept { return x & ~(x - 1); }

// Pow2 checks
[[nodiscard]] constexpr bool btwIsPow2(int v) noexcept { return (btwLowestBitMask(v) == v); }
[[nodiscard]] constexpr bool btwIsPow2(u32 v) noexcept { return (btwLowestBitMask(v) == v); }

[[nodiscard]] inline int btwPow2_Ceil(int v) noexcept {
    int i = btwLowestBitMask(v);
    while (i < v) i <<= 1;
    return i;
}
[[nodiscard]] inline u32 btwPow2_Ceil(u32 v) noexcept {
    u32 i = btwLowestBitMask(v);
    while (i < v) i <<= 1;
    return i;
}

[[nodiscard]] inline u8 btwCount1(u8 v) noexcept {
    return static_cast<u8>(__popcnt16(static_cast<u16>(v)));
}

[[nodiscard]] inline u32 btwCount1(u32 v) noexcept {
    return __popcnt(v);
}

[[nodiscard]] inline u64 btwCount1(u64 v) noexcept {
#ifdef _M_AMD64
    return __popcnt64(v);
#else
    return __popcnt(static_cast<u32>(v)) + __popcnt(static_cast<u32>(v >> 32));
#endif
}

[[nodiscard]] inline int iFloor(float x) noexcept {
    return _mm_cvt_ss2si(_mm_round_ss(_mm_set_ss(x), _mm_set_ss(x), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
}

[[nodiscard]] inline int iCeil(float x) noexcept {
    return _mm_cvt_ss2si(_mm_round_ss(_mm_set_ss(x), _mm_set_ss(x), _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
}

[[nodiscard]] inline bool fis_gremlin(const float& f) noexcept {
    u8 value = static_cast<u8>(((*reinterpret_cast<const u32*>(&f) & 0x7f800000) >> 23) - 0x20);
    return value > 0xc0;
}

[[nodiscard]] inline bool fis_denormal(const float& f) noexcept {
    return !(*reinterpret_cast<const u32*>(&f) & 0x7f800000);
}

[[nodiscard]] inline float apx_InvSqrt(const float& n) noexcept {
    u32 tmp = (0xBE800000 - *reinterpret_cast<const u32*>(&n)) >> 1;
    float y = *reinterpret_cast<float*>(&tmp);
    return y * (1.47f - 0.47f * n * y * y);
}

[[nodiscard]] inline float apx_asin(const float x) noexcept {
    constexpr float c1 = 0.892399f;
    constexpr float c3 = 1.693204f;
    constexpr float c5 = -3.853735f;
    constexpr float c7 = 2.838933f;

    const float x2 = x * x;
    return x * (c1 + x2 * (c3 + x2 * (c5 + x2 * c7)));
}

[[nodiscard]] inline float apx_acos(const float x) noexcept { 
    constexpr float PI_DIV_2 = 1.57079632679f;
    return PI_DIV_2 - apx_asin(x); 
}

#endif // _BITWISE_