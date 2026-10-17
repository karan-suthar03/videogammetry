// Shared by every ORB shader: buffer bindings, the parameter table layout
// (written by gpu_orb.cpp, keep in sync with kParam* there) and 8-bit image
// access. Images are 8-bit, packed four pixels per uint, rows padded to whole
// words (stride in words).

layout(std430, set = 0, binding = 0) buffer Images { uint img[]; };
layout(std430, set = 0, binding = 1) buffer Temp { uint tmp[]; };           // blur: one pixel per uint
layout(std430, set = 0, binding = 2) buffer Scores { float scoreMap[]; };   // per level, 0 = no candidate
layout(std430, set = 0, binding = 3) buffer Cells { uint cellHasMain[]; };  // per level grid cell
layout(std430, set = 0, binding = 4) buffer Candidates { uint cand[]; };    // 3 words each: x | y << 16, level, score
layout(std430, set = 0, binding = 5) buffer Results { uint res[]; };        // counters, then kRecordWords words per local maximum
layout(std430, set = 0, binding = 6) readonly buffer Params { int params[]; };
layout(std430, set = 0, binding = 7) readonly buffer Frame { uint frame[]; };

// Results header.
const uint kResCandidates = 0u;  // candidates found (may exceed capacity)
const uint kResMaxima = 1u;      // local maxima found (may exceed capacity)
const uint kResHeader = 16u;     // words before the first record
const uint kRecordWords = 20u;   // x | y << 16, level, score, m01, m10, unused, descriptor[8], unused[2],
                                 // Harris response left, right, up, down of the corner (sub-pixel)

// Params header.
const int kParamCandidateCapacity = 0;
const int kParamMaximaCapacity = 1;
const int kParamPattern = 2;  // offset of 256 tests, one int each: x1, y1, x2, y2 as signed bytes
const int kParamLevels = 3;   // offset of the level table
const int kLevelWords = 16;
// Level table entry.
const int kLvWidth = 0, kLvHeight = 1, kLvImage = 2, kLvStride = 3, kLvBlurred = 4, kLvScore = 5, kLvCells = 6,
          kLvX0 = 7, kLvY0 = 8, kLvX1 = 9, kLvY1 = 10, kLvCell = 11, kLvCols = 12, kLvThreshold = 13,
          kLvMinThreshold = 14;

int level(int l, int field) { return params[params[kParamLevels] + l * kLevelWords + field]; }

int pixelAt(int offset, int strideWords, int x, int y) {
    const uint w = img[uint(offset) + uint(y) * uint(strideWords) + (uint(x) >> 2)];
    return int((w >> ((uint(x) & 3u) * 8u)) & 0xFFu);
}

int frameByte(uint index) { return int((frame[index >> 2] >> ((index & 3u) * 8u)) & 0xFFu); }

// vg::harrisResponse. The sums are exact integers in float; `precise` keeps
// the final expression from being fused, so it rounds exactly like the CPU.
float harrisResponse(int off, int stride, int x, int y) {
    precise float a = 0.0, b = 0.0, c = 0.0;
    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            precise float ix = float(pixelAt(off, stride, x + dx + 1, y + dy)) - float(pixelAt(off, stride, x + dx - 1, y + dy));
            precise float iy = float(pixelAt(off, stride, x + dx, y + dy + 1)) - float(pixelAt(off, stride, x + dx, y + dy - 1));
            a += ix * ix;
            b += iy * iy;
            c += ix * iy;
        }
    }
    const float k = 0.04;
    precise float r = a * b - c * c - k * (a + b) * (a + b);
    return r;
}
