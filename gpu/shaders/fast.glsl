// FAST-9 segment test exactly as vg::isFastCorner (orb.cpp): 9 contiguous
// pixels of the radius-3 circle all brighter than p + t or all darker than p - t.

const ivec2 kCircle[16] = ivec2[16](ivec2(0, -3), ivec2(1, -3), ivec2(2, -2), ivec2(3, -1), ivec2(3, 0),
                                    ivec2(3, 1), ivec2(2, 2), ivec2(1, 3), ivec2(0, 3), ivec2(-1, 3),
                                    ivec2(-2, 2), ivec2(-3, 1), ivec2(-3, 0), ivec2(-3, -1), ivec2(-2, -2),
                                    ivec2(-1, -3));

bool hasArc(int v[16], bool bright, int hi, int lo) {
    int run = 0;
    for (int i = 0; i < 24; ++i) {
        const int q = v[i & 15];
        if (bright ? q > hi : q < lo) {
            if (++run >= 9) return true;
        } else {
            run = 0;
        }
    }
    return false;
}

bool isFastCorner(int off, int stride, int x, int y, int t) {
    const int p = pixelAt(off, stride, x, y);
    const int hi = p + t, lo = p - t;
    int v[16];
    for (int i = 0; i < 16; ++i) v[i] = pixelAt(off, stride, x + kCircle[i].x, y + kCircle[i].y);
    const int brightCompass = int(v[0] > hi) + int(v[4] > hi) + int(v[8] > hi) + int(v[12] > hi);
    const int darkCompass = int(v[0] < lo) + int(v[4] < lo) + int(v[8] < lo) + int(v[12] < lo);
    if (brightCompass < 2 && darkCompass < 2) return false;
    return (brightCompass >= 2 && hasArc(v, true, hi, lo)) || (darkCompass >= 2 && hasArc(v, false, hi, lo));
}
