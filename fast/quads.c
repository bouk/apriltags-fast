/* Quad fitting: sort each cluster's points by angle, find the four
   corners as maxima of the line-fit error, and fit four lines. The
   arithmetic mirrors the C library's AVX2 build bit for bit. */
#include "common.h"

/* A line fit between two maxima (par[0..2] are not set). */
typedef struct {
    double par[4];
    double err, mse;
} PairFit;

/* The scratch arrays of one cluster of n points, bumped from cx: the
   arrays of the pair fits are bumped later, when nm is known. */
typedef struct {
    at_ctx *cx;
    uint64_t *keys, *tmp, *bh, *rk;
    double *mx, *my, *mxx, *mxy, *myy, *mw, *errs, *yfilt, *wbuf, *maxima_errs;
    int32_t *cidx;
    int64_t *maxima;
} QuadScratch;

static void quad_scratch_alloc(at_ctx *cx, QuadScratch *sc, int64_t n) {
    sc->cx = cx;
    sc->keys = STK_ALLOC(cx, uint64_t, n);
    sc->tmp = STK_ALLOC(cx, uint64_t, n);
    sc->bh = STK_ALLOC(cx, uint64_t, n);
    sc->rk = STK_ALLOC(cx, uint64_t, n);
    sc->mx = STK_ALLOC(cx, double, n);
    sc->my = STK_ALLOC(cx, double, n);
    sc->mxx = STK_ALLOC(cx, double, n);
    sc->mxy = STK_ALLOC(cx, double, n);
    sc->myy = STK_ALLOC(cx, double, n);
    sc->mw = STK_ALLOC(cx, double, n);
    sc->errs = STK_ALLOC(cx, double, n);
    sc->yfilt = STK_ALLOC(cx, double, n);
    sc->wbuf = STK_ALLOC(cx, double, n);
    sc->maxima_errs = STK_ALLOC(cx, double, n);
    sc->cidx = STK_ALLOC(cx, int32_t, n);
    sc->maxima = STK_ALLOC(cx, int64_t, n);
}

/* ---------------------------------------------------------------------
   The angle sort. Keys are (order-preserving slope bits << 32) | ~index.
   The leaf networks compare the high words only and never swap ties;
   the merges compare full keys and take the right run on ties. This
   exact structure decides the order of equal slopes. */

AT_INLINE void key_swap(uint64_t *k, int a, int b) {
    if ((k[a] >> 32) > (k[b] >> 32)) {
        uint64_t t = k[a];
        k[a] = k[b];
        k[b] = t;
    }
}

static void key_network_sort(uint64_t *k, int64_t sz) {
    if (sz <= 1) return;
    if (sz == 2) {
        key_swap(k, 0, 1);
        return;
    }
    if (sz == 3) {
        key_swap(k, 0, 1);
        key_swap(k, 1, 2);
        key_swap(k, 0, 1);
        return;
    }
    if (sz == 4) {
        key_swap(k, 0, 1);
        key_swap(k, 2, 3);
        key_swap(k, 0, 2);
        key_swap(k, 1, 3);
        key_swap(k, 1, 2);
        return;
    }
    key_swap(k, 0, 1);
    key_swap(k, 3, 4);
    key_swap(k, 1, 2);
    key_swap(k, 0, 1);
    key_swap(k, 0, 3);
    key_swap(k, 2, 4);
    key_swap(k, 1, 2);
    key_swap(k, 2, 3);
    key_swap(k, 1, 2);
}

/* The smaller of two keys, without a branch: merge comparisons are coin
   flips, so a select beats a mispredicted branch. */
AT_INLINE uint64_t key_select(bool take_a, uint64_t av, uint64_t bv) {
    uint64_t m = (uint64_t)0 - (uint64_t)take_a;
    return (av & m) | (bv & ~m);
}

static void key_merge(const uint64_t *a, int64_t na, const uint64_t *b, int64_t nb, uint64_t *out) {
    int64_t ap = 0, bp = 0, o = 0;
    while (ap < na && bp < nb) {
        uint64_t av = a[ap], bv = b[bp];
        bool ta = av < bv;
        out[o++] = key_select(ta, av, bv);
        int64_t t = ta;
        ap += t;
        bp += 1 - t;
    }
    if (ap < na)
        memcpy(out + ap + bp, a + ap, (size_t)(na - ap) * 8);
    else if (bp < nb)
        memcpy(out + ap + bp, b + bp, (size_t)(nb - bp) * 8);
}

static void finish_merge4(const uint64_t *l1, int64_t n1, const uint64_t *l2, int64_t n2, const uint64_t *r1,
                          int64_t n3, const uint64_t *r2, int64_t n4, uint64_t *out, int64_t i1, int64_t i2,
                          int64_t i3, int64_t i4, int64_t o) {
    while ((i1 < n1 || i2 < n2) && (i3 < n3 || i4 < n4)) {
        bool lt1 = i1 < n1 && (i2 >= n2 || l1[i1] < l2[i2]);
        bool rt1 = i3 < n3 && (i4 >= n4 || r1[i3] < r2[i4]);
        uint64_t lv = lt1 ? l1[i1] : l2[i2];
        uint64_t rv = rt1 ? r1[i3] : r2[i4];
        if (lv < rv) {
            out[o++] = lv;
            if (lt1)
                i1++;
            else
                i2++;
        } else {
            out[o++] = rv;
            if (rt1)
                i3++;
            else
                i4++;
        }
    }
    if (i1 < n1 || i2 < n2)
        key_merge(l1 + i1, n1 - i1, l2 + i2, n2 - i2, out + o);
    else
        key_merge(r1 + i3, n3 - i3, r2 + i4, n4 - i4, out + o);
}

/* What key_merge(key_merge(l1, l2), key_merge(r1, r2)) gives, in one
   pass. */
static void key_merge4(const uint64_t *l1, int64_t n1, const uint64_t *l2, int64_t n2, const uint64_t *r1, int64_t n3,
                       const uint64_t *r2, int64_t n4, uint64_t *out) {
    int64_t i1 = 0, i2 = 0, i3 = 0, i4 = 0, o = 0;
    while (i1 < n1 && i2 < n2 && i3 < n3 && i4 < n4) {
        uint64_t v1 = l1[i1], v2 = l2[i2], v3 = r1[i3], v4 = r2[i4];
        bool lt1 = v1 < v2;
        bool rt1 = v3 < v4;
        uint64_t lv = key_select(lt1, v1, v2);
        uint64_t rv = key_select(rt1, v3, v4);
        bool tl = lv < rv;
        out[o++] = key_select(tl, lv, rv);
        i1 += tl && lt1;
        i2 += tl && !lt1;
        i3 += !tl && rt1;
        i4 += !tl && !rt1;
    }
    finish_merge4(l1, n1, l2, n2, r1, n3, r2, n4, out, i1, i2, i3, i4, o);
}

/* Sort a into itself (to_b false, b is scratch) or into b (to_b true, a
   is clobbered). */
static void keysort(uint64_t *a, uint64_t *b, int64_t sz, bool to_b) {
    if (sz <= 5) {
        key_network_sort(a, sz);
        if (to_b) memcpy(b, a, (size_t)sz * 8);
        return;
    }
    int64_t asz = sz / 2;
    if (to_b) {
        keysort(a, b, asz, false);
        keysort(a + asz, b + asz, sz - asz, false);
        key_merge(a, asz, a + asz, sz - asz, b);
    } else {
        keysort(a, b, asz, true);
        keysort(a + asz, b + asz, sz - asz, true);
        key_merge(b, asz, b + asz, sz - asz, a);
    }
}

/* keysort with two merge levels fused into one pass. The splits and
   therefore the result are those of keysort. */
static void keysort4(uint64_t *a, uint64_t *b, int64_t sz, bool to_b) {
    if (sz <= 5) {
        key_network_sort(a, sz);
        if (to_b) memcpy(b, a, (size_t)sz * 8);
        return;
    }
    int64_t asz = sz / 2;
    int64_t bsz = sz - asz;
    if (asz <= 5) {
        keysort(a, b, sz, to_b);
        return;
    }
    int64_t a1 = asz / 2;
    int64_t b1 = asz + bsz / 2;
    if (to_b) {
        keysort4(a, b, a1, false);
        keysort4(a + a1, b + a1, asz - a1, false);
        keysort4(a + asz, b + asz, b1 - asz, false);
        keysort4(a + b1, b + b1, sz - b1, false);
        key_merge4(a, a1, a + a1, asz - a1, a + asz, b1 - asz, a + b1, sz - b1, b);
    } else {
        keysort4(a, b, a1, true);
        keysort4(a + a1, b + a1, asz - a1, true);
        keysort4(a + asz, b + asz, b1 - asz, true);
        keysort4(a + b1, b + b1, sz - b1, true);
        key_merge4(b, a1, b + a1, asz - a1, b + asz, b1 - asz, b + b1, sz - b1, a);
    }
}

/* The leaf of keysort4 over positions 0..n that holds position p, as
   (start << 8) | length. keysort4 splits in four while half the range
   has more than 5 positions, then keysort splits in two down to 5 or
   fewer; this follows the same splits down to p, without branches on p
   (the CPU cannot predict which part holds p). */
static uint32_t leaf_of(int64_t n, int64_t p) {
    int64_t lo = 0, sz = n;
    while (sz / 2 > 5) {
        /* the four parts of keysort4 start at 0, a1, asz and b1 */
        int64_t asz = sz / 2;
        int64_t a1 = asz / 2;
        int64_t b1 = asz + (sz - asz) / 2;
        int64_t r = p - lo;
        int64_t c1 = r >= a1, c2 = r >= asz, c3 = r >= b1;
        int64_t s = c1 * a1 + c2 * (asz - a1) + c3 * (b1 - asz);
        int64_t e = a1 + c1 * (asz - a1) + c2 * (b1 - asz) + c3 * (sz - b1);
        lo += s;
        sz = e - s;
    }
    /* then halves */
    while (sz > 5) {
        int64_t asz = sz / 2;
        int64_t c = p - lo >= asz;
        lo += c * asz;
        sz = asz + c * (sz - 2 * asz);
    }
    return (uint32_t)((lo << 8) | sz);
}

AT_INLINE int64_t key_index(uint64_t k) { return (uint32_t)~(uint32_t)k; }

/* Order one run of equal high parts. r holds the run's keys in full-key
   order (descending index); orig the n input keys. ent and bh are
   scratch of r's length. The keys go to ent in leaf order from the last
   leaf to the first, which is close to the result, so the insertion sort
   by block start is short. */
static void order_run(const uint64_t *orig, int64_t n, uint64_t *r, int64_t nr, uint64_t *ent, uint64_t *bh) {
    uint64_t hi = r[0] & 0xffffffff00000000u;
    int64_t m = 0, k = 0;
    while (k < nr) {
        /* a leaf spans at most 5 positions: a key with no next key of
           the run within 4 positions is alone in its leaf */
        if (k + 1 == nr || key_index(r[k]) - key_index(r[k + 1]) >= 5) {
            ent[m] = r[k];
            bh[m] = r[k];
            m++;
            k++;
            continue;
        }
        int64_t lv = leaf_of(n, key_index(r[k]));
        int64_t lo = lv >> 8;
        int64_t end = lo + (lv & 255);
        /* the run's keys in this leaf follow in r */
        int64_t k1 = k + 1;
        while (k1 < nr && key_index(r[k1]) >= lo) k1++;
        if (k1 == k + 1) {
            ent[m] = r[k];
            bh[m] = r[k];
            m++;
        } else {
            /* the leaf's order after its network sort, and the blocks */
            uint64_t lf[5] = {0};
            int64_t ln = end - lo;
            for (int64_t t = 0; t < ln; t++) lf[t] = orig[lo + t];
            key_network_sort(lf, ln);
            uint64_t top = 0;
            bool first = true;
            for (int64_t t = 0; t < ln; t++) {
                uint64_t key = lf[t];
                if ((key & 0xffffffff00000000u) != hi) continue;
                if (first || key > top) top = key;
                first = false;
                ent[m] = key;
                bh[m] = top;
                m++;
            }
        }
        k = k1;
    }
    /* stable insertion sort by block start */
    for (int64_t t = 1; t < m; t++) {
        uint64_t e = ent[t], b = bh[t];
        int64_t j = t;
        while (j > 0 && bh[j - 1] > b) {
            ent[j] = ent[j - 1];
            bh[j] = bh[j - 1];
            j--;
        }
        ent[j] = e;
        bh[j] = b;
    }
    memcpy(r, ent, (size_t)m * 8);
}

/* Put the runs of res (in full-key order) that have keys in one leaf in
   leaf-block order. orig is the input; ent and bh are scratch. */
static void fix_runs(uint64_t *res, int64_t n, const uint64_t *orig, uint64_t *ent, uint64_t *bh) {
    int64_t i = 1;
    while (i < n) {
        /* keys of one run less than 5 positions apart differ by less
           than 5; other keys rarely do, and are checked here */
        if (res[i] - res[i - 1] >= 5) {
            i++;
            continue;
        }
        uint64_t x = res[i - 1];
        if ((x >> 32) != (res[i] >> 32)) {
            i++;
            continue;
        }
        int64_t s = i - 1;
        while (s > 0 && (res[s - 1] >> 32) == (x >> 32)) s--;
        int64_t e = i + 1;
        while (e < n && (res[e] >> 32) == (x >> 32)) e++;
        order_run(orig, n, res + s, e - s, ent, bh);
        i = e + 1;
    }
}

static void radix_histograms(const uint64_t *a, int64_t n, uint32_t *h) {
    for (int64_t i = 0; i < n; i++) {
        uint32_t hk = (uint32_t)(a[i] >> 32);
        h[hk & 255]++;
        h[256 + ((hk >> 8) & 255)]++;
        h[512 + ((hk >> 16) & 255)]++;
        h[768 + (hk >> 24)]++;
    }
}

/* radix_pass, reading the source from its end. */
static void radix_pass_back(const uint64_t *src, int64_t n, uint64_t *dst, const uint32_t *cnt, int shift) {
    uint32_t pos[256];
    uint32_t at = 0;
    for (int i = 0; i < 256; i++) {
        pos[i] = at;
        at += cnt[i];
    }
    for (int64_t i = n - 1; i >= 0; i--) {
        uint64_t k = src[i];
        int64_t d = (int64_t)((k >> shift) & 255);
        dst[pos[d]++] = k;
    }
}

/* One stable counting pass on the 8-bit digit at shift. */
static void radix_pass(const uint64_t *src, int64_t n, uint64_t *dst, const uint32_t *cnt, int shift) {
    uint32_t pos[256];
    uint32_t at = 0;
    for (int i = 0; i < 256; i++) {
        pos[i] = at;
        at += cnt[i];
    }
    for (int64_t i = 0; i < n; i++) {
        uint64_t k = src[i];
        int64_t d = (int64_t)((k >> shift) & 255);
        dst[pos[d]++] = k;
    }
}

/* Sort the keys exactly as keysort4 does, without its merges.
   keysort4's leaves (at most 5 keys) are sorted by the high 32 bits
   only, so keys with equal high parts can stay out of order in a leaf.
   Its merges take the smallest head by the full key. Such a merge keeps
   each "block" of a sequence together (a key larger than all keys
   before it, and the smaller keys after it) and orders the blocks by
   their first key; merges of merges do the same. So the result is the
   leaf keys ordered by (first key of their block, position in the leaf).
   A block lies inside a run of equal high parts, and a leaf spans fewer
   than 5 positions. So the result is the full-key order, except in runs
   of equal high parts with keys less than 5 positions apart: those runs
   are put in leaf-block order.
   tmp, bh and rk are scratch of the same length as a. The result is in
   rk (return 1) or tmp (return 2); a keeps the input. */
static int keysort_blocks(uint64_t *a, int64_t n, uint64_t *tmp, uint64_t *bh, uint64_t *rk) {
    uint32_t h[1024];
    memset(h, 0, sizeof h);
    radix_histograms(a, n, h);
    uint32_t hk0 = (uint32_t)(a[0] >> 32);
    int at = 0;
    for (int d = 0; d < 4; d++) {
        uint32_t dig = (hk0 >> (8 * d)) & 255;
        if (h[256 * d + dig] == n) continue;
        const uint32_t *cnt = h + 256 * d;
        int shift = 32 + 8 * d;
        if (at == 0) {
            /* backwards: equal high parts end in ascending ~index order */
            radix_pass_back(a, n, rk, cnt, shift);
            at = 1;
        } else if (at == 1) {
            radix_pass(rk, n, tmp, cnt, shift);
            at = 2;
        } else {
            radix_pass(tmp, n, rk, cnt, shift);
            at = 1;
        }
    }
    if (at == 0) {
        /* all high parts are equal: the full-key order is the reverse */
        for (int64_t i = 0; i < n; i++) rk[i] = a[n - 1 - i];
        at = 1;
    }
    if (at == 1)
        fix_runs(rk, n, a, tmp, bh);
    else
        fix_runs(tmp, n, a, rk, bh);
    return at;
}

AT_INLINE uint64_t slope_sort_key(float slope) {
    uint32_t bits = bits_of_f32(slope);
    uint32_t sgn = (uint32_t)((int32_t)bits >> 31);
    return bits ^ (sgn | 0x80000000u);
}

/* The sort key of a point: its pseudo-angle around the center, then the
   complemented index. Every step is a select or one rounded operation,
   so the vector versions give the same bits. */
AT_INLINE uint64_t point_key(float dx, float dy, int64_t idx) {
    /* quadrants[dy > 0][dx > 0] = {{-65536, 0}, {131072, 65536}} */
    uint32_t dxpos = mask32(dx > 0.0f);
    float qlo = fsel(dxpos, 0.0f, -65536.0f);
    float qhi = fsel(dxpos, 65536.0f, 131072.0f);
    float quadrant = fsel(mask32(dy > 0.0f), qhi, qlo);
    /* if dy < 0, negate both (a sign flip, as -x); then if dx < 0,
       rotate by 90 degrees */
    uint32_t dyneg = mask32(dy < 0.0f) & 0x80000000u;
    float fx = f32_of_bits(bits_of_f32(dx) ^ dyneg);
    float fy = f32_of_bits(bits_of_f32(dy) ^ dyneg);
    uint32_t dxneg = mask32(fx < 0.0f);
    float ndx = fsel(dxneg, fy, fx);
    float ndy = fsel(dxneg, -fx, fy);
    return (slope_sort_key(quadrant + ndy / ndx) << 32) | (uint32_t)~(uint32_t)idx;
}

/* The keys of all points, and the dot product of the offsets from the
   center with the gradients: 8 lanes for the multiple-of-8 part (summed
   as the C library's AVX2 code does), then the rest in order. The lanes
   are an explicit vector, the 8 points of a step split by field with
   shuffles: gcc vectorized the lane loop strip-mined by 8 (as Goose
   writes it) over the outer loop, with thousands of permutes. */
#if defined(__has_builtin)
#if __has_builtin(__builtin_shufflevector) && __has_builtin(__builtin_convertvector) && \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define AT_HAVE_SHUFFLE 1
#endif
#endif
SIMD_FN(float, point_keys, (const Pt *restrict pts, int64_t sz, float cx, float cy, uint64_t *restrict keys),
        (pts, sz, cx, cy, keys)) {
    for (int64_t i = 0; i < sz; i++) {
        Pt p = pts[i];
        keys[i] = point_key(p.x - cx, p.y - cy, i);
    }
    int64_t nv = sz - sz % 8;
#ifdef AT_HAVE_SHUFFLE
    typedef float v8f __attribute__((vector_size(32)));
    typedef uint32_t v8u __attribute__((vector_size(32), aligned(2)));
    typedef int32_t v8i __attribute__((vector_size(32)));
    v8f acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    v8f vcx = {cx, cx, cx, cx, cx, cx, cx, cx}, vcy = {cy, cy, cy, cy, cy, cy, cy, cy};
    for (int64_t i = 0; i < nv; i += 8) {
        /* per point two u32: x | y << 16, then gx | gy << 16 */
        v8u a, b;
        memcpy(&a, pts + i, sizeof a);
        memcpy(&b, pts + i + 4, sizeof b);
        v8u xy = __builtin_shufflevector(a, b, 0, 2, 4, 6, 8, 10, 12, 14);
        v8u g = __builtin_shufflevector(a, b, 1, 3, 5, 7, 9, 11, 13, 15);
        v8f dx = __builtin_convertvector((v8i)(xy & 0xffff), v8f) - vcx;
        v8f dy = __builtin_convertvector((v8i)(xy >> 16), v8f) - vcy;
        v8f gx = __builtin_convertvector((v8i)(g << 16) >> 16, v8f);
        v8f gy = __builtin_convertvector((v8i)g >> 16, v8f);
        acc = acc + (dx * gx + dy * gy);
    }
    float lanes[8];
    memcpy(lanes, &acc, sizeof lanes);
#else
    float lanes[8] = {0};
    for (int64_t i = 0; i < nv; i += 8) {
        for (int64_t t = 0; t < 8; t++) {
            Pt p = pts[i + t];
            lanes[t] = lanes[t] + ((p.x - cx) * p.gx + (p.y - cy) * p.gy);
        }
    }
#endif
    float dot = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6])) + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
    for (int64_t i = nv; i < sz; i++) {
        Pt p = pts[i];
        float dx = p.x - cx;
        float dy = p.y - cy;
        dot += dx * p.gx + dy * p.gy;
    }
    return dot;
}

/* The bounding box of the points: xmin, xmax, ymin, ymax. */
SIMD_VOID(point_bounds, (const Pt *restrict pts, int64_t n, uint16_t *restrict out), (pts, n, out)) {
    uint16_t xmin = 0xffff, xmax = 0, ymin = 0xffff, ymax = 0;
    for (int64_t i = 0; i < n; i++) {
        Pt p = pts[i];
        xmin = p.x < xmin ? p.x : xmin;
        xmax = p.x > xmax ? p.x : xmax;
        ymin = p.y < ymin ? p.y : ymin;
        ymax = p.y > ymax ? p.y : ymax;
    }
    out[0] = xmin;
    out[1] = xmax;
    out[2] = ymin;
    out[3] = ymax;
}

/* ---------------------------------------------------------------------
   Line fitting over the cumulative moments. */

typedef struct {
    const double *mx, *my, *mxx, *mxy, *myy, *w;
} Moments;

/* Fit a line to points [i0, i1] (wrapping). Gives the line parameters
   (Ex, Ey, nx, ny) and the mean squared error. */
static void fit_line(Moments a, int64_t sz, int64_t i0, int64_t i1, double lp[4], double *mse_out) {
    double mx, my, mxx, mxy, myy, w;
    if (i0 < i1) {
        mx = a.mx[i1];
        my = a.my[i1];
        mxx = a.mxx[i1];
        mxy = a.mxy[i1];
        myy = a.myy[i1];
        w = a.w[i1];
        if (i0 > 0) {
            mx -= a.mx[i0 - 1];
            my -= a.my[i0 - 1];
            mxx -= a.mxx[i0 - 1];
            mxy -= a.mxy[i0 - 1];
            myy -= a.myy[i0 - 1];
            w -= a.w[i0 - 1];
        }
    } else {
        mx = a.mx[sz - 1] - a.mx[i0 - 1];
        my = a.my[sz - 1] - a.my[i0 - 1];
        mxx = a.mxx[sz - 1] - a.mxx[i0 - 1];
        mxy = a.mxy[sz - 1] - a.mxy[i0 - 1];
        myy = a.myy[sz - 1] - a.myy[i0 - 1];
        w = a.w[sz - 1] - a.w[i0 - 1];
        mx += a.mx[i1];
        my += a.my[i1];
        mxx += a.mxx[i1];
        mxy += a.mxy[i1];
        myy += a.myy[i1];
        w += a.w[i1];
    }
    double ex = mx / w;
    double ey = my / w;
    double cxx = mxx / w - ex * ex;
    double cxy = mxy / w - ex * ey;
    double cyy = myy / w - ey * ey;
    double root = (double)at_sqrtf((float)((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy));
    double eig_small = 0.5 * (cxx + cyy - root);
    double eig = 0.5 * (cxx + cyy + root);
    double nx1 = cxx - eig;
    double ny1 = cxy;
    double m1 = nx1 * nx1 + ny1 * ny1;
    double nx2 = cxy;
    double ny2 = cyy - eig;
    double m2 = nx2 * nx2 + ny2 * ny2;
    double nx = nx2, ny = ny2, m = m2;
    if (m1 > m2) {
        nx = nx1;
        ny = ny1;
        m = m1;
    }
    double length = (double)at_sqrtf((float)m);
    lp[0] = ex;
    lp[1] = ey;
    lp[2] = 0.0;
    lp[3] = 0.0;
    if (!(fabs(length) < 1e-12)) {
        lp[2] = nx / length;
        lp[3] = ny / length;
    }
    *mse_out = eig_small;
}

/* The window errors of the middle windows lo..hi (no wrap), as
   fit_line's error. */
SIMD_VOID(window_errors,
          (const double *restrict amx, const double *restrict amy, const double *restrict amxx,
           const double *restrict amxy, const double *restrict amyy, const double *restrict aw,
           double *restrict errs, int64_t ksz, int64_t lo, int64_t hi),
          (amx, amy, amxx, amxy, amyy, aw, errs, ksz, lo, hi)) {
    double nn = (double)(2 * ksz + 1);
    int64_t n = hi - lo + 1;
    if (n <= 0) return;
    int64_t u0 = lo + ksz;
    int64_t l0 = lo - ksz - 1;
    const double *uxs = amx + u0, *lxs = amx + l0, *uys = amy + u0, *lys = amy + l0;
    const double *uxxs = amxx + u0, *lxxs = amxx + l0, *uxys = amxy + u0, *lxys = amxy + l0;
    const double *uyys = amyy + u0, *lyys = amyy + l0, *uws = aw + u0, *lws = aw + l0;
    double *out = errs + lo;
    for (int64_t k = 0; k < n; k++) {
        double mx = uxs[k] - lxs[k];
        double my = uys[k] - lys[k];
        double mxx = uxxs[k] - lxxs[k];
        double mxy = uxys[k] - lxys[k];
        double myy = uyys[k] - lyys[k];
        double w = uws[k] - lws[k];
        double ex = mx / w;
        double ey = my / w;
        double cxx = mxx / w - ex * ex;
        double cxy = mxy / w - ex * ey;
        double cyy = myy / w - ey * ey;
        double d = cxx - cyy;
        double root = (double)at_sqrtf((float)(d * d + 4.0 * (cxy * cxy)));
        out[k] = nn * (0.5 * (cxx + cyy - root));
    }
}

/* fit_line's error for n windows of nn points with these moments. */
SIMD_VOID(moment_errors,
          (const double *restrict amx, const double *restrict amy, const double *restrict amxx,
           const double *restrict amxy, const double *restrict amyy, const double *restrict aw, double nn,
           double *restrict out, int64_t n),
          (amx, amy, amxx, amxy, amyy, aw, nn, out, n)) {
    for (int64_t k = 0; k < n; k++) {
        double w = aw[k];
        double ex = amx[k] / w;
        double ey = amy[k] / w;
        double cxx = amxx[k] / w - ex * ex;
        double cxy = amxy[k] / w - ex * ey;
        double cyy = amyy[k] / w - ey * ey;
        double d = cxx - cyy;
        double root = (double)at_sqrtf((float)(d * d + 4.0 * (cxy * cxy)));
        out[k] = nn * (0.5 * (cxx + cyy - root));
    }
}

/* The window errors of the first and last windows, which wrap around
   (and the first unwrapped one, which starts at 0): the moments as
   fit_line sums them, then fit_line's error, for all at once. */
static void edge_errors(Moments a, double *errs, int64_t sz, int64_t ksz, int64_t mid_lo, int64_t mid_hi) {
    double mx[64], my[64], mxx[64], mxy[64], myy[64], w[64], out[64];
    int64_t at[64];
    int64_t n = 0;
    for (int64_t i = 0; i < mid_lo; i++) {
        int64_t i0 = i - ksz;
        int64_t i1 = i + ksz;
        if (i0 < 0) {
            int64_t j = i0 + sz;
            mx[n] = (a.mx[sz - 1] - a.mx[j - 1]) + a.mx[i1];
            my[n] = (a.my[sz - 1] - a.my[j - 1]) + a.my[i1];
            mxx[n] = (a.mxx[sz - 1] - a.mxx[j - 1]) + a.mxx[i1];
            mxy[n] = (a.mxy[sz - 1] - a.mxy[j - 1]) + a.mxy[i1];
            myy[n] = (a.myy[sz - 1] - a.myy[j - 1]) + a.myy[i1];
            w[n] = (a.w[sz - 1] - a.w[j - 1]) + a.w[i1];
        } else if (i0 > 0) {
            mx[n] = a.mx[i1] - a.mx[i0 - 1];
            my[n] = a.my[i1] - a.my[i0 - 1];
            mxx[n] = a.mxx[i1] - a.mxx[i0 - 1];
            mxy[n] = a.mxy[i1] - a.mxy[i0 - 1];
            myy[n] = a.myy[i1] - a.myy[i0 - 1];
            w[n] = a.w[i1] - a.w[i0 - 1];
        } else {
            mx[n] = a.mx[i1];
            my[n] = a.my[i1];
            mxx[n] = a.mxx[i1];
            mxy[n] = a.mxy[i1];
            myy[n] = a.myy[i1];
            w[n] = a.w[i1];
        }
        at[n++] = i;
    }
    for (int64_t i = mid_hi + 1; i < sz; i++) {
        int64_t i0 = i - ksz;
        int64_t i1 = i + ksz - sz;
        mx[n] = (a.mx[sz - 1] - a.mx[i0 - 1]) + a.mx[i1];
        my[n] = (a.my[sz - 1] - a.my[i0 - 1]) + a.my[i1];
        mxx[n] = (a.mxx[sz - 1] - a.mxx[i0 - 1]) + a.mxx[i1];
        mxy[n] = (a.mxy[sz - 1] - a.mxy[i0 - 1]) + a.mxy[i1];
        myy[n] = (a.myy[sz - 1] - a.myy[i0 - 1]) + a.myy[i1];
        w[n] = (a.w[sz - 1] - a.w[i0 - 1]) + a.w[i1];
        at[n++] = i;
    }
    moment_errors(mx, my, mxx, mxy, myy, w, (double)(2 * ksz + 1), out, n);
    for (int64_t k = 0; k < n; k++) errs[at[k]] = out[k];
}

/* The 7-tap gaussian low-pass of the sz window errors, wrapping around. */
SIMD_VOID(filter_errors, (const double *restrict errs, double *restrict y, const double *restrict kern, int64_t sz),
          (errs, y, kern, sz)) {
    int64_t m = sz - 6;
    const double *e0 = errs, *e1 = errs + 1, *e2 = errs + 2, *e3 = errs + 3, *e4 = errs + 4, *e5 = errs + 5,
                 *e6 = errs + 6;
    double *out = y + 3;
    double k0 = kern[0], k1 = kern[1], k2 = kern[2], k3 = kern[3], k4 = kern[4], k5 = kern[5], k6 = kern[6];
    for (int64_t k = 0; k < m; k++) {
        double acc = e0[k] * k0;
        acc += e1[k] * k1;
        acc += e2[k] * k2;
        acc += e3[k] * k3;
        acc += e4[k] * k4;
        acc += e5[k] * k5;
        acc += e6[k] * k6;
        out[k] = acc;
    }
    for (int64_t e = 0; e < 6; e++) {
        int64_t iy = e < 3 ? e : sz - 6 + e;
        double acc = 0.0;
        int64_t j = iy - 3;
        if (j < 0) j += sz;
        for (int i = 0; i < 7; i++) {
            acc += errs[j] * kern[i];
            j++;
            if (j == sz) j = 0;
        }
        y[iy] = acc;
    }
}

/* exp(-j*j/2) for j in [-3, 3], rounded to f32 as the C library does. */
static double qsm_kernel[7];

static void qsm_kernel_init(void) {
    for (int i = 0; i < 7; i++) {
        int64_t j = i - 3;
        qsm_kernel[i] = (float)exp(-(double)(j * j) / 2.0);
    }
}

/* fit_line for pairs of maxima, from their moments and point counts:
   the normal (nx, ny), the error and the mean squared error. The same
   operations as fit_line, with selects for its branches. */
SIMD_VOID(pair_lines,
          (const double *restrict pmx, const double *restrict pmy, const double *restrict pmxx,
           const double *restrict pmxy, const double *restrict pmyy, const double *restrict pw,
           const double *restrict pn, double *restrict onx, double *restrict ony, double *restrict oerr,
           double *restrict omse, int64_t n),
          (pmx, pmy, pmxx, pmxy, pmyy, pw, pn, onx, ony, oerr, omse, n)) {
    for (int64_t k = 0; k < n; k++) {
        double w = pw[k];
        double ex = pmx[k] / w;
        double ey = pmy[k] / w;
        double cxx = pmxx[k] / w - ex * ex;
        double cxy = pmxy[k] / w - ex * ey;
        double cyy = pmyy[k] / w - ey * ey;
        double root = (double)at_sqrtf((float)((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy));
        double eig_small = 0.5 * (cxx + cyy - root);
        double eig = 0.5 * (cxx + cyy + root);
        double nx1 = cxx - eig;
        double ny1 = cxy;
        double m1 = nx1 * nx1 + ny1 * ny1;
        double nx2 = cxy;
        double ny2 = cyy - eig;
        double m2 = nx2 * nx2 + ny2 * ny2;
        bool big = m1 > m2;
        double nx = big ? nx1 : nx2;
        double ny = big ? ny1 : ny2;
        double m = big ? m1 : m2;
        /* length >= 0 (or NaN): the same test as !(abs(length) < 1e-12) */
        double length = (double)at_sqrtf((float)m);
        bool ok = !(length < 1e-12);
        onx[k] = ok ? nx / length : 0.0;
        ony[k] = ok ? ny / length : 0.0;
        oerr[k] = pn[k] * eig_small;
        omse[k] = eig_small;
    }
}

/* Find the four corners into indices. Returns false if no good quad
   split exists. */
static bool quad_segment_maxima(const at_quad_thresh_params *qtp, QuadScratch *sc, int64_t sz, int64_t indices[4]) {
    int64_t ksz = imin(20, sz / 12);
    if (ksz < 2) return false;
    Moments a = {sc->mx, sc->my, sc->mxx, sc->mxy, sc->myy, sc->mw};

    int64_t mid_lo = ksz + 1;
    int64_t mid_hi = sz - ksz - 1;
    window_errors(a.mx, a.my, a.mxx, a.mxy, a.myy, a.w, sc->errs, ksz, mid_lo, mid_hi);
    edge_errors(a, sc->errs, sz, ksz, mid_lo, mid_hi);

    filter_errors(sc->errs, sc->yfilt, qsm_kernel, sz);
    memcpy(sc->errs, sc->yfilt, (size_t)sz * 8);

    const double *errs = sc->errs;
    int64_t nmaxima = 0;
    int64_t *maxima = sc->maxima;
    double *maxima_errs = sc->maxima_errs;
    /* local maxima, with wrap-around at both ends */
    if (errs[0] > errs[1] && errs[0] > errs[sz - 1]) {
        maxima[0] = 0;
        maxima_errs[0] = errs[0];
        nmaxima = 1;
    }
    /* write each candidate, and keep it by moving on (no branch): at
       index i, nmaxima <= i + 1 < sz */
    for (int64_t i = 0; i < sz - 2; i++) {
        double e = errs[i + 1];
        maxima[nmaxima] = i + 1;
        maxima_errs[nmaxima] = e;
        nmaxima += (e > errs[i + 2]) & (e > errs[i]);
    }
    if (errs[sz - 1] > errs[0] && errs[sz - 1] > errs[sz - 2]) {
        maxima[nmaxima] = sz - 1;
        maxima_errs[nmaxima] = errs[sz - 1];
        nmaxima++;
    }
    if (nmaxima < 4) return false;

    int64_t max_nmaxima = qtp->max_nmaxima;
    if (nmaxima > max_nmaxima) {
        /* the threshold is the (max_nmaxima+1)-th largest error, found by
           the C library's quickselect */
        double *sel = sc->yfilt;
        memcpy(sel, maxima_errs, (size_t)nmaxima * 8);
        int64_t k = max_nmaxima;
        int64_t lo = 0, hi = nmaxima - 1;
        while (lo < hi) {
            double pivot = sel[(lo + hi) / 2];
            int64_t i = lo, j = hi;
            while (i <= j) {
                while (sel[i] > pivot) i++;
                while (sel[j] < pivot) j--;
                if (i <= j) {
                    double t = sel[i];
                    sel[i] = sel[j];
                    sel[j] = t;
                    i++;
                    j--;
                }
            }
            if (k <= j)
                hi = j;
            else if (k >= i)
                lo = i;
            else
                break;
        }
        double thresh = sel[k];
        int64_t out = 0;
        for (int64_t i = 0; i < nmaxima; i++) {
            if (maxima_errs[i] <= thresh) continue;
            maxima[out++] = maxima[i];
        }
        nmaxima = out;
    }

    int64_t best[4] = {0, 0, 0, 0};
    const double inf = (double)INFINITY;
    double best_error = inf;
    double max_dot = qtp->cos_critical_rad;
    double max_mse = qtp->max_line_fit_mse;

    /* The line fits between pairs of maxima: (a, b) for a < b, and the
       closing pairs (b, a) for b >= a + 3. ferr holds the error, or
       infinity where the fit is rejected (mse > max_mse): a sum with it
       is never < best_error, which skips the quad as a test would. */
    int64_t nm = nmaxima;
    int64_t nn = nm * nm;
    /* freed with the cluster's scratch */
    PairFit *memo = STK_ALLOC(sc->cx, PairFit, nn);
    double *ferr = STK_ALLOC(sc->cx, double, nn);
    double *pf = STK_ALLOC(sc->cx, double, 11 * nn);
    int64_t *pidx = STK_ALLOC(sc->cx, int64_t, nn);
    const int64_t *mx = maxima;
    double *pmx = pf, *pmy = pf + nn, *pmxx = pf + 2 * nn, *pmxy = pf + 3 * nn, *pmyy = pf + 4 * nn;
    double *pw = pf + 5 * nn, *pn = pf + 6 * nn;
    int64_t np = 0;
    for (int64_t ma = 0; ma < nm; ma++) {
        for (int64_t mb = ma + 1; mb < nm; mb++) {
            /* a < b: no wrap */
            int64_t i0 = mx[ma], i1 = mx[mb];
            double smx = a.mx[i1], smy = a.my[i1], smxx = a.mxx[i1], smxy = a.mxy[i1], smyy = a.myy[i1];
            double sw = a.w[i1];
            if (i0 > 0) {
                smx -= a.mx[i0 - 1];
                smy -= a.my[i0 - 1];
                smxx -= a.mxx[i0 - 1];
                smxy -= a.mxy[i0 - 1];
                smyy -= a.myy[i0 - 1];
                sw -= a.w[i0 - 1];
            }
            pmx[np] = smx;
            pmy[np] = smy;
            pmxx[np] = smxx;
            pmxy[np] = smxy;
            pmyy[np] = smyy;
            pw[np] = sw;
            pn[np] = (double)(i1 - i0 + 1);
            pidx[np] = ma * nm + mb;
            np++;
            if (mb >= ma + 3) {
                /* the closing pair b..a wraps around */
                int64_t j0 = mx[mb], j1 = mx[ma];
                pmx[np] = (a.mx[sz - 1] - a.mx[j0 - 1]) + a.mx[j1];
                pmy[np] = (a.my[sz - 1] - a.my[j0 - 1]) + a.my[j1];
                pmxx[np] = (a.mxx[sz - 1] - a.mxx[j0 - 1]) + a.mxx[j1];
                pmxy[np] = (a.mxy[sz - 1] - a.mxy[j0 - 1]) + a.mxy[j1];
                pmyy[np] = (a.myy[sz - 1] - a.myy[j0 - 1]) + a.myy[j1];
                pw[np] = (a.w[sz - 1] - a.w[j0 - 1]) + a.w[j1];
                pn[np] = (double)(sz - j0 + j1 + 1);
                pidx[np] = mb * nm + ma;
                np++;
            }
        }
    }
    double *onx = pf + 7 * nn, *ony = pf + 8 * nn, *oerr = pf + 9 * nn, *omse = pf + 10 * nn;
    pair_lines(pmx, pmy, pmxx, pmxy, pmyy, pw, pn, onx, ony, oerr, omse, np);
    for (int64_t k = 0; k < np; k++) {
        int64_t m = pidx[k];
        memo[m] = (PairFit){{0.0, 0.0, onx[k], ony[k]}, oerr[k], omse[k]};
        ferr[m] = omse[k] > max_mse ? inf : oerr[k];
    }

    for (int64_t m0 = 0; m0 < nm - 3; m0++) {
        int64_t i0 = mx[m0];
        for (int64_t m1 = m0 + 1; m1 < nm - 2; m1++) {
            int64_t p01 = m0 * nm + m1;
            if (memo[p01].mse > max_mse) continue;
            for (int64_t m2 = m1 + 1; m2 < nm - 1; m2++) {
                int64_t p12 = m1 * nm + m2;
                if (memo[p12].mse > max_mse) continue;
                double dot = memo[p01].par[2] * memo[p12].par[2] + memo[p01].par[3] * memo[p12].par[3];
                if (fabs(dot) > max_dot) continue;
                double e012 = memo[p01].err + memo[p12].err;
                for (int64_t m3 = m2 + 1; m3 < nm; m3++) {
                    double err = e012 + ferr[m2 * nm + m3] + ferr[m3 * nm + m0];
                    if (err < best_error) {
                        best_error = err;
                        best[0] = i0;
                        best[1] = mx[m1];
                        best[2] = mx[m2];
                        best[3] = mx[m3];
                    }
                }
            }
        }
    }

    if (best_error == inf) return false;
    memcpy(indices, best, sizeof best);
    return best_error / (double)sz < max_mse;
}

/* The gradient loops prefetch the three image rows of the point this
   many points ahead (16 was slower, 8 gave less). */
#define PF_PTS 32

/* The image index of the pixel of each point: (ix, iy) = ((x + 1) / 2,
   (y + 1) / 2) of the point's doubled coordinates, or -1 where a
   neighbor of it is outside the image (gradient 0). */
SIMD_VOID(point_centers, (const Pt *restrict pts, int64_t n, int32_t iw, int32_t ih, int32_t *restrict ci),
          (pts, n, iw, ih, ci)) {
    for (int64_t i = 0; i < n; i++) {
        int32_t px = pts[i].x;
        int32_t py = pts[i].y;
        int32_t ix = (px >> 1) + (px & 1);
        int32_t iy = (py >> 1) + (py & 1);
        bool inside = ix > 0 && ix + 1 < iw && iy > 0 && iy + 1 < ih;
        ci[i] = inside ? iy * iw + ix : -1;
    }
}

static void point_gradients(const int32_t *ci, int64_t n, const uint8_t *im, int64_t iw, double *wb) {
    for (int64_t i = 0; i < n; i++) {
        if (i + PF_PTS < n) {
            int64_t qc = ci[i + PF_PTS];
            if (qc >= 0) {
                __builtin_prefetch(im + qc - iw);
                __builtin_prefetch(im + qc);
                __builtin_prefetch(im + qc + iw);
            }
        }
        int64_t c = ci[i];
        int32_t g2 = 0;
        if (c >= 0) {
            int32_t gx = (int32_t)im[c + 1] - im[c - 1];
            int32_t gy = (int32_t)im[c + iw] - im[c - iw];
            g2 = gx * gx + gy * gy;
        }
        wb[i] = g2;
    }
}

SIMD_VOID(sqrt_plus_one, (double *restrict v, int64_t n), (v, n)) {
    for (int64_t i = 0; i < n; i++) v[i] = at_sqrt(v[i]) + 1.0;
}

/* Cumulative weighted moments of the points in key order. */
static void compute_lfps(QuadScratch *sc, const uint64_t *keys, const Pt *pts, int64_t sz, const uint8_t *im,
                         int64_t iw, int64_t ih) {
    double *wb = sc->wbuf;
    int32_t *ci = sc->cidx;
    point_centers(pts, sz, (int32_t)iw, (int32_t)ih, ci);
    /* squared gradients (0 at the border), then the weights at_sqrt(g2) + 1 */
    point_gradients(ci, sz, im, iw, wb);
    sqrt_plus_one(wb, sz);
    double *mx = sc->mx, *my = sc->my, *mxx = sc->mxx, *mxy = sc->mxy, *myy = sc->myy, *mw = sc->mw;
    double smx = 0.0, smy = 0.0, smxx = 0.0, smyy = 0.0, smxy = 0.0, sw = 0.0;
    for (int64_t k = 0; k < sz; k++) {
        int64_t src = (uint32_t)~(uint32_t)keys[k];
        double w = wb[src];
        Pt p = pts[src];
        double fx = (double)p.x * 0.5 + 0.5;
        double fy = (double)p.y * 0.5 + 0.5;
        smx += w * fx;
        smy += w * fy;
        smxx += w * fx * fx;
        smxy += w * fx * fy;
        smyy += w * fy * fy;
        sw += w;
        mx[k] = smx;
        my[k] = smy;
        mxx[k] = smxx;
        mxy[k] = smxy;
        myy[k] = smyy;
        mw[k] = sw;
    }
}

AT_INLINE double sq(double x) { return x * x; }

/* From this many points on, the radix sort is faster than keysort4. */
#define RADIX_MIN 64

/* Fit a quad to one cluster of sz points. im is the (decimated) image,
   iw x ih, rows iw bytes apart. */
static bool fit_quad_in(const at_quad_thresh_params *qtp, QuadScratch *sc, const Pt *pts, int64_t sz,
                        const uint8_t *im, int64_t iw, int64_t ih, int64_t tag_width, bool normal_border,
                        bool reversed_border, const uint16_t bb[4], Quad *quad) {
    int64_t xmin = bb[0], xmax = bb[1], ymin = bb[2], ymax = bb[3];
    float cx = (float)((double)(xmin + xmax) * 0.5 + 0.05118);
    float cy = (float)((double)(ymin + ymax) * 0.5 + -0.028581);

    uint64_t *keys = sc->keys;
    float dot = point_keys(pts, sz, cx, cy, keys);

    quad->reversed_border = dot < 0.0f;
    if (!reversed_border && quad->reversed_border) return false;
    if (!normal_border && !quad->reversed_border) return false;

    int at = 0;
    if (sz >= RADIX_MIN)
        at = keysort_blocks(keys, sz, sc->tmp, sc->bh, sc->rk);
    else if (sz >= 2)
        keysort4(keys, sc->tmp, sz, false);
    const uint64_t *sorted = at == 1 ? sc->rk : at == 2 ? sc->tmp : keys;
    compute_lfps(sc, sorted, pts, sz, im, iw, ih);

    int64_t indices[4];
    if (!quad_segment_maxima(qtp, sc, sz, indices)) return false;

    double lines[4][4];
    Moments a = {sc->mx, sc->my, sc->mxx, sc->mxy, sc->myy, sc->mw};
    for (int i = 0; i < 4; i++) {
        double mse;
        fit_line(a, sz, indices[i], indices[(i + 1) & 3], lines[i], &mse);
        if (mse > qtp->max_line_fit_mse) return false;
    }

    for (int i = 0; i < 4; i++) {
        int j = (i + 1) & 3;
        double a00 = lines[i][3];
        double a01 = -lines[j][3];
        double a10 = -lines[i][2];
        double a11 = lines[j][2];
        double b0 = -lines[i][0] + lines[j][0];
        double b1 = -lines[i][1] + lines[j][1];
        double det = a00 * a11 - a10 * a01;
        if (fabs(det) < 0.001) return false;
        double w00 = a11 / det;
        double w01 = -a01 / det;
        double l0 = w00 * b0 + w01 * b1;
        quad->p[i][0] = (float)(lines[i][0] + l0 * a00);
        quad->p[i][1] = (float)(lines[i][1] + l0 * a10);
    }

    /* reject quads that are too small */
    double area = 0.0;
    double length[3];
    for (int i = 0; i < 3; i++) {
        int ia = i, ib = (i + 1) % 3;
        length[i] = at_sqrt(sq(quad->p[ib][0] - quad->p[ia][0]) + sq(quad->p[ib][1] - quad->p[ia][1]));
    }
    double p = (length[0] + length[1] + length[2]) / 2.0;
    area += at_sqrt(p * (p - length[0]) * (p - length[1]) * (p - length[2]));
    static const int idxs[4] = {2, 3, 0, 2};
    for (int i = 0; i < 3; i++) {
        int ia = idxs[i], ib = idxs[i + 1];
        length[i] = at_sqrt(sq(quad->p[ib][0] - quad->p[ia][0]) + sq(quad->p[ib][1] - quad->p[ia][1]));
    }
    p = (length[0] + length[1] + length[2]) / 2.0;
    area += at_sqrt(p * (p - length[0]) * (p - length[1]) * (p - length[2]));
    if (area < 0.95 * (double)tag_width * (double)tag_width) return false;

    /* reject quads whose corners are not convex enough */
    double ccr = qtp->cos_critical_rad;
    for (int i = 0; i < 4; i++) {
        int i0 = i, i1 = (i + 1) & 3, i2 = (i + 2) & 3;
        double dx1 = quad->p[i1][0] - quad->p[i0][0];
        double dy1 = quad->p[i1][1] - quad->p[i0][1];
        double dx2 = quad->p[i2][0] - quad->p[i1][0];
        double dy2 = quad->p[i2][1] - quad->p[i1][1];
        double den = at_sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2));
        if (den == 0.0) return false;
        double cos_dtheta = (dx1 * dx2 + dy1 * dy2) / den;
        if ((cos_dtheta > ccr || cos_dtheta < -ccr) || dx1 * dy2 < dy1 * dx2) return false;
    }
    return true;
}

static bool fit_quad(at_ctx *cx, const at_quad_thresh_params *qtp, const Pt *pts, int64_t sz, const uint8_t *im,
                     int64_t iw, int64_t ih, int64_t tag_width, bool normal_border, bool reversed_border,
                     Quad *quad) {
    uint16_t bb[4];
    point_bounds(pts, sz, bb);
    int64_t xmin = bb[0], xmax = bb[1], ymin = bb[2], ymax = bb[3];
    if ((xmax - xmin) * (ymax - ymin) < tag_width) return false;
    at_mark mk = stk_enter(cx);
    QuadScratch sc;
    quad_scratch_alloc(cx, &sc, sz);
    bool ok = fit_quad_in(qtp, &sc, pts, sz, im, iw, ih, tag_width, normal_border, reversed_border, bb, quad);
    stk_leave(cx, mk);
    return ok;
}

/* Fit quads to every cluster, in cluster order. Cluster c is the points
   cstart[c]..cstart[c + 1] - gap of cpts. */
static void fit_quads(at_ctx *cx, const at_quad_thresh_params *qtp, const Pt *cpts, const uint32_t *cstart,
                      int64_t nclusters, int64_t gap, const uint8_t *im, int64_t w, int64_t h, int64_t tag_width,
                      bool normal_border, bool reversed_border, vec_quad *quads) {
    int64_t min_px = qtp->min_cluster_pixels;
    int64_t max_px = 2 * (2 * w + 2 * h);
    for (int64_t c = 0; c < nclusters; c++) {
        int64_t s0 = cstart[c];
        int64_t s1 = (int64_t)cstart[c + 1] - gap;
        int64_t n = s1 - s0;
        if (n < min_px || n > max_px) continue;
        Quad q;
        memset(&q, 0, sizeof q);
        if (fit_quad(cx, qtp, cpts + s0, n, im, w, h, tag_width, normal_border, reversed_border, &q))
            svec_push(*quads, q);
    }
}
