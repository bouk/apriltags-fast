/* Adaptive thresholding and row run-length encoding.

   The image is cut into 4x4 tiles. Each pixel is compared against the
   mid-range of the 3x3 tile neighborhood around its tile. Tiles with too
   little contrast become 127 ("unknown"). Each row is then encoded as
   runs of equal non-127 pixels over columns [0, w-2]; the last column
   never takes part in runs. */
#include "common.h"

AT_INLINE uint8_t max8(uint8_t a, uint8_t b) { return a > b ? a : b; }
AT_INLINE uint8_t min8(uint8_t a, uint8_t b) { return a < b ? a : b; }

/* Min and max of each 4x4 tile of one tile row (tw tiles). The four rows
   are the tile row's pixel rows. */
SIMD_VOID(tile_minmax_row,
          (const uint8_t *restrict r0, const uint8_t *restrict r1, const uint8_t *restrict r2,
           const uint8_t *restrict r3, uint8_t *restrict mx, uint8_t *restrict mn, int64_t tw),
          (r0, r1, r2, r3, mx, mn, tw)) {
    for (int64_t tx = 0; tx < tw; tx++) {
        int64_t x = 4 * tx;
        uint8_t hi = max8(max8(max8(r0[x], r0[x + 1]), max8(r0[x + 2], r0[x + 3])),
                          max8(max8(r1[x], r1[x + 1]), max8(r1[x + 2], r1[x + 3])));
        hi = max8(hi, max8(max8(max8(r2[x], r2[x + 1]), max8(r2[x + 2], r2[x + 3])),
                           max8(max8(r3[x], r3[x + 1]), max8(r3[x + 2], r3[x + 3]))));
        uint8_t lo = min8(min8(min8(r0[x], r0[x + 1]), min8(r0[x + 2], r0[x + 3])),
                          min8(min8(r1[x], r1[x + 1]), min8(r1[x + 2], r1[x + 3])));
        lo = min8(lo, min8(min8(min8(r2[x], r2[x + 1]), min8(r2[x + 2], r2[x + 3])),
                           min8(min8(r3[x], r3[x + 1]), min8(r3[x + 2], r3[x + 3]))));
        mx[tx] = hi;
        mn[tx] = lo;
    }
}

/* 3-wide horizontal max/min of one tile row (edges use what exists). */
SIMD_VOID(tile_hblur_row,
          (const uint8_t *restrict mx, const uint8_t *restrict mn, uint8_t *restrict omx, uint8_t *restrict omn,
           int64_t tw),
          (mx, mn, omx, omn, tw)) {
    if (tw == 1) {
        omx[0] = mx[0];
        omn[0] = mn[0];
        return;
    }
    omx[0] = max8(mx[0], mx[1]);
    omn[0] = min8(mn[0], mn[1]);
    for (int64_t tx = 1; tx < tw - 1; tx++) {
        omx[tx] = max8(max8(mx[tx - 1], mx[tx]), mx[tx + 1]);
        omn[tx] = min8(min8(mn[tx - 1], mn[tx]), mn[tx + 1]);
    }
    omx[tw - 1] = max8(mx[tw - 2], mx[tw - 1]);
    omn[tw - 1] = min8(mn[tw - 2], mn[tw - 1]);
}

SIMD_VOID(vmax3, (const uint8_t *restrict a, const uint8_t *restrict b, const uint8_t *restrict c, uint8_t *restrict o,
                  int64_t n),
          (a, b, c, o, n)) {
    for (int64_t i = 0; i < n; i++) o[i] = max8(max8(a[i], b[i]), c[i]);
}

SIMD_VOID(vmin3, (const uint8_t *restrict a, const uint8_t *restrict b, const uint8_t *restrict c, uint8_t *restrict o,
                  int64_t n),
          (a, b, c, o, n)) {
    for (int64_t i = 0; i < n; i++) o[i] = min8(min8(a[i], b[i]), c[i]);
}

/* The per-pixel thresholds of one tile row (w pixels, tw tiles): thr[x]
   is the threshold of pixel column x, flat[x] is 1 where the tile has
   too little contrast (output 127). Columns past the last whole tile use
   the last tile and are never flat. With allow_flat false no column is
   flat (the rows below the last whole tile row). */
SIMD_VOID(expand_thresholds,
          (const uint8_t *restrict mx, const uint8_t *restrict mn, int64_t tw, int32_t diff, bool allow_flat,
           uint8_t *restrict thr, uint8_t *restrict flat, int64_t w),
          (mx, mn, tw, diff, allow_flat, thr, flat, w)) {
    for (int64_t tx = 0; tx < tw; tx++) {
        int32_t hi = mx[tx];
        int32_t lo = mn[tx];
        uint8_t t = (uint8_t)(lo + (hi - lo) / 2);
        uint8_t f = allow_flat && hi - lo < diff ? 1 : 0;
        for (int d = 0; d < 4; d++) {
            thr[4 * tx + d] = t;
            flat[4 * tx + d] = f;
        }
    }
    int32_t hi = mx[tw - 1];
    int32_t lo = mn[tw - 1];
    uint8_t t = (uint8_t)(lo + (hi - lo) / 2);
    for (int64_t x = 4 * tw; x < w; x++) {
        thr[x] = t;
        flat[x] = 0;
    }
}

SIMD_VOID(threshold_row,
          (const uint8_t *restrict src, const uint8_t *restrict thr, const uint8_t *restrict flat,
           uint8_t *restrict out, int64_t w),
          (src, thr, flat, out, w)) {
    for (int64_t x = 0; x < w; x++) {
        uint8_t v = src[x] > thr[x] ? 255 : 0;
        out[x] = flat[x] != 0 ? 127 : v;
    }
}

/* The rows of tile rows [ty0, ty1) of the thresholded image; the last
   tile row also takes the rows below the last whole tile row. rows[y -
   row0] is image row y, for at least the rows of tile rows ty0-1 to ty1
   that exist (a row table: the rows can be in different arrays, so none
   is copied); out gets the band's rows, w bytes apart. */
static void threshold_band(at_ctx *cx, const uint8_t *const *rows, int64_t row0, int64_t w, int64_t h, int32_t diff,
                           int64_t ty0, int64_t ty1, uint8_t *out) {
    int64_t tw = w / 4;
    int64_t th = h / 4;
    /* tile rows t0..t1 (exclusive) take part */
    int64_t t0 = imax(ty0 - 1, 0);
    int64_t t1 = imin(ty1 + 1, th);
    int64_t nt = t1 - t0;
    at_mark mk = stk_enter(cx);
    uint8_t *buf = STK_ALLOC(cx, uint8_t, 4 * tw * nt + 2 * w);
    uint8_t *mx = buf, *mn = mx + tw * nt, *hmx = mn + tw * nt, *hmn = hmx + tw * nt;
    uint8_t *thr = hmn + tw * nt, *flat = thr + w;
    for (int64_t k = 0; k < nt; k++) {
        const uint8_t *const *r = rows + 4 * (t0 + k) - row0;
        tile_minmax_row(r[0], r[1], r[2], r[3], mx + k * tw, mn + k * tw, tw);
    }
    /* 3x3 max/min over the tile grid, as a horizontal then a vertical
       pass (both exclude cells outside the grid) */
    for (int64_t k = 0; k < nt; k++) tile_hblur_row(mx + k * tw, mn + k * tw, hmx + k * tw, hmn + k * tw, tw);
    for (int64_t ty = ty0; ty < ty1; ty++) {
        int64_t k = ty - t0;
        int64_t up = imax(ty - 1, 0) - t0;
        int64_t dn = imin(ty + 1, th - 1) - t0;
        vmax3(hmx + up * tw, hmx + k * tw, hmx + dn * tw, mx + k * tw, tw);
        vmin3(hmn + up * tw, hmn + k * tw, hmn + dn * tw, mn + k * tw, tw);
    }
    int64_t ob = 4 * ty0;
    for (int64_t ty = ty0; ty < ty1; ty++) {
        int64_t k = ty - t0;
        expand_thresholds(mx + k * tw, mn + k * tw, tw, diff, true, thr, flat, w);
        for (int64_t dy = 0; dy < 4; dy++) {
            int64_t y = 4 * ty + dy;
            threshold_row(rows[y - row0], thr, flat, out + (y - ob) * w, w);
        }
    }
    if (ty1 == th && 4 * th < h) {
        int64_t k = th - 1 - t0;
        expand_thresholds(mx + k * tw, mn + k * tw, tw, diff, false, thr, flat, w);
        for (int64_t y = 4 * th; y < h; y++) threshold_row(rows[y - row0], thr, flat, out + (y - ob) * w, w);
    }
    stk_leave(cx, mk);
}

/* Threshold im (w x h, rows s bytes apart) into out (rows w bytes
   apart). Requires w >= 4 and h >= 4. */
static void threshold_image(at_ctx *cx, const uint8_t *im, int64_t w, int64_t h, int64_t s, int32_t diff,
                            uint8_t *out) {
    at_mark mk = stk_enter(cx);
    const uint8_t **rows = STK_ALLOC(cx, const uint8_t *, h);
    for (int64_t y = 0; y < h; y++) rows[y] = im + y * s;
    threshold_band(cx, rows, 0, w, h, diff, 0, h / 4, out);
    stk_leave(cx, mk);
}

/* The image rows of tile rows [ty0, ty1) (with the rows below the last
   whole tile row for the last band). */
static void band_rows(int64_t h, int64_t ty0, int64_t ty1, int64_t *y0, int64_t *y1) {
    int64_t th = h / 4;
    *y0 = 4 * ty0;
    *y1 = ty1 == th ? h : 4 * ty1;
}

/* The 3x3 dilate-then-erode of the C library's deglitch option. The
   border rows and columns of the intermediate image are zero, as there. */
static void deglitch(at_ctx *cx, uint8_t *t, int64_t w, int64_t h) {
    at_mark mk = stk_enter(cx);
    uint8_t *tmp = STK_ZALLOC(cx, uint8_t, w * h);
    for (int64_t y = 1; y < h - 1; y++) {
        for (int64_t x = 1; x < w - 1; x++) {
            uint8_t m = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) m = max8(m, t[(y + dy) * w + x + dx]);
            tmp[y * w + x] = m;
        }
    }
    for (int64_t y = 1; y < h - 1; y++) {
        for (int64_t x = 1; x < w - 1; x++) {
            uint8_t m = 255;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) m = min8(m, tmp[(y + dy) * w + x + dx]);
            t[y * w + x] = m;
        }
    }
    stk_leave(cx, mk);
}

/* c[i] = 1 where row[i + 1] differs from row[i], else 0 (n entries). */
SIMD_VOID(row_changes, (const uint8_t *restrict row, uint8_t *restrict c, int64_t n), (row, c, n)) {
    const uint8_t *a = row + 1;
    for (int64_t i = 0; i < n; i++) c[i] = a[i] != row[i] ? 1 : 0;
}

/* Bit k of p[i] is c[8i + k] (0 or 1), for i < n. */
SIMD_VOID(pack_bits, (const uint8_t *restrict c, uint8_t *restrict p, int64_t n), (c, p, n)) {
    for (int64_t i = 0; i < n; i++) {
        int64_t j = 8 * i;
        p[i] = (uint8_t)((c[j] | (c[j + 1] << 1)) | ((c[j + 2] << 2) | (c[j + 3] << 3)) |
                         ((c[j + 4] << 4) | (c[j + 5] << 5)) | ((c[j + 6] << 6) | (c[j + 7] << 7)));
    }
}

/* The runs of one thresholded row of w pixels (pixels [0, w-2]) into
   out (at least w - 1 entries). Returns the number of runs. cb and pb
   are scratch: w + 64 and w / 8 + 8 bytes. */
static int64_t rle_row(const uint8_t *row, int64_t w, Run *out, uint8_t *cb, uint8_t *pb) {
    int64_t xmax = w - 2;
    /* the changes 64 pixels at a time, as bit masks: block k is x in
       1 + 64k .. 64k + 64, the changes cb[64k .. 64k + 64]; the last
       block has zeros past xmax */
    int64_t nb = (xmax + 63) / 64;
    row_changes(row, cb, xmax);
    memset(cb + xmax, 0, (size_t)(64 * nb - xmax));
    pack_bits(cb, pb, 8 * nb);
    int64_t n = 0;
    int64_t start = 0;
    uint8_t v = row[0];
    for (int64_t k = 0; k < nb; k++) {
        int64_t x = 1 + 64 * k;
        uint64_t m = load64(pb + 8 * k);
        while (m != 0) {
            int64_t b = __builtin_ctzll(m);
            m &= m - 1;
            int64_t cx = x + b;
            /* no branch on 127 (it alternates with the other values) */
            out[n] = (Run){(uint16_t)start, (uint16_t)(cx - 1), v};
            n += v != 127;
            start = cx;
            v = row[cx];
        }
    }
    if (v != 127) {
        out[n] = (Run){(uint16_t)start, (uint16_t)xmax, v};
        n++;
    }
    return n;
}

/* Run-length encode all rows into runs (on a stack of its own: each
   row's runs go straight to its end). row_off gets h+1 entries: row y's
   runs are runs[row_off[y]..row_off[y+1]]. */
static void rle_image(at_ctx *cx, const uint8_t *t, int64_t w, int64_t h, vec_run *runs, uint32_t *row_off) {
    at_mark mk = stk_enter(cx);
    uint8_t *cb = STK_ALLOC(cx, uint8_t, w + 64);
    uint8_t *pb = STK_ALLOC(cx, uint8_t, w / 8 + 8);
    row_off[0] = 0;
    int64_t n = runs->len;
    for (int64_t y = 0; y < h; y++) {
        n += rle_row(t + y * w, w, runs->data + n, cb, pb);
        row_off[y + 1] = (uint32_t)n;
    }
    runs->len = n;
    stk_leave(cx, mk);
}
