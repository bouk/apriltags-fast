/* Tag decoding: edge refinement, homography, bit sampling, code lookup,
   and the removal of overlapping detections. The arithmetic mirrors the
   C library's AVX2 build bit for bit. */
#include "common.h"

/* C's (int) conversion of a double on x86-64: truncation, with INT_MIN
   for NaN and values out of range. */
AT_INLINE int64_t ctrunc(double x) {
    if (x > -2147483649.0 && x < 2147483648.0) return (int64_t)x;
    return -2147483648LL;
}

/* ---------------------------------------------------------------------
   Quick decode: the codes are split into 4 chunks; a code within the
   maximum hamming distance matches the received code exactly in at
   least one chunk. Each chunk has a table of the code ids per chunk
   value. The tables of all families are in three arrays of the detector
   (stacks of their own, which grow in place). */

static QuickDecode quick_decode_init(at_ctx *cx, const at_family *fam, int64_t maxhamming, vec_u32 *offsets,
                                     vec_u16 *ids, vec_u64 *bcodes) {
    int64_t nbits = fam->nbits;
    int64_t ncodes = fam->ncodes;
    int64_t chunk_size = (nbits + 3) / 4;
    int64_t capacity = (int64_t)1 << chunk_size;
    uint64_t mask = (uint64_t)(capacity - 1);
    QuickDecode qd = {maxhamming, chunk_size, capacity, offsets->len, ids->len};
    svec_resize0(*offsets, offsets->len + 4 * (capacity + 1));
    svec_resize0(*ids, ids->len + 4 * ncodes);
    svec_resize0(*bcodes, bcodes->len + 4 * ncodes);
    uint32_t *off = offsets->data + qd.off_base;
    uint16_t *idv = ids->data + qd.ids_base;
    uint64_t *cv = bcodes->data + qd.ids_base;
    for (int64_t i = 0; i < ncodes; i++) {
        uint64_t code = fam->codes[i];
        for (int64_t j = 0; j < 4; j++) {
            int64_t val = (int64_t)((code >> (j * chunk_size)) & mask);
            off[j * (capacity + 1) + val + 1] += 1;
        }
    }
    for (int64_t j = 0; j < 4; j++)
        for (int64_t v = 0; v < capacity; v++) off[j * (capacity + 1) + v + 1] += off[j * (capacity + 1) + v];
    at_mark mk = stk_enter(cx);
    uint32_t *cursor = STK_ALLOC(cx, uint32_t, 4 * (capacity + 1));
    memcpy(cursor, off, (size_t)(4 * (capacity + 1)) * 4);
    for (int64_t i = 0; i < ncodes; i++) {
        uint64_t code = fam->codes[i];
        for (int64_t j = 0; j < 4; j++) {
            int64_t val = (int64_t)((code >> (j * chunk_size)) & mask);
            int64_t c = j * (capacity + 1) + val;
            idv[j * ncodes + cursor[c]] = (uint16_t)i;
            cv[j * ncodes + cursor[c]] = code;
            cursor[c] += 1;
        }
    }
    stk_leave(cx, mk);
    return qd;
}

/* Rotate a code by 90 degrees. */
static uint64_t rotate90(uint64_t w0, int64_t nbits) {
    int64_t p = nbits;
    uint64_t l = 0;
    if (nbits % 4 == 1) {
        p = nbits - 1;
        l = 1;
    }
    uint64_t w = ((w0 >> l) << (p / 4 + (int64_t)l)) | (w0 >> (3 * p / 4 + (int64_t)l) << l) | (w0 & l);
    w &= ((uint64_t)1 << nbits) - 1;
    return w;
}

/* A code lookup in one u32: the id, the hamming distance (255: no match)
   and the rotation. */
AT_INLINE uint32_t pack_lookup(int64_t id, int64_t hamming, int64_t rotation) {
    return (uint32_t)(id | hamming << 16 | rotation << 24);
}

AT_INLINE void unpack_lookup(uint32_t v, int64_t *id, int64_t *hamming, int64_t *rotation) {
    *id = v & 0xffff;
    *hamming = (v >> 16) & 0xff;
    *rotation = v >> 24;
}

/* Look up a received code: the packed id, hamming distance and rotation,
   or hamming 255 when nothing matches. */
static uint32_t quick_decode_codeword(const FamLookup *lk, QdTables t, uint64_t rcode0) {
    const QuickDecode *qd = &lk->qd;
    uint64_t rcode = rcode0;
    uint64_t mask = (uint64_t)(qd->capacity - 1);
    for (int64_t ridx = 0; ridx < 4; ridx++) {
        for (int64_t i = 0; i < 4; i++) {
            int64_t val = (int64_t)((rcode >> (i * qd->chunk_size)) & mask);
            int64_t ob = qd->off_base + i * (qd->capacity + 1) + val;
            int64_t base = qd->ids_base + i * lk->ncodes;
            int64_t start = base + t.offsets[ob];
            int64_t end = base + t.offsets[ob + 1];
            /* the codes of a chunk value are next to each other */
            for (int64_t j = start; j < end; j++) {
                int64_t hamming = __builtin_popcountll(t.codes[j] ^ rcode);
                if (hamming <= qd->maxhamming) return pack_lookup(t.ids[j], hamming, ridx);
            }
        }
        rcode = rotate90(rcode, lk->nbits);
    }
    return pack_lookup(65535, 255, 0);
}

/* ---------------------------------------------------------------------
   Homography. */

/* The homography from the four correspondences (tag x, tag y, image x,
   image y). Returns false when the system is singular. */
static bool homography_compute2(const double c[4][4], double hres[9]) {
    double a[72];
    memset(a, 0, sizeof a);
    for (int i = 0; i < 4; i++) {
        int r0 = 18 * i, r1 = r0 + 9;
        a[r0 + 0] = c[i][0];
        a[r0 + 1] = c[i][1];
        a[r0 + 2] = 1.0;
        a[r0 + 6] = -c[i][0] * c[i][2];
        a[r0 + 7] = -c[i][1] * c[i][2];
        a[r0 + 8] = c[i][2];
        a[r1 + 3] = c[i][0];
        a[r1 + 4] = c[i][1];
        a[r1 + 5] = 1.0;
        a[r1 + 6] = -c[i][0] * c[i][3];
        a[r1 + 7] = -c[i][1] * c[i][3];
        a[r1 + 8] = c[i][3];
    }
    for (int col = 0; col < 8; col++) {
        double max_val = 0.0;
        int max_idx = -1;
        for (int row = col; row < 8; row++) {
            double v = fabs(a[row * 9 + col]);
            if (v > max_val) {
                max_val = v;
                max_idx = row;
            }
        }
        if (max_idx < 0) return false;
        if (max_val < 1e-10) return false;
        if (max_idx != col) {
            for (int i = col; i < 9; i++) {
                double t = a[col * 9 + i];
                a[col * 9 + i] = a[max_idx * 9 + i];
                a[max_idx * 9 + i] = t;
            }
        }
        for (int i = col + 1; i < 8; i++) {
            double f = a[i * 9 + col] / a[col * 9 + col];
            a[i * 9 + col] = 0.0;
            for (int j = col + 1; j < 9; j++) a[i * 9 + j] -= f * a[col * 9 + j];
        }
    }
    for (int col = 7; col >= 0; col--) {
        double sum = 0.0;
        for (int i = col + 1; i < 8; i++) sum += a[col * 9 + i] * a[i * 9 + 8];
        a[col * 9 + 8] = (a[col * 9 + 8] - sum) / a[col * 9 + col];
    }
    const double h[9] = {a[8], a[17], a[26], a[35], a[44], a[53], a[62], a[71], 1.0};
    memcpy(hres, h, sizeof h);
    return true;
}

/* Is the 3x3 matrix invertible by the C library's PLU decomposition? */
static bool matd_invertible(const double m[9]) {
    double lu[9];
    memcpy(lu, m, sizeof lu);
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            int kmax = i < j ? i : j;
            double acc = 0.0;
            for (int k = 0; k < kmax; k++) acc += lu[i * 3 + k] * lu[k * 3 + j];
            lu[i * 3 + j] -= acc;
        }
        int p = j;
        for (int i = j + 1; i < 3; i++)
            if (fabs(lu[i * 3 + j]) > fabs(lu[p * 3 + j])) p = i;
        if (p != j) {
            for (int k = 0; k < 3; k++) {
                double t = lu[p * 3 + k];
                lu[p * 3 + k] = lu[j * 3 + k];
                lu[j * 3 + k] = t;
            }
        }
        double lujj = lu[j * 3 + j];
        if (fabs(lujj) < 1e-8) return false;
        if (lujj != 0.0) {
            double inv = 1.0 / lujj;
            for (int i = j + 1; i < 3; i++) lu[i * 3 + j] *= inv;
        }
    }
    return true;
}

static bool quad_update_homography(Quad *q) {
    double corr[4][4];
    for (int i = 0; i < 4; i++) {
        corr[i][0] = i == 0 || i == 3 ? -1.0 : 1.0;
        corr[i][1] = i == 0 || i == 1 ? -1.0 : 1.0;
        corr[i][2] = q->p[i][0];
        corr[i][3] = q->p[i][1];
    }
    double h[9];
    if (!homography_compute2(corr, h)) return false;
    memcpy(q->H, h, sizeof h);
    return matd_invertible(h);
}

AT_INLINE void homography_project(const double h[9], double x, double y, double *ox, double *oy) {
    double xx = h[0] * x + h[1] * y + h[2];
    double yy = h[3] * x + h[4] * y + h[5];
    double zz = h[6] * x + h[7] * y + h[8];
    *ox = xx / zz;
    *oy = yy / zz;
}

/* ---------------------------------------------------------------------
   Bit sampling. */

/* The gray value model intensity = C0*x + C1*y + C2, fit by least
   squares. */
typedef struct {
    double a[9], b[3], c[3];
} GrayModel;

AT_INLINE void graymodel_add(GrayModel *g, double x, double y, double gray) {
    g->a[0] += x * x;
    g->a[1] += x * y;
    g->a[2] += x;
    g->a[4] += y * y;
    g->a[5] += y;
    g->a[8] += 1.0;
    g->b[0] += x * gray;
    g->b[1] += y * gray;
    g->b[2] += gray;
}

/* mat33_sym_solve: Cholesky, lower triangular inverse, two products. */
static void graymodel_solve(GrayModel *g) {
    const double *a = g->a;
    double l[9] = {0};
    l[0] = at_sqrt(a[0]);
    l[3] = a[1] / l[0];
    l[6] = a[2] / l[0];
    l[4] = at_sqrt(a[4] - l[3] * l[3]);
    l[7] = (a[5] - l[3] * l[6]) / l[4];
    l[8] = at_sqrt(a[8] - l[6] * l[6] - l[7] * l[7]);
    double m[9] = {0};
    m[0] = 1.0 / l[0];
    m[3] = -l[3] * m[0] / l[4];
    m[4] = 1.0 / l[4];
    m[6] = (-l[6] * m[0] - l[7] * m[3]) / l[8];
    m[7] = -l[7] * m[4] / l[8];
    m[8] = 1.0 / l[8];
    const double *b = g->b;
    double t0 = m[0] * b[0];
    double t1 = m[3] * b[0] + m[4] * b[1];
    double t2 = m[6] * b[0] + m[7] * b[1] + m[8] * b[2];
    g->c[0] = m[0] * t0 + m[3] * t1 + m[6] * t2;
    g->c[1] = m[4] * t1 + m[7] * t2;
    g->c[2] = m[8] * t2;
}

/* The bilinear value at (px, py), or -1.0 outside the image. */
AT_INLINE double value_for_pixel(const ImView *v, double px, double py) {
    int64_t x1 = ctrunc(floor(px - 0.5));
    int64_t x2 = ctrunc(ceil(px - 0.5));
    double x = px - 0.5 - (double)x1;
    int64_t y1 = ctrunc(floor(py - 0.5));
    int64_t y2 = ctrunc(ceil(py - 0.5));
    double y = py - 0.5 - (double)y1;
    if (x1 < 0 || x2 >= v->w || y1 < 0 || y2 >= v->h) return -1.0;
    const uint8_t *r1 = v->im + y1 * v->s, *r2 = v->im + y2 * v->s;
    return r1[x1] * (1.0 - x) * (1.0 - y) + r1[x2] * x * (1.0 - y) + r2[x1] * (1.0 - x) * y + r2[x2] * x * y;
}

static FamGeom fam_geom(const at_family *fam) {
    return (FamGeom){fam->width_at_border, fam->total_width, fam->nbits, fam->reversed_border, fam->bit_x,
                     fam->bit_y};
}

#define SAMPLE_REJECT 0
#define SAMPLE_OK 1

/* Sample the bits of one quad for one family: SAMPLE_OK with the
   decision margin and the code read, or SAMPLE_REJECT if the border has
   the wrong polarity. `values` is scratch of 2 * total_width^2. */
static int quad_sample(const FamGeom *fam, double decode_sharpening, const ImView *v, const Quad *q, double *values,
                       float *margin_out, uint64_t *rcode_out) {
    int64_t w = v->w, h = v->h;
    int32_t wab = fam->wab;
    float wabf = (float)wab;
    const float patterns[40] = {
        -0.5f, 0.5f, 0.0f, 1.0f, 1.0f,
        0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
        (float)(wab + 0.5), 0.5f, 0.0f, 1.0f, 1.0f,
        (float)(wab - 0.5), 0.5f, 0.0f, 1.0f, 0.0f,
        0.5f, -0.5f, 1.0f, 0.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 0.0f, 0.0f,
        0.5f, (float)(wab + 0.5), 1.0f, 0.0f, 1.0f,
        0.5f, (float)(wab - 0.5), 1.0f, 0.0f, 0.0f,
    };
    GrayModel white, black;
    memset(&white, 0, sizeof white);
    memset(&black, 0, sizeof black);
    const double *hm = q->H;
    for (int pi = 0; pi < 8; pi++) {
        bool is_white = patterns[pi * 5 + 4] != 0.0f;
        for (int32_t i = 0; i < wab; i++) {
            double tagx01 = (double)((patterns[pi * 5] + (float)i * patterns[pi * 5 + 2]) / wabf);
            double tagy01 = (double)((patterns[pi * 5 + 1] + (float)i * patterns[pi * 5 + 3]) / wabf);
            double tagx = 2.0 * (tagx01 - 0.5);
            double tagy = 2.0 * (tagy01 - 0.5);
            double pzz = hm[6] * tagx + hm[7] * tagy + hm[8];
            double px = (hm[0] * tagx + hm[1] * tagy + hm[2]) / pzz;
            double py = (hm[3] * tagx + hm[4] * tagy + hm[5]) / pzz;
            int64_t ix = ctrunc(px);
            int64_t iy = ctrunc(py);
            if (ix < 0 || iy < 0 || ix >= w || iy >= h) continue;
            double val = v->im[iy * v->s + ix];
            graymodel_add(is_white ? &white : &black, tagx, tagy, val);
        }
    }
    if (wab > 1) {
        graymodel_solve(&white);
        graymodel_solve(&black);
    } else {
        graymodel_solve(&white);
        black.c[0] = 0.0;
        black.c[1] = 0.0;
        black.c[2] = black.b[2] / 4.0;
    }
    double wi = white.c[0] * 0.0 + white.c[1] * 0.0 + white.c[2];
    double bi = black.c[0] * 0.0 + black.c[1] * 0.0 + black.c[2];
    if ((wi - bi < 0.0) != fam->reversed_border) return SAMPLE_REJECT;

    int64_t tw = fam->total_width;
    double *vals = values;
    double *sharp = values + tw * tw;
    memset(vals, 0, (size_t)(tw * tw) * sizeof(double));
    int64_t min_coord = (wab - fam->total_width) / 2;
    int64_t nbits = fam->nbits;
    for (int64_t i = 0; i < nbits; i++) {
        int64_t bity = fam->bit_y[i];
        int64_t bitx = fam->bit_x[i];
        double tagx01 = ((double)bitx + 0.5) / wab;
        double tagy01 = ((double)bity + 0.5) / wab;
        double tagx = 2.0 * (tagx01 - 0.5);
        double tagy = 2.0 * (tagy01 - 0.5);
        double zz = hm[6] * tagx + hm[7] * tagy + hm[8];
        double px = (hm[0] * tagx + hm[1] * tagy + hm[2]) / zz;
        double py = (hm[3] * tagx + hm[4] * tagy + hm[5]) / zz;
        double val = value_for_pixel(v, px, py);
        if (val == -1.0) continue;
        double thresh = ((black.c[0] * tagx + black.c[1] * tagy + black.c[2]) +
                         (white.c[0] * tagx + white.c[1] * tagy + white.c[2])) /
                        2.0;
        vals[tw * (bity - min_coord) + (bitx - min_coord)] = val - thresh;
    }

    /* sharpen with the kernel {0,-1,0; -1,4,-1; 0,-1,0}, in the C
       library's tap order (zero taps add nothing) */
    for (int64_t y = 0; y < tw; y++) {
        for (int64_t x = 0; x < tw; x++) {
            double acc = 0.0;
            if (y > 0) acc += -vals[(y - 1) * tw + x];
            if (x > 0) acc += -vals[y * tw + x - 1];
            acc += vals[y * tw + x] * 4.0;
            if (x + 1 < tw) acc += -vals[y * tw + x + 1];
            if (y + 1 < tw) acc += -vals[(y + 1) * tw + x];
            sharp[y * tw + x] = acc;
        }
    }
    for (int64_t i = 0; i < tw * tw; i++) vals[i] = vals[i] + decode_sharpening * sharp[i];

    float black_score = 0.0f, white_score = 0.0f, black_count = 1.0f, white_count = 1.0f;
    uint64_t rcode = 0;
    for (int64_t i = 0; i < nbits; i++) {
        int64_t bity = fam->bit_y[i];
        int64_t bitx = fam->bit_x[i];
        rcode = rcode << 1;
        double vv = vals[(bity - min_coord) * tw + (bitx - min_coord)];
        if (vv > 0.0) {
            white_score = (float)(white_score + vv);
            white_count += 1.0f;
            rcode |= 1;
        } else {
            black_score = (float)(black_score - vv);
            black_count += 1.0f;
        }
    }
    /* C's fmin of two floats widened to double */
    double x = (double)(white_score / white_count);
    double y = (double)(black_score / black_count);
    double m = x != x ? y : y != y ? x : x < y ? x : y;
    *margin_out = (float)m;
    *rcode_out = rcode;
    return SAMPLE_OK;
}

/* ---------------------------------------------------------------------
   Edge refinement: move each edge to the strongest gradient along its
   normal, sampled in the full-resolution image. */

AT_INLINE double bilinear1(const ImView *v, int64_t xi, int64_t yi, double a, double b) {
    const uint8_t *p = v->im + yi * v->s + xi;
    return (1.0 - a) * (1.0 - b) * p[0] + a * (1.0 - b) * p[1] + (1.0 - a) * b * p[v->s] + a * b * p[v->s + 1];
}

/* Both sample points inside the image: (int)x in [0, w-1) for both
   coordinates, without branches. */
AT_INLINE bool in_box(double x1, double y1, double x2, double y2, double wl, double hl) {
    int m = (x1 > -1.0) & (x1 < wl) & (y1 > -1.0) & (y1 < hl) & (x2 > -1.0) & (x2 < wl) & (y2 > -1.0) & (y2 < hl);
    return m != 0;
}

/* trunc(t) clamped to [0, tmax], NaN giving 0, as an integer: 2^52
   added to an integral double in [0, 2^31) puts it in the low bits. */
AT_INLINE int64_t index_of(double t, double tmax) {
    double c = t >= 0.0 ? t : 0.0;
    double c2 = c <= tmax ? c : tmax;
    return (int64_t)(bits_of_f64(c2 + 4503599627370496.0) & 0xffffffffu);
}

/* An integral double in [0, 2^31) as an integer. */
AT_INLINE int64_t low_int(double t) { return (int64_t)(bits_of_f64(t + 4503599627370496.0) & 0xffffffffu); }

AT_INLINE uint16_t ld2(const uint8_t *p) {
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}

/* The weight of one step: (g2 - g1)^2, or 0.0 where a sample point is
   outside the image or g1 < g2. Arithmetic as in the C library's AVX2
   lanes. */
AT_INLINE double lane_weight(const uint8_t *im, int64_t s, double x0, double y0, double nx, double ny, double n,
                             double wl, double hl, double wmax, double hmax) {
    double x1 = x0 + (n + 1.0) * nx - 0.5;
    double y1 = y0 + (n + 1.0) * ny - 0.5;
    double x2 = x0 + (n - 1.0) * nx - 0.5;
    double y2 = y0 + (n - 1.0) * ny - 0.5;
    bool ok = in_box(x1, y1, x2, y2, wl, hl);
    double tx1 = trunc(x1), ty1 = trunc(y1), tx2 = trunc(x2), ty2 = trunc(y2);
    const uint8_t *p = im + index_of(ty1, hmax) * s + index_of(tx1, wmax);
    const uint8_t *q = im + index_of(ty2, hmax) * s + index_of(tx2, wmax);
    double a1 = x1 - tx1, b1 = y1 - ty1, a2 = x2 - tx2, b2 = y2 - ty2;
    double na1 = 1.0 - a1, nb1 = 1.0 - b1, na2 = 1.0 - a2, nb2 = 1.0 - b2;
    double g1 = ((na1 * nb1) * p[0] + (a1 * nb1) * p[1]) + ((na1 * b1) * p[s] + (a1 * b1) * p[s + 1]);
    double g2 = ((na2 * nb2) * q[0] + (a2 * nb2) * q[1]) + ((na2 * b2) * q[s] + (a2 * b2) * q[s + 1]);
    double d = g2 - g1;
    return ok && !(g1 < g2) ? d * d : 0.0;
}

/* The weights of steps 0..n-1 of one sample point, all inside the
   image, into fbuf[0..n]. fbuf[n..2n] holds the steps' n, and the last m
   entries the point offsets rf - 1 + 0.25 * j (refine_tables); the rest
   of fbuf, ibuf and pbuf is scratch.

   The second point of step k, x0 + (n - 1) * nx - 0.5, is the first
   point of step k - 8: n - 1 and (n - 2) + 1 are the same multiple of
   0.25, exactly. So the bilinear values of the points
   x0 + (rf - 1 + 0.25 * j) * nx - 0.5, j in 0..n+8, give g1 (j = k + 8)
   and g2 (j = k) of every step: n + 8 values instead of 2n. */
/* The loops of refine_batch, each with restrict pointers: its arrays are
   all in fbuf, so without them clang needs runtime alias checks, which
   it does not make at -Os (and they cost time at -O2). */
AT_INLINE void rb_points(const double *restrict mjt, double x0, double y0, double nx, double ny, int64_t s,
                         int64_t m, double *restrict fa, double *restrict bs, int64_t *restrict idx) {
    AT_VFOR(j, 0, m, {
        double mj = mjt[j];
        double x = x0 + mj * nx - 0.5;
        double y = y0 + mj * ny - 0.5;
        double tx = trunc(x), ty = trunc(y);
        fa[j] = x - tx;
        bs[j] = y - ty;
        idx[j] = low_int(ty) * s + low_int(tx);
    });
}

AT_INLINE void rb_values(const double *restrict fa, const double *restrict bs, const uint32_t *restrict pk, int64_t m,
                         double *restrict g) {
    AT_VFOR(j, 0, m, {
        double a = fa[j], b = bs[j];
        double na = 1.0 - a, nb = 1.0 - b;
        uint32_t p = pk[j];
        double p00 = (double)(p & 255), p01 = (double)((p >> 8) & 255);
        double p10 = (double)((p >> 16) & 255), p11 = (double)(p >> 24);
        g[j] = ((na * nb) * p00 + (a * nb) * p01) + ((na * b) * p10 + (a * b) * p11);
    });
}

AT_INLINE void rb_weights(const double *restrict g1s, const double *restrict g2s, int64_t n, double *restrict wgt) {
    AT_VFOR(k, 0, n, {
        double g1 = g1s[k], g2 = g2s[k];
        double d = g2 - g1;
        wgt[k] = !(g1 < g2) ? d * d : 0.0;
    });
}

AT_INLINE void refine_batch(const uint8_t *restrict im, int64_t s, double x0, double y0, double nx, double ny,
                            int64_t n, double *restrict fbuf, int64_t *restrict ibuf, uint32_t *restrict pbuf) {
    int64_t m = n + 8;
    double *wgt = fbuf;
    int64_t o = 2 * n;
    double *fa = fbuf + o, *bs = fbuf + o + m, *g = fbuf + o + 2 * m;
    const double *mjt = fbuf + o + 3 * m;
    int64_t *idx = ibuf;
    uint32_t *pk = pbuf;
    rb_points(mjt, x0, y0, nx, ny, s, m, fa, bs, idx);
    /* the pixels of a sample in one u32: p00, p01, p10 and p11 in bytes
       0 to 3 */
    /* scalar: clang vectorizes it with an insert per load, which is
       slower than plain loads (refine_lanes +20% at AVX-512) */
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#elif defined(__GNUC__) && __GNUC__ >= 14
#pragma GCC novector
#endif
    for (int64_t j = 0; j < m; j++) pk[j] = ld2(im + idx[j]) | ((uint32_t)ld2(im + idx[j] + s) << 16);
    rb_values(fa, bs, pk, m, g);
    /* g1 of step k is point k + 8, g2 is point k */
    rb_weights(g + 8, g, n, wgt);
}

/* The weighted sums of refine_edges over the first vsteps steps (a
   multiple of 4) of one sample point, 4 steps at a time as the C
   library's AVX2 code does: lane l takes the steps congruent to l mod 4.
   Out-of-bounds lanes and lanes with g1 < g2 add 0.0, which leaves the
   sums as skipping them would (the sums are never -0.0). (int)x is in
   [0, k) exactly when -1 < x < k, and then x - (int)x is x - trunc(x). */
SIMD_VOID(refine_lanes,
          (const uint8_t *restrict im, int64_t s, int64_t w, int64_t h, double x0, double y0, double nx, double ny,
           int64_t range, int64_t vsteps, double *restrict fbuf, int64_t *restrict ibuf, uint32_t *restrict pbuf,
           double *restrict out),
          (im, s, w, h, x0, y0, nx, ny, range, vsteps, fbuf, ibuf, pbuf, out)) {
    double wl = (double)(w - 1), hl = (double)(h - 1);
    double wmax = (double)(w - 2), hmax = (double)(h - 2);
    double rf = (double)-range;
    double mn0 = 0.0, mn1 = 0.0, mn2 = 0.0, mn3 = 0.0, mc0 = 0.0, mc1 = 0.0, mc2 = 0.0, mc3 = 0.0;
    int64_t step = 0;
    /* x1, y1, x2 and y2 are monotonic in the step (each operation
       rounds monotonically), so when the first and the last step are
       inside the image, all steps are: no per-step tests or clamps */
    double nf = rf + 0.25 * (double)(vsteps - 1);
    if (vsteps > 0 &&
        in_box(x0 + (rf + 1.0) * nx - 0.5, y0 + (rf + 1.0) * ny - 0.5, x0 + (rf - 1.0) * nx - 0.5,
               y0 + (rf - 1.0) * ny - 0.5, wl, hl) &&
        in_box(x0 + (nf + 1.0) * nx - 0.5, y0 + (nf + 1.0) * ny - 0.5, x0 + (nf - 1.0) * nx - 0.5,
               y0 + (nf - 1.0) * ny - 0.5, wl, hl)) {
        refine_batch(im, s, x0, y0, nx, ny, vsteps, fbuf, ibuf, pbuf);
        /* the four lanes as one vector (clang does not join the scalar
           sums): the same operations in each lane */
        typedef double v4d __attribute__((vector_size(32), aligned(8)));
        const double *wgt = fbuf, *ns = fbuf + vsteps;
        v4d vmn = {0.0, 0.0, 0.0, 0.0}, vmc = {0.0, 0.0, 0.0, 0.0};
        for (int64_t c = 0; c + 4 <= vsteps; c += 4) {
            v4d wv, nv;
            memcpy(&wv, wgt + c, sizeof wv);
            memcpy(&nv, ns + c, sizeof nv);
            vmn += wv * nv;
            vmc += wv;
        }
        mn0 = vmn[0];
        mn1 = vmn[1];
        mn2 = vmn[2];
        mn3 = vmn[3];
        mc0 = vmc[0];
        mc1 = vmc[1];
        mc2 = vmc[2];
        mc3 = vmc[3];
        step = vsteps;
    }
    for (; step < vsteps; step += 4) {
        double n0 = rf + 0.25 * (double)step;
        double n1 = rf + 0.25 * (double)(step + 1);
        double n2 = rf + 0.25 * (double)(step + 2);
        double n3 = rf + 0.25 * (double)(step + 3);
        double w0 = lane_weight(im, s, x0, y0, nx, ny, n0, wl, hl, wmax, hmax);
        double w1 = lane_weight(im, s, x0, y0, nx, ny, n1, wl, hl, wmax, hmax);
        double w2 = lane_weight(im, s, x0, y0, nx, ny, n2, wl, hl, wmax, hmax);
        double w3 = lane_weight(im, s, x0, y0, nx, ny, n3, wl, hl, wmax, hmax);
        mn0 += w0 * n0;
        mn1 += w1 * n1;
        mn2 += w2 * n2;
        mn3 += w3 * n3;
        mc0 += w0;
        mc1 += w1;
        mc2 += w2;
        mc3 += w3;
    }
    out[0] = (mn0 + mn2) + (mn1 + mn3);
    out[1] = (mc0 + mc2) + (mc1 + mc3);
}

/* The tables of refine_batch in fbuf (n steps, from rf): each step's n,
   and the offset of each sample point. They depend only on the range. */
static void refine_tables(double *fbuf, int64_t n, double rf) {
    int64_t m = n + 8;
    double *ns = fbuf + n;
    AT_VFOR(k, 0, n, ns[k] = rf + 0.25 * (double)k;);
    double rm = rf - 1.0;
    double *mjt = fbuf + 2 * n + 3 * m;
    AT_VFOR(j, 0, m, mjt[j] = rm + 0.25 * (double)j;);
}

#define PREFETCH_AHEAD 2

/* Prefetch the pixels along the search line of one sample point: steps
   -range-1 to range+1, two pixels apart, each with the row below (which
   covers the rows between the steps). */
static void prefetch_line(const ImView *v, double x0, double y0, double nx, double ny, int64_t range) {
    int64_t n = v->h * v->s;
    for (int64_t k = -range - 1; k <= range + 1; k += 2) {
        double x = x0 + (double)k * nx - 0.5;
        double y = y0 + (double)k * ny - 0.5;
        if (!(x >= 0.0 && y >= 0.0 && x < 1.0e9 && y < 1.0e9)) continue;
        /* rounded, not truncated: one pixel off does not matter here */
        int64_t i = low_int(y) * v->s + low_int(x);
        if ((uint64_t)i < (uint64_t)n) __builtin_prefetch(v->im + i);
        if ((uint64_t)(i + v->s) < (uint64_t)n) __builtin_prefetch(v->im + i + v->s);
    }
}

static void refine_edges(at_ctx *cx, float quad_decimate, const ImView *v, Quad *q) {
    int64_t w = v->w, h = v->h;
    double lines[4][4];
    int64_t range = (int64_t)(quad_decimate + 1.0f);
    const double step_length = 0.25;
    int64_t max_steps = 2 * 4 * range + 1;
    const double delta = 0.5;
    int64_t vsteps = imax(max_steps - max_steps % 4, 0);
    at_mark mk = stk_enter(cx);
    double *fbuf = STK_ALLOC(cx, double, 2 * vsteps + 4 * (vsteps + 8));
    refine_tables(fbuf, vsteps, (double)-range);
    int64_t *ibuf = STK_ALLOC(cx, int64_t, vsteps + 8);
    uint32_t *pbuf = STK_ALLOC(cx, uint32_t, vsteps + 8);
    for (int edge = 0; edge < 4; edge++) {
        int a = edge, b = (edge + 1) & 3;
        double nx = (double)(q->p[b][1] - q->p[a][1]);
        double ny = (double)(-q->p[b][0] + q->p[a][0]);
        double mag = at_sqrt(nx * nx + ny * ny);
        nx /= mag;
        ny /= mag;
        if (q->reversed_border) {
            nx = -nx;
            ny = -ny;
        }
        int64_t nsamples = imax(16, ctrunc(mag / 8.0));
        double mx = 0.0, my = 0.0, mxx = 0.0, mxy = 0.0, myy = 0.0, nn = 0.0;
        for (int64_t si = 0; si < nsamples; si++) {
            double alpha = (1.0 + (double)si) / (double)(nsamples + 1);
            double x0 = alpha * q->p[a][0] + (1.0 - alpha) * q->p[b][0];
            double y0 = alpha * q->p[a][1] + (1.0 - alpha) * q->p[b][1];
            /* the pixels of a later sample point, to hide cache misses */
            double pa = (1.0 + (double)(si + PREFETCH_AHEAD)) / (double)(nsamples + 1);
            prefetch_line(v, pa * q->p[a][0] + (1.0 - pa) * q->p[b][0], pa * q->p[a][1] + (1.0 - pa) * q->p[b][1],
                          nx, ny, range);
            double lsc[2];
            refine_lanes(v->im, v->s, w, h, x0, y0, nx, ny, range, vsteps, fbuf, ibuf, pbuf, lsc);
            double mn = 0.0 + lsc[0];
            double mcount = 0.0 + lsc[1];
            for (int64_t step = vsteps; step < max_steps;) {
                double n = (double)-range + step_length * (double)step;
                step++;
                double x1 = x0 + (n + 1.0) * nx - delta;
                double y1 = y0 + (n + 1.0) * ny - delta;
                int64_t x1i = ctrunc(x1), y1i = ctrunc(y1);
                if (x1i < 0 || x1i + 1 >= w || y1i < 0 || y1i + 1 >= h) continue;
                double x2 = x0 + (n - 1.0) * nx - delta;
                double y2 = y0 + (n - 1.0) * ny - delta;
                int64_t x2i = ctrunc(x2), y2i = ctrunc(y2);
                if (x2i < 0 || x2i + 1 >= w || y2i < 0 || y2i + 1 >= h) continue;
                double g1 = bilinear1(v, x1i, y1i, x1 - (double)x1i, y1 - (double)y1i);
                double g2 = bilinear1(v, x2i, y2i, x2 - (double)x2i, y2 - (double)y2i);
                if (g1 < g2) continue;
                double weight = (g2 - g1) * (g2 - g1);
                mn += weight * n;
                mcount += weight;
            }
            if (mcount == 0.0) continue;
            double n0 = mn / mcount;
            double bestx = x0 + n0 * nx;
            double besty = y0 + n0 * ny;
            mx += bestx;
            my += besty;
            mxx += bestx * bestx;
            mxy += bestx * besty;
            myy += besty * besty;
            nn += 1.0;
        }
        double ex = mx / nn, ey = my / nn;
        double cxx = mxx / nn - ex * ex;
        double cxy = mxy / nn - ex * ey;
        double cyy = myy / nn - ey * ey;
        double normal_theta = 0.5 * (double)atan2f((float)(-2.0 * cxy), (float)(cyy - cxx));
        lines[edge][0] = ex;
        lines[edge][1] = ey;
        lines[edge][2] = cosf((float)normal_theta);
        lines[edge][3] = sinf((float)normal_theta);
    }
    stk_leave(cx, mk);
    for (int i = 0; i < 4; i++) {
        int j = (i + 1) & 3;
        double a00 = lines[i][3], a01 = -lines[j][3], a10 = -lines[i][2], a11 = lines[j][2];
        double b0 = -lines[i][0] + lines[j][0];
        double b1 = -lines[i][1] + lines[j][1];
        double det = a00 * a11 - a10 * a01;
        if (fabs(det) > 0.001) {
            double w00 = a11 / det;
            double w01 = -a01 / det;
            double l0 = w00 * b0 + w01 * b1;
            q->p[j][0] = (float)(lines[i][0] + l0 * a00);
            q->p[j][1] = (float)(lines[i][1] + l0 * a10);
        }
    }
}

/* ---------------------------------------------------------------------
   Overlap test of two quadrilaterals (g2d_polygon_overlaps_polygon). */

typedef struct {
    double p[2], u[2];
} G2dLine;

AT_INLINE G2dLine g2d_line(const double p0[2], const double p1[2]) {
    double u0 = p1[0] - p0[0], u1 = p1[1] - p0[1];
    double mag = (double)at_sqrtf((float)(u0 * u0 + u1 * u1));
    return (G2dLine){{p0[0], p0[1]}, {u0 / mag, u1 / mag}};
}

AT_INLINE double g2d_coord(const G2dLine *l, const double q[2]) {
    return (q[0] - l->p[0]) * l->u[0] + (q[1] - l->p[1]) * l->u[1];
}

static bool g2d_intersect(const G2dLine *la, const G2dLine *lb, double out[2]) {
    double m00 = la->u[0], m01 = -lb->u[0], m10 = la->u[1], m11 = -lb->u[1];
    double det = m00 * m11 - m01 * m10;
    if (fabs(det) < 0.00000001) return false;
    double i00 = m11 / det;
    double i01 = -m01 / det;
    double b00 = lb->p[0] - la->p[0];
    double b10 = lb->p[1] - la->p[1];
    double x00 = i00 * b00 + i01 * b10;
    out[0] = la->u[0] * x00 + la->p[0];
    out[1] = la->u[1] * x00 + la->p[1];
    return true;
}

static bool g2d_segments_intersect(const double a0[2], const double a1[2], const double b0[2], const double b1[2]) {
    G2dLine la = g2d_line(a0, a1), lb = g2d_line(b0, b1);
    double t[2];
    if (!g2d_intersect(&la, &lb, t)) return false;
    double a = g2d_coord(&la, la.p), b = g2d_coord(&la, a1), c = g2d_coord(&la, t);
    if ((c < a && c < b) || (c > a && c > b)) return false;
    a = g2d_coord(&lb, lb.p);
    b = g2d_coord(&lb, b1);
    c = g2d_coord(&lb, t);
    if ((c < a && c < b) || (c > a && c > b)) return false;
    return true;
}

static bool g2d_contains_point(const double poly[4][2], const double q[2]) {
    int last_quadrant = 0, quad_acc = 0;
    for (int i = 0; i < 5; i++) {
        const double *p = poly[i % 4];
        int quadrant = p[0] < q[0] ? (p[1] < q[1] ? 2 : 1) : (p[1] < q[1] ? 3 : 0);
        if (i > 0) {
            int dq = quadrant - last_quadrant;
            if (dq == -3 || dq == 1) {
                quad_acc++;
            } else if (dq == -1 || dq == 3) {
                quad_acc--;
            } else if (dq == -2 || dq == 2) {
                const double *p0 = poly[i - 1];
                double nx = p[1] - q[1];
                double ny = -p[0] + q[0];
                double dot = nx * (p0[0] - q[0]) + ny * (p0[1] - q[1]);
                if (dot < 0.0)
                    quad_acc -= 2;
                else
                    quad_acc += 2;
            }
        }
        last_quadrant = quadrant;
    }
    return quad_acc >= 2 || quad_acc <= -2;
}

static bool g2d_overlaps(const double pa[4][2], const double pb[4][2]) {
    for (int ia = 0; ia < 4; ia++)
        for (int ib = 0; ib < 4; ib++)
            if (g2d_segments_intersect(pa[ia], pa[(ia + 1) % 4], pb[ib], pb[(ib + 1) % 4])) return true;
    const double qb[2] = {(pb[0][0] + pb[1][0] + pb[2][0]) / 3.0, (pb[0][1] + pb[1][1] + pb[2][1]) / 3.0};
    if (g2d_contains_point(pa, qb)) return true;
    const double qa[2] = {(pa[0][0] + pa[1][0] + pa[2][0]) / 3.0, (pa[0][1] + pa[1][1] + pa[2][1]) / 3.0};
    if (g2d_contains_point(pb, qa)) return true;
    return false;
}

AT_INLINE int prefer_smaller(int pref, double q0, double q1) {
    if (pref != 0) return pref;
    if (q0 < q1) return -1;
    if (q1 < q0) return 1;
    return 0;
}

/* Remove detection i of n, keeping the order. */
static void remove_at(at_detection *d, int64_t *n, int64_t i) {
    memmove(d + i, d + i + 1, (size_t)(*n - i - 1) * sizeof *d);
    (*n)--;
}

/* Remove overlapping detections of the same tag, keeping the better
   one. */
static void reconcile(at_detection *dets, int64_t *ndets) {
    int64_t i0 = 0;
    while (i0 < *ndets) {
        bool removed0 = false;
        int64_t i1 = i0 + 1;
        while (i1 < *ndets) {
            const at_detection *d0 = &dets[i0], *d1 = &dets[i1];
            if (d0->id != d1->id || d0->family != d1->family) {
                i1++;
                continue;
            }
            if (g2d_overlaps(d0->p, d1->p)) {
                int pref = 0;
                pref = prefer_smaller(pref, d0->hamming, d1->hamming);
                pref = prefer_smaller(pref, -(double)d0->decision_margin, -(double)d1->decision_margin);
                for (int i = 0; i < 4; i++) {
                    pref = prefer_smaller(pref, d0->p[i][0], d1->p[i][0]);
                    pref = prefer_smaller(pref, d0->p[i][1], d1->p[i][1]);
                }
                if (pref < 0) {
                    remove_at(dets, ndets, i1);
                    continue;
                }
                remove_at(dets, ndets, i0);
                removed0 = true;
                break;
            }
            i1++;
        }
        if (!removed0) i0++;
    }
}
