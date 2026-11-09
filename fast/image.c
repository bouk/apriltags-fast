/* Image preprocessing: decimation, gaussian blur, sharpening, and PNM
   files. The arithmetic mirrors the C library's AVX2 build bit for bit. */
#include "common.h"

/* The size of the image that decimate_rows makes. */
static void decimated_size(int64_t width, int64_t height, float ffactor, int64_t *ow, int64_t *oh) {
    if (ffactor == 1.5f) {
        *ow = width / 3 * 2;
        *oh = height / 3 * 2;
        return;
    }
    int64_t factor = (int64_t)ffactor;
    if (factor == 2) {
        *ow = width / 2;
        *oh = height / 2;
        return;
    }
    *ow = 1 + (width - 1) / factor;
    *oh = 1 + (height - 1) / factor;
}

/* One output row of the 2x2 box filter. The C library's AVX2 path rounds
   twice (vertical average, then the pair average) for the part of the
   row that is a multiple of 32 pixels wide, and computes the exact
   rounded mean for the rest. r0 and r1 have 2n pixels. */
SIMD_VOID(decimate2_row, (const uint8_t *restrict r0, const uint8_t *restrict r1, uint8_t *restrict o, int64_t n),
          (r0, r1, o, n)) {
    int64_t nv = n - n % 32;
    int64_t sx0 = 0;
#if defined(__ARM_NEON)
    /* vrhadd is the rounded average of the AVX2 path */
    for (; sx0 < nv; sx0 += 16) {
        uint8x16x2_t a = vld2q_u8(r0 + 2 * sx0), b = vld2q_u8(r1 + 2 * sx0);
        vst1q_u8(o + sx0, vrhaddq_u8(vrhaddq_u8(a.val[0], b.val[0]), vrhaddq_u8(a.val[1], b.val[1])));
    }
#endif
    for (int64_t sx = sx0; sx < nv; sx++) {
        uint16_t v0 = (uint16_t)((r0[2 * sx] + r1[2 * sx] + 1) >> 1);
        uint16_t v1 = (uint16_t)((r0[2 * sx + 1] + r1[2 * sx + 1] + 1) >> 1);
        o[sx] = (uint8_t)((v0 + v1 + 1) >> 1);
    }
    for (int64_t sx = nv; sx < n; sx++) {
        uint16_t s = (uint16_t)(r0[2 * sx] + r0[2 * sx + 1] + r1[2 * sx] + r1[2 * sx + 1] + 2);
        o[sx] = (uint8_t)(s >> 2);
    }
}

/* Decimate src (rows stride bytes apart) into out (rows ow bytes apart),
   rows [sy0, sy1) of the output. The factor is 1.5, 2 (2x2 box filter),
   or an integer (point sampling). */
static void decimate_rows(const uint8_t *src, int64_t stride, float ffactor, uint8_t *out, int64_t ow, int64_t sy0,
                          int64_t sy1) {
    if (ffactor == 1.5f) {
        for (int64_t sy = sy0; sy < sy1; sy += 2) {
            int64_t y = sy / 2 * 3;
            const uint8_t *s0 = src + y * stride, *s1 = s0 + stride, *s2 = s1 + stride;
            uint8_t *o0 = out + sy * ow, *o1 = o0 + ow;
            for (int64_t sx = 0, x = 0; sx < ow; sx += 2, x += 3) {
                int32_t a = s0[x], b = s0[x + 1], c = s0[x + 2];
                int32_t d = s1[x], e = s1[x + 1], f = s1[x + 2];
                int32_t g = s2[x], h = s2[x + 1], i = s2[x + 2];
                o0[sx] = (uint8_t)((4 * a + 2 * b + 2 * d + e) / 9);
                o0[sx + 1] = (uint8_t)((4 * c + 2 * b + 2 * f + e) / 9);
                o1[sx] = (uint8_t)((4 * g + 2 * d + 2 * h + e) / 9);
                o1[sx + 1] = (uint8_t)((4 * i + 2 * f + 2 * h + e) / 9);
            }
        }
        return;
    }
    int64_t factor = (int64_t)ffactor;
    if (factor == 2) {
        for (int64_t sy = sy0; sy < sy1; sy++)
            decimate2_row(src + 2 * sy * stride, src + (2 * sy + 1) * stride, out + sy * ow, ow);
        return;
    }
    for (int64_t sy = sy0; sy < sy1; sy++) {
        const uint8_t *row = src + sy * factor * stride;
        uint8_t *o = out + sy * ow;
        for (int64_t sx = 0, x = 0; sx < ow; sx++, x += factor) o[sx] = row[x];
    }
}

/* The 1D convolution of the C library. `parallel` selects the variant of
   image_u8_parallel.c (used for images of 65536 pixels or more), which
   convolves one more output pixel than the single-threaded image_u8.c. */
static void convolve(const uint8_t *x, uint8_t *y, int64_t sz, const uint8_t *k, int64_t ksz, bool parallel) {
    int64_t i = 0;
    while (i < ksz / 2 && i < sz) {
        y[i] = x[i];
        i++;
    }
    int64_t n = parallel ? sz - ksz + 1 : sz - ksz;
    for (i = 0; i < n; i++) {
        uint32_t acc = 0;
        for (int64_t j = 0; j < ksz; j++) acc += (uint32_t)k[j] * x[i + j];
        y[ksz / 2 + i] = (uint8_t)(acc >> 8);
    }
    int64_t tail = parallel ? sz - ksz / 2 : sz - ksz + ksz / 2;
    for (i = imax(tail, 0); i < sz; i++) y[i] = x[i];
}

/* Gaussian blur in place, as image_u8_gaussian_blur_parallel. */
static void gaussian_blur(at_ctx *cx, uint8_t *im, int64_t w, int64_t h, double sigma, int64_t ksz) {
    if (sigma == 0.0) return;
    at_mark mk = stk_enter(cx);
    double *dk = STK_ALLOC(cx, double, ksz);
    for (int64_t i = 0; i < ksz; i++) {
        int64_t x = -(ksz / 2) + i;
        double t = (double)x / sigma;
        dk[i] = exp(-0.5 * (t * t));
    }
    double acc = 0.0;
    for (int64_t i = 0; i < ksz; i++) acc += dk[i];
    uint8_t *k = STK_ALLOC(cx, uint8_t, ksz);
    for (int64_t i = 0; i < ksz; i++) k[i] = (uint8_t)(dk[i] / acc * 255.0);

    bool parallel = w * h >= 65536;
    uint8_t *tmp = STK_ALLOC(cx, uint8_t, imax(w, h));
    uint8_t *tmp2 = STK_ALLOC(cx, uint8_t, h);
    for (int64_t y = 0; y < h; y++) {
        uint8_t *row = im + y * w;
        memcpy(tmp, row, (size_t)w);
        convolve(tmp, row, w, k, ksz, parallel);
    }
    for (int64_t x = 0; x < w; x++) {
        for (int64_t y = 0; y < h; y++) tmp[y] = im[y * w + x];
        convolve(tmp, tmp2, h, k, ksz, parallel);
        for (int64_t y = 0; y < h; y++) im[y * w + x] = tmp2[y];
    }
    stk_leave(cx, mk);
}

/* The kernel size of blur_or_sharpen; 1 or less does nothing. */
static int64_t blur_kernel_size(float quad_sigma) {
    if (quad_sigma == 0.0f) return 0;
    int64_t ksz = (int64_t)(4.0f * fabsf(quad_sigma));
    if ((ksz & 1) == 0) ksz++;
    return ksz;
}

SIMD_VOID(sharpen_with, (uint8_t *restrict im, const uint8_t *restrict orig, int64_t n), (im, orig, n)) {
    for (int64_t i = 0; i < n; i++) {
        int32_t v = 2 * (int32_t)orig[i] - im[i];
        im[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
}

/* Blur (sigma > 0) or sharpen (sigma < 0) as the detector's quad_sigma
   does. Returns false if the kernel is too small to change anything. */
static bool blur_or_sharpen(at_ctx *cx, uint8_t *im, int64_t w, int64_t h, float quad_sigma) {
    int64_t ksz = blur_kernel_size(quad_sigma);
    if (ksz <= 1) return false;
    double sigma = fabsf(quad_sigma);
    if (quad_sigma > 0.0f) {
        gaussian_blur(cx, im, w, h, sigma, ksz);
        return true;
    }
    at_mark mk = stk_enter(cx);
    uint8_t *orig = STK_ALLOC(cx, uint8_t, w * h);
    memcpy(orig, im, (size_t)(w * h));
    gaussian_blur(cx, im, w, h, sigma, ksz);
    sharpen_with(im, orig, w * h);
    stk_leave(cx, mk);
    return true;
}

/* ---------------------------------------------------------------------
   PNM files. */

static void skip_pnm_space(const uint8_t *data, int64_t len, int64_t *at) {
    while (*at < len) {
        uint8_t c = data[*at];
        if (c == '#') {
            while (*at < len && data[*at] != '\n') (*at)++;
        } else if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
            (*at)++;
        } else {
            return;
        }
    }
}

static int64_t read_pnm_int(const uint8_t *data, int64_t len, int64_t *at) {
    skip_pnm_space(data, len, at);
    int64_t v = 0;
    while (*at < len && data[*at] >= '0' && data[*at] <= '9') {
        v = v * 10 + (data[*at] - '0');
        (*at)++;
    }
    return v;
}

/* Decode a binary PGM (P5) or PPM (P6) image into *pixels, which holds
   *cap bytes (64-byte aligned, free()); a larger one replaces it. Color
   is converted to gray as the C library does ((r + g + g + b) / 4). */
static bool decode_pnm(const uint8_t *data, int64_t len, int64_t *width, int64_t *height, uint8_t **pixels,
                       int64_t *cap) {
    if (len < 2 || data[0] != 'P') return false;
    uint8_t kind = data[1];
    if (kind != '5' && kind != '6') return false;
    int64_t at = 2;
    int64_t w = read_pnm_int(data, len, &at);
    int64_t h = read_pnm_int(data, len, &at);
    int64_t maxval = read_pnm_int(data, len, &at);
    at++;
    if (w <= 0 || h <= 0 || maxval != 255) return false;
    int64_t ch = kind == '5' ? 1 : 3;
    if (len - at < w * h * ch) return false;
    if (!*pixels || *cap < w * h) {
        uint8_t *px = (uint8_t *)aligned_alloc(64, (size_t)(w * h + 63) / 64 * 64);
        if (!px) return false;
        free(*pixels);
        *pixels = px;
        *cap = w * h;
    }
    uint8_t *px = *pixels;
    if (ch == 1) {
        memcpy(px, data + at, (size_t)(w * h));
    } else {
        for (int64_t i = 0; i < w * h; i++) {
            int32_t r = data[at + 3 * i], g = data[at + 3 * i + 1], b = data[at + 3 * i + 2];
            px[i] = (uint8_t)((r + g + g + b) / 4);
        }
    }
    *width = w;
    *height = h;
    return true;
}

bool at_read_pnm_into(const char *path, int64_t *width, int64_t *height, uint8_t **pixels, int64_t *cap) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    bool ok = false;
    uint8_t *data = NULL;
    if (fseek(f, 0, SEEK_END) == 0) {
        long len = ftell(f);
        if (len > 0 && fseek(f, 0, SEEK_SET) == 0 && (data = (uint8_t *)malloc((size_t)len)) &&
            fread(data, 1, (size_t)len, f) == (size_t)len)
            ok = decode_pnm(data, len, width, height, pixels, cap);
    }
    free(data);
    fclose(f);
    return ok;
}

bool at_read_pnm(const char *path, int64_t *width, int64_t *height, uint8_t **pixels) {
    int64_t cap = 0;
    *pixels = NULL;
    return at_read_pnm_into(path, width, height, pixels, &cap);
}

bool at_write_pgm(const char *path, const uint8_t *pixels, int64_t width, int64_t height) {
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    fprintf(f, "P5\n%lld %lld\n255\n", (long long)width, (long long)height);
    bool ok = fwrite(pixels, 1, (size_t)(width * height), f) == (size_t)(width * height);
    return fclose(f) == 0 && ok;
}
