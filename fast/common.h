/* Internal declarations of the apriltag C port. All .c files of src/ but
   families.c are compiled as one unit (apriltag_all.c), so that the
   compiler can inline across them. */
#ifndef AT_COMMON_H
#define AT_COMMON_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "apriltag_fast.h"

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

#define AT_INLINE static inline __attribute__((always_inline))
#define AT_NOINLINE __attribute__((noinline))
/* AT_VFOR(i, lo, hi, body): for (int64_t i = lo; i < hi; i++) body, for
   the loops that clang vectorizes. Xcode builds Swift package C code with
   -Os in Release. There clang vectorizes a loop only if it needs no
   scalar remainder loop, so these loops stayed scalar (the detector was
   about 1.5x slower than at -O2). At -Os the loop is split into a part of
   a multiple of 16 iterations, which vectorizes, and a scalar remainder.
   At other levels it is the plain loop, so -O2 code does not change. */
#if defined(__clang__) && defined(__OPTIMIZE_SIZE__)
#define AT_VFOR(i, lo, hi, ...)                                                    \
    do {                                                                          \
        int64_t at_lo_ = (lo), at_n_ = (hi) - at_lo_;                             \
        int64_t at_nv_ = at_n_ > 0 ? at_n_ & ~(int64_t)15 : 0;                    \
        for (int64_t at_k_ = 0; at_k_ < at_nv_; at_k_++) {                        \
            int64_t i = at_lo_ + at_k_;                                           \
            __VA_ARGS__                                                           \
        }                                                                         \
        _Pragma("clang loop vectorize(disable) unroll(disable)") for (int64_t i = \
                                                                          at_lo_ + at_nv_; \
                                                                      i < at_lo_ + at_n_; i++) { \
            __VA_ARGS__                                                           \
        }                                                                         \
    } while (0)
#else
#define AT_VFOR(i, lo, hi, ...)                \
    do {                                      \
        for (int64_t i = (lo); i < (hi); i++) { \
            __VA_ARGS__                       \
        }                                     \
    } while (0)
#endif
#define AT_LIKELY(x) __builtin_expect(!!(x), 1)
#define AT_UNLIKELY(x) __builtin_expect(!!(x), 0)

AT_INLINE int64_t imin(int64_t a, int64_t b) { return a < b ? a : b; }
AT_INLINE int64_t imax(int64_t a, int64_t b) { return a > b ? a : b; }
AT_INLINE int64_t b2i(bool b) { return b ? 1 : 0; }

/* Square roots without errno (as Goose's math::sqrt): C's sqrt has to set
   errno for a negative argument, so the compiler guards each square root
   with a test and a library call, and does not vectorize a loop with one.
   The value is the same correctly rounded root. Clang has a builtin
   without errno; gcc has none, and needs -fno-math-errno (build.sh). */
#if defined(__has_builtin)
#if __has_builtin(__builtin_elementwise_sqrt)
#define AT_SQRT_ELEMENTWISE 1
#endif
#endif
#if defined(AT_SQRT_ELEMENTWISE)
AT_INLINE double at_sqrt(double x) { return __builtin_elementwise_sqrt(x); }
AT_INLINE float at_sqrtf(float x) { return __builtin_elementwise_sqrt(x); }
#else
#if defined(__GNUC__) && !defined(__NO_MATH_ERRNO__)
#warning "compile with -fno-math-errno: sqrt with errno does not vectorize"
#endif
AT_INLINE double at_sqrt(double x) { return __builtin_sqrt(x); }
AT_INLINE float at_sqrtf(float x) { return __builtin_sqrtf(x); }
#endif

/* ---------------------------------------------------------------------
   SIMD functions. The body is written once (an always_inline function)
   and compiled three times: for the baseline, for AVX2 (with FMA, BMI1,
   BMI2, LZCNT, POPCNT) and for AVX-512 (F, BW, CD, DQ, VL). The baseline
   version calls the highest version that the CPU supports. The
   contraction pragma above keeps the float results the same in all
   versions. Clang and gcc on x86-64 get the versions. */
#if defined(__GNUC__) && defined(__x86_64__)
#define AT_SIMD 1
#define AT_TARGET1 __attribute__((target("avx2,fma,bmi,bmi2,lzcnt,popcnt")))
#define AT_TARGET2 \
    __attribute__((target("avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx2,fma,bmi,bmi2,lzcnt,popcnt")))
int at_simd_level(void);
#ifdef AT_SIMD_DEBUG
/* For measurements (build with -DAT_SIMD_DEBUG): AT_SIMD_V1=",a,b," runs
   the SIMD functions a and b at most at level 1 (AVX2). */
static inline int at_simd_level_named(const char *name) {
    int l = at_simd_level();
    const char *e = getenv("AT_SIMD_V1");
    char key[64];
    snprintf(key, sizeof key, ",%s,", name);
    if (e && strstr(e, key) && l > 1) l = 1;
    return l;
}
#define AT_LEVEL(name) \
    static int lvl_ = -1; \
    if (lvl_ < 0) lvl_ = at_simd_level_named(#name); \
    switch (lvl_)
#else
#define AT_LEVEL(name) switch (at_simd_level())
#endif

#define SIMD_FN(ret, name, params, args)                                                   \
    AT_INLINE ret name##_impl params;                                                      \
    AT_TARGET2 AT_NOINLINE static ret name##_v2 params { return name##_impl args; }        \
    AT_TARGET1 AT_NOINLINE static ret name##_v1 params { return name##_impl args; }        \
    AT_NOINLINE static ret name##_v0 params { return name##_impl args; }                   \
    static inline ret name params {                                                        \
        AT_LEVEL(name) {                                                                   \
        case 2: return name##_v2 args;                                                     \
        case 1: return name##_v1 args;                                                     \
        default: return name##_v0 args;                                                    \
        }                                                                                  \
    }                                                                                      \
    AT_INLINE ret name##_impl params

#define SIMD_VOID(name, params, args)                                                      \
    AT_INLINE void name##_impl params;                                                     \
    AT_TARGET2 AT_NOINLINE static void name##_v2 params { name##_impl args; }              \
    AT_TARGET1 AT_NOINLINE static void name##_v1 params { name##_impl args; }              \
    AT_NOINLINE static void name##_v0 params { name##_impl args; }                         \
    static inline void name params {                                                       \
        AT_LEVEL(name) {                                                                   \
        case 2: name##_v2 args; return;                                                    \
        case 1: name##_v1 args; return;                                                    \
        default: name##_v0 args; return;                                                   \
        }                                                                                  \
    }                                                                                      \
    AT_INLINE void name##_impl params
#else
#define AT_SIMD 0
#define SIMD_FN(ret, name, params, args) static ret name params
#define SIMD_VOID(name, params, args) static void name params
#endif

/* ---------------------------------------------------------------------
   Data stacks, as in the Goose runtime. Each thread has a context with
   a fixed number of stacks. A stack is a region of `reserve` bytes of
   address space (mmap, MAP_NORESERVE), followed by an unmapped gap of
   AT_STACK_GAP bytes: the OS commits a page when it is first touched,
   and the pages stay mapped, so later frames reuse them warm. Runaway
   growth faults in the gap; nothing else checks a size.

   - An array that can grow takes a stack of its own (stk_take): it
     starts at the stack's base and grows in place, so a push is a store
     and a length increment, with no capacity test, no realloc and no
     copy. Hot loops keep the length in a local.
   - Scratch of a size known in advance is bumped from stack 0
     (stk_alloc).
   - stk_enter and stk_leave bracket a scope: leave frees all stacks
     taken and all scratch bumped since the enter (a watermark).
   No memory is freed to the OS while the context lives. */
#define AT_STACK_GAP ((size_t)1 << 20)
/* The smallest reserve of a stack. Linux and macOS give a process
   terabytes of address space, so a stack gets 2 GB whatever the image.
   iOS gives tens of GB (with the extended virtual addressing
   entitlement), so there a stack gets what the image needs (below). */
#if defined(AT_STACK_RESERVE_MIN)
#elif defined(__APPLE__) && TARGET_OS_IPHONE
#define AT_STACK_RESERVE_MIN ((size_t)16 << 20)
#else
#define AT_STACK_RESERVE_MIN ((size_t)2048 << 20)
#endif
/* The stacks of each context, and the bytes a stack can need for each
   pixel of the quad image (measured at most 31, 7.1 and 0.6, on noise
   images of 48 MP), with a margin. The image context holds the image
   arrays of the detector (9 stacks); the matching context is the
   scratch of the calling thread. */
#define AT_DETECTOR_NSTACKS 8 /* the decode tables and the detections */
#define AT_DETECTOR_RESERVE ((size_t)256 << 20)
#define AT_IMAGE_NSTACKS 10
#define AT_IMAGE_BYTES_PER_PIXEL 40
#define AT_MATCH_NSTACKS 20
#define AT_MATCH_BYTES_PER_PIXEL 10
#define AT_WORKER_NSTACKS 8
#define AT_WORKER_BYTES_PER_PIXEL 2

typedef struct at_ctx {
    uint8_t *region; /* stack k starts at region + k * span */
    size_t span;     /* the reserve and the gap of a stack */
    int n;           /* the number of stacks */
    int sp;          /* the next free stack (stack 0 is the bump stack) */
    uint8_t *bump;   /* the top of stack 0 */
} at_ctx;

typedef struct {
    int sp;
    uint8_t *bump;
} at_mark;

/* A context with its stacks reserved (abort if the OS refuses), and its
   release. */
void at_ctx_init(at_ctx *cx, int n, size_t reserve);
void at_ctx_release(at_ctx *cx);
/* Reserve the stacks again if they are smaller than reserve (or the
   minimum); return true if it did. The context must hold no data. */
bool at_ctx_fit(at_ctx *cx, int n, size_t reserve);

AT_INLINE at_mark stk_enter(const at_ctx *cx) { return (at_mark){cx->sp, cx->bump}; }
AT_INLINE void stk_leave(at_ctx *cx, at_mark m) {
    cx->sp = m.sp;
    cx->bump = m.bump;
}

AT_INLINE void *stk_take_(at_ctx *cx) {
    if (AT_UNLIKELY(cx->sp >= cx->n)) {
        fprintf(stderr, "apriltag: out of data stacks\n");
        abort();
    }
    return cx->region + (size_t)cx->sp++ * cx->span;
}

AT_INLINE void *stk_alloc_(at_ctx *cx, size_t bytes) {
    uint8_t *p = cx->bump;
    cx->bump += (bytes + 63) & ~(size_t)63;
    return p;
}

/* The base of a stack of its own, for an array of T that may grow. */
#define STK_TAKE(cx, T) ((T *)stk_take_(cx))
/* Scratch for n elements of T (64-byte aligned), not set. */
#define STK_ALLOC(cx, T, n) ((T *)stk_alloc_((cx), (size_t)(n) * sizeof(T)))
/* Scratch for n elements of T, set to zero. */
#define STK_ZALLOC(cx, T, n) ((T *)memset(STK_ALLOC(cx, T, n), 0, (size_t)(n) * sizeof(T)))

/* A growable array on a stack of its own: data is the stack's base and
   never moves. */
#define svec_t(T) struct { T *data; int64_t len; }
#define svec_take(cx, v) ((v).data = stk_take_(cx), (v).len = 0)
#define svec_push(v, x) ((v).data[(v).len++] = (x))
#define svec_append(v, src, n)                                                             \
    do {                                                                                   \
        int64_t svec_n_ = (n);                                                             \
        if (svec_n_ > 0) memcpy((v).data + (v).len, (src), (size_t)svec_n_ * sizeof(*(v).data)); \
        (v).len += svec_n_;                                                                \
    } while (0)
/* Set the length to n; elements past the old length are zero. */
#define svec_resize0(v, n)                                                                 \
    do {                                                                                   \
        int64_t svec_n_ = (n);                                                             \
        if (svec_n_ > (v).len)                                                             \
            memset((v).data + (v).len, 0, (size_t)(svec_n_ - (v).len) * sizeof(*(v).data)); \
        (v).len = svec_n_;                                                                 \
    } while (0)

double at_clock(void);

/* ---------------------------------------------------------------------
   Shared types. */

/* A row run of equal thresholded pixels. `end` is inclusive. Packed (5
   bytes, as in Goose): the photos have millions of runs. */
typedef struct __attribute__((packed)) { uint16_t start, end; uint8_t v; } Run;

/* A union-find node: its parent, and for a root the size of the tree. */
typedef struct { uint32_t parent, size; } UfNode;

/* A boundary point between a black and a white pixel. The coordinates
   are twice the actual value. */
typedef struct { uint16_t x, y; int16_t gx, gy; } Pt;

/* A quad candidate. */
typedef struct {
    float p[4][2];
    bool reversed_border;
    double H[9];
} Quad;

typedef svec_t(uint8_t) vec_u8;
typedef svec_t(uint16_t) vec_u16;
typedef svec_t(uint32_t) vec_u32;
typedef svec_t(uint64_t) vec_u64;
typedef svec_t(int64_t) vec_i64;
typedef svec_t(float) vec_f32;
typedef svec_t(double) vec_f64;
typedef svec_t(Run) vec_run;
typedef svec_t(Pt) vec_pt;
typedef svec_t(Quad) vec_quad;
typedef svec_t(UfNode) vec_uf;

AT_INLINE uint64_t load64(const uint8_t *b) {
    uint64_t v;
    memcpy(&v, b, 8);
    return v;
}

AT_INLINE uint64_t bits_of_f64(double x) {
    uint64_t b;
    memcpy(&b, &x, 8);
    return b;
}

AT_INLINE uint32_t bits_of_f32(float x) {
    uint32_t b;
    memcpy(&b, &x, 4);
    return b;
}

AT_INLINE float f32_of_bits(uint32_t b) {
    float x;
    memcpy(&x, &b, 4);
    return x;
}

/* All ones if c, else zero. */
AT_INLINE uint32_t mask32(bool c) { return (uint32_t)0 - (uint32_t)c; }

/* m ? a : b for a mask m of mask32, with bit operations: gcc's jump
   threading makes nested float selects into branches that its loop
   vectorizer does not take. */
AT_INLINE float fsel(uint32_t m, float a, float b) {
    return f32_of_bits((bits_of_f32(a) & m) | (bits_of_f32(b) & ~m));
}

/* ---------------------------------------------------------------------
   Quick decode tables (decode.c). */
typedef struct {
    int64_t maxhamming;
    int64_t chunk_size;
    int64_t capacity;
    /* where this family's tables start in the detector's arrays */
    int64_t off_base;
    int64_t ids_base;
} QuickDecode;

/* A gray image w x h, rows s bytes apart. */
typedef struct {
    const uint8_t *im;
    int64_t s, w, h;
} ImView;

/* What quad_sample needs of a family. */
typedef struct {
    int32_t wab, total_width, nbits;
    bool reversed_border;
    const int32_t *bit_x, *bit_y;
} FamGeom;

/* What a code lookup needs of a family. */
typedef struct {
    QuickDecode qd;
    int64_t ncodes;
    int64_t nbits;
} FamLookup;

/* The shared quick-decode tables of a detector. */
typedef struct {
    const uint32_t *offsets;
    const uint16_t *ids;
    const uint64_t *codes;
} QdTables;

/* The detector internals. */
#define AT_MAX_FAMILIES 16

struct at_pool;

struct at_internal {
    at_family families[AT_MAX_FAMILIES];
    QuickDecode qd[AT_MAX_FAMILIES];
    int nfamilies;
    /* The detector's own stacks: the arrays below keep their stack while
       the detector lives, so they grow in place between frames and their
       pages stay mapped. */
    at_ctx dcx;
    vec_u32 qd_offsets;
    vec_u16 qd_ids;
    vec_u64 qd_codes;
    svec_t(at_detection) dets;
    /* buffers of quad_thresh: each frame writes their elements before it
       reads them */
    uint8_t *buf_qim;
    uint8_t *buf_t;
    UfNode *buf_uf;
    /* the stacks of the image arrays (above and those of the pool), sized
       for the largest image so far */
    at_ctx icx;
    /* the scratch stacks of the calling thread */
    at_ctx mcx;
    struct at_pool *pool;
    /* the GPU stage (NULL: not in use), and whether it runs for this image */
    struct at_metal *metal;
    at_gpu_mode gpu_mode;
    bool gpu_pending;
    bool gpu_used; /* the GPU stage gave the clusters of this image */
    /* AT_GPU_AUTO: the images so far, the average seconds for each
       boundary point without and with the GPU, and the boundary points
       of this image */
    int64_t gpu_frames;
    int64_t gpu_points;
    /* the path in use, and when it tries the other one */
    bool gpu_best;
    int64_t gpu_probe, gpu_next_probe;
    /* with the pool: the first cluster task of the GPU, the part of the
       tasks that it gets, and the images to go without it */
    int64_t gpu_k0;
    double gpu_frac;
    int64_t gpu_skip, gpu_backoff;
    double gpu_ema[2];
};

#endif
