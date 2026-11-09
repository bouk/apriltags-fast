/* Data stack contexts, the SIMD level and the clock. */
#include "common.h"

#include <sys/mman.h>
#include <time.h>

/* Reserve the stacks of a context as one mapping: commit on touch
   (overcommit, MAP_NORESERVE), each stack followed by a PROT_NONE gap.
   As the Goose runtime does, the first 2 MB of each stack get no
   transparent huge page: most stacks hold a few bytes, which a huge
   page would commit 2 MB for. */
void at_ctx_init(at_ctx *cx) {
    /* Where the address space is limited (valgrind, ulimit -v), reserve
       less: halve the stacks until the mapping succeeds. */
    size_t reserve = AT_STACK_RESERVE;
    void *p = MAP_FAILED;
    for (; reserve >= ((size_t)64 << 20); reserve /= 2) {
        p = mmap(NULL, AT_NSTACKS * (reserve + AT_STACK_GAP), PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p != MAP_FAILED) break;
    }
    if (p == MAP_FAILED) {
        fprintf(stderr, "apriltag: cannot reserve data stacks\n");
        abort();
    }
    size_t span = reserve + AT_STACK_GAP;
    uint8_t *r = (uint8_t *)p;
    for (int k = 0; k < AT_NSTACKS; k++) {
        uint8_t *s = r + (size_t)k * span;
        if (mprotect(s + reserve, AT_STACK_GAP, PROT_NONE)) abort();
#ifdef MADV_NOHUGEPAGE
        madvise(s, (size_t)2 << 20, MADV_NOHUGEPAGE);
#endif
    }
    cx->region = r;
    cx->span = span;
    cx->sp = 1;
    cx->bump = r;
}

void at_ctx_release(at_ctx *cx) {
    if (cx->region) munmap(cx->region, AT_NSTACKS * cx->span);
    cx->region = NULL;
}

double at_clock(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)((int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec) / 1000000000.0;
}

#if AT_SIMD
static inline void cpuid(uint32_t leaf, uint32_t r[4]) {
    __asm__ volatile("cpuid" : "=a"(r[0]), "=b"(r[1]), "=c"(r[2]), "=d"(r[3]) : "a"(leaf), "c"(0));
}

/* The highest level both the CPU and the OS support (the OS has to save
   the wider registers: XGETBV's XCR0). */
static int simd_detect(void) {
    uint32_t r[4], b7, xcr0, xhi;
    cpuid(0, r);
    if (r[0] < 7) return 0;
    cpuid(1, r);
    /* OSXSAVE, AVX, POPCNT, FMA */
    uint32_t need1 = 1u << 27 | 1u << 28 | 1u << 23 | 1u << 12;
    if ((r[2] & need1) != need1) return 0;
    __asm__ volatile("xgetbv" : "=a"(xcr0), "=d"(xhi) : "c"(0));
    (void)xhi;
    if ((xcr0 & 0x6) != 0x6) return 0;
    cpuid(7, r);
    b7 = r[1];
    /* AVX2, BMI1, BMI2 */
    if ((b7 & (1u << 5 | 1u << 3 | 1u << 8)) != (1u << 5 | 1u << 3 | 1u << 8)) return 0;
    cpuid(0x80000000u, r);
    if (r[0] < 0x80000001u) return 0;
    cpuid(0x80000001u, r);
    if (!(r[2] & (1u << 5))) return 0; /* LZCNT */
    if ((xcr0 & 0xe0) != 0xe0) return 1;
    /* AVX512F, DQ, CD, BW, VL */
    uint32_t need7 = 1u << 16 | 1u << 17 | 1u << 28 | 1u << 30 | 1u << 31;
    if ((b7 & need7) != need7) return 1;
    return 2;
}

int at_simd_level(void) {
    static int level = -1;
    int l = __atomic_load_n(&level, __ATOMIC_RELAXED);
    if (l < 0) {
        l = simd_detect();
        const char *e = getenv("AT_SIMD");
        if (e && *e >= '0' && *e < '0' + l) l = *e - '0';
        __atomic_store_n(&level, l, __ATOMIC_RELAXED);
    }
    return l;
}
#endif
