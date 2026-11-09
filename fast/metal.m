/* The GPU (Metal) cluster stage: see metal.h. Compile with ARC. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "metal.h"

/* The kernels. The pixel connections, the component sizes and the point
   rules are those of fast/unionfind.c and fast/clusters.c:
   - A pixel of columns 0..w-2 connects to an equal pixel (not 127) to
     its left; from column 1 on to an equal pixel above it; a white pixel
     from column 1 on also to the white pixels above-left and above-right
     (not in the last column).
   - The pixel of the last column is a component of its own. A white
     pixel at column w-2 joins the last pixel of the row above if that is
     white and the pixel below it is not.
   - Only rows 1..yend-1 connect to the row above; the pixels of the rows
     from yend on are never in a big enough component. */
#define MSL(...) #__VA_ARGS__
static const char *const kernel_source = "#include <metal_stdlib>\nusing namespace metal;\n" MSL(
struct P {
    uint w;       /* the quad detection image */
    uint h;
    uint sstride; /* the caller's image */
    uint soff;
    int diff;
    uint minpx;
    uint yend;
    uint tw;
    uint th;
    uint tmask;   /* table size - 1 */
    uint cap;     /* the most points */
    uint nseg;    /* segments a row */
    uint ya;      /* the region: rows ya..yb, labeled on their own */
    uint yb;
    uint touch;   /* 1: components can go on above row ya */
    uint cy0;     /* the first row of k_copy and k_dec2 */
    uint ty0;     /* the first tile row of k_tile */
    uint hy0;     /* the first tile row of k_tthr */
    uint full;    /* 1: count the points and the entries of each slot, and list the slots */
};
constant uint SEG = 64;

kernel void k_copy(constant P &p [[buffer(0)]], device const uchar *src [[buffer(1)]],
                   device uchar *qim [[buffer(2)]], uint2 gg [[thread_position_in_grid]]) {
    uint2 g = uint2(gg.x, gg.y + p.cy0);
    if (g.x >= p.w || g.y >= p.h) return;
    qim[g.y * p.w + g.x] = src[p.soff + g.y * p.sstride + g.x];
}

/* The 2x2 box filter of decimate2_row: two rounded averages for the
   part of the row that is a multiple of 32 pixels, the exact rounded
   mean for the rest. */
kernel void k_dec2(constant P &p [[buffer(0)]], device const uchar *src [[buffer(1)]],
                   device uchar *qim [[buffer(2)]], uint2 gg [[thread_position_in_grid]]) {
    uint2 g = uint2(gg.x, gg.y + p.cy0);
    if (g.x >= p.w || g.y >= p.h) return;
    uint a = p.soff + 2 * g.y * p.sstride + 2 * g.x;
    uint b = a + p.sstride;
    uint s00 = src[a], s01 = src[a + 1], s10 = src[b], s11 = src[b + 1];
    uint nv = p.w - p.w % 32;
    uint o;
    if (g.x < nv) {
        uint v0 = (s00 + s10 + 1) >> 1;
        uint v1 = (s01 + s11 + 1) >> 1;
        o = (v0 + v1 + 1) >> 1;
    } else {
        o = (s00 + s01 + s10 + s11 + 2) >> 2;
    }
    qim[g.y * p.w + g.x] = uchar(o);
}

/* Min and max of each 4x4 tile. */
kernel void k_tile(constant P &p [[buffer(0)]], device const uchar *qim [[buffer(1)]],
                   device uchar2 *tmm [[buffer(2)]], uint2 gg [[thread_position_in_grid]]) {
    uint2 g = uint2(gg.x, gg.y + p.ty0);
    if (g.x >= p.tw || g.y >= p.th) return;
    uint lo = 255, hi = 0;
    for (uint dy = 0; dy < 4; dy++) {
        uint base = (4 * g.y + dy) * p.w + 4 * g.x;
        for (uint dx = 0; dx < 4; dx++) {
            uint v = qim[base + dx];
            lo = min(lo, v);
            hi = max(hi, v);
        }
    }
    tmm[g.y * p.tw + g.x] = uchar2(uchar(lo), uchar(hi));
}

/* The threshold of each tile from the 3x3 tiles around it, and whether
   the tile is flat. */
kernel void k_tthr(constant P &p [[buffer(0)]], device const uchar2 *tmm [[buffer(1)]],
                   device uchar2 *tth [[buffer(2)]], uint2 gg [[thread_position_in_grid]]) {
    uint2 g = uint2(gg.x, gg.y + p.hy0);
    if (g.x >= p.tw || g.y >= p.th) return;
    int lo = 255, hi = 0;
    for (int dy = -1; dy <= 1; dy++) {
        int y = int(g.y) + dy;
        if (y < 0 || y >= int(p.th)) continue;
        for (int dx = -1; dx <= 1; dx++) {
            int x = int(g.x) + dx;
            if (x < 0 || x >= int(p.tw)) continue;
            uchar2 m = tmm[uint(y) * p.tw + uint(x)];
            lo = min(lo, int(m.x));
            hi = max(hi, int(m.y));
        }
    }
    tth[g.y * p.tw + g.x] = uchar2(uchar(lo + (hi - lo) / 2), uchar(hi - lo < p.diff ? 1 : 0));
}

kernel void k_thresh(constant P &p [[buffer(0)]], device const uchar *qim [[buffer(1)]],
                     device const uchar2 *tth [[buffer(2)]], device uchar *t [[buffer(3)]],
                     uint2 gg [[thread_position_in_grid]]) {
    uint2 g = uint2(gg.x, gg.y + p.ya);
    if (g.x >= p.w || g.y > p.yb) return;
    uint tx = min(g.x / 4, p.tw - 1);
    uint ty = min(g.y / 4, p.th - 1);
    uchar2 tf = tth[ty * p.tw + tx];
    bool flat = tf.y != 0 && g.x < 4 * p.tw && g.y < 4 * p.th;
    uint i = g.y * p.w + g.x;
    t[i] = flat ? uchar(127) : (qim[i] > tf.x ? uchar(255) : uchar(0));
}

/* The distance of pixel x of a row to the first pixel of its piece. */
static uint soff_left(device const uchar *t, uint base, uint x) {
    uint x0 = x - x % SEG;
    uchar c = t[base + x];
    uint k = x;
    while (k > x0 && t[base + k - 1] == c) k--;
    return x - k;
}

/* The rows are cut into segments of SEG pixels: a thread a segment, as a
   long loop in one thread is slow. The nodes of the union-find are the
   pieces of the runs in the segments: a node is the index of the first
   pixel of its piece, and soff is the distance of each pixel to it (the
   other pixels have no entry in parent and size). The first piece of a
   run that goes on from the segment before points at the piece to its
   left. */
kernel void k_rows(constant P &p [[buffer(0)]], device const uchar *t [[buffer(1)]],
                   device uint *parent [[buffer(2)]], device uint *size [[buffer(3)]],
                   device uchar *soff [[buffer(4)]], uint2 gg [[thread_position_in_grid]]) {
    uint2 g = uint2(gg.x, gg.y + p.ya);
    uint x0 = g.x * SEG;
    if (g.y > p.yb || x0 >= p.w) return;
    uint base = g.y * p.w;
    uint x1 = min(x0 + SEG, p.w);
    /* the last column is a component of its own */
    uint xe = min(x1, p.w - 1);
    uint s = x0;
    uchar v = 127;
    for (uint x = x0; x < xe; x++) {
        uchar c = t[base + x];
        if (c != v || x == x0) {
            s = x;
            v = c;
            uint par = base + x;
            if (x == x0 && x0 > 0 && c != 127 && t[base + x0 - 1] == c) par = base + x0 - 1 - soff_left(t, base, x0 - 1);
            parent[base + x] = par;
            size[base + x] = 0;
        }
        soff[base + x] = uchar(x - s);
    }
    if (x1 == p.w) {
        parent[base + p.w - 1] = base + p.w - 1;
        size[base + p.w - 1] = 0;
        soff[base + p.w - 1] = 0;
    }
}

static uint uf_find(device atomic_uint *parent, uint i) {
    for (;;) {
        uint q = atomic_load_explicit(&parent[i], memory_order_relaxed);
        if (q == i) return i;
        i = q;
    }
}

/* Lock-free union: the root with the larger index points at the other. */
static void uf_unite(device atomic_uint *parent, uint a, uint b) {
    for (;;) {
        a = uf_find(parent, a);
        b = uf_find(parent, b);
        if (a == b) return;
        if (a < b) {
            uint tmp = a;
            a = b;
            b = tmp;
        }
        uint e = a;
        if (atomic_compare_exchange_weak_explicit(&parent[a], &e, b, memory_order_relaxed, memory_order_relaxed))
            return;
    }
}

kernel void k_union(constant P &p [[buffer(0)]], device const uchar *t [[buffer(1)]],
                    device atomic_uint *parent [[buffer(2)]], device const uchar *soff [[buffer(3)]],
                    uint2 gg [[thread_position_in_grid]]) {
    /* the rows after ya: the components of the region are its own */
    uint2 g = uint2(gg.x, gg.y + p.ya + 1);
    if (g.y > p.yb || g.y >= p.yend || g.x + 1 >= p.w) return;
    uint x = g.x;
    uint i = g.y * p.w + x;
    uchar v = t[i];
    if (v == 127) return;
    uint u = i - p.w;
    uchar up = t[u];
    uint ni = i - soff[i];
    if (x >= 1 && up == v) {
        /* once for each pair of runs */
        if (x == 1 || t[i - 1] != v || t[u - 1] != v) uf_unite(parent, ni, u - soff[u]);
    }
    if (v == 255 && up != 255) {
        if (x >= 1) {
            if (t[u - 1] == 255) uf_unite(parent, ni, u - 1 - soff[u - 1]);
            if (x + 2 < p.w && t[u + 1] == 255) uf_unite(parent, ni, u + 1 - soff[u + 1]);
        }
        if (x + 2 == p.w && t[u + 1] == 255) uf_unite(parent, ni, u + 1);
    }
}

/* A thread a segment: each node points at its root (the label of its
   pixels), and the pixels of each piece are added to the size of its
   component. A thread can read a parent that another thread just set to
   the root: that is also right. With touch, bit 31 of the size is set
   for the components of row ya: they can go on above the region, so a
   small one can be a part of a big one. */
kernel void k_label(constant P &p [[buffer(0)]], device const uchar *t [[buffer(1)]],
                    device atomic_uint *parent [[buffer(2)]], device atomic_uint *size [[buffer(3)]],
                    uint2 gg [[thread_position_in_grid]]) {
    uint2 g = uint2(gg.x, gg.y + p.ya);
    uint x0 = g.x * SEG;
    if (g.y > p.yb || g.y >= p.yend || x0 >= p.w) return;
    uint base = g.y * p.w;
    uint x1 = min(x0 + SEG, p.w);
    uint xe = min(x1, p.w - 1);
    bool mark = p.touch != 0 && g.y == p.ya;
    uint x = x0;
    while (x < xe) {
        uchar v = t[base + x];
        uint s = x;
        x++;
        while (x < xe && t[base + x] == v) x++;
        if (v != 127) {
            uint root = uf_find(parent, base + s);
            atomic_store_explicit(&parent[base + s], root, memory_order_relaxed);
            atomic_fetch_add_explicit(&size[root], x - s, memory_order_relaxed);
            if (mark) atomic_fetch_or_explicit(&size[root], 0x80000000u, memory_order_relaxed);
        }
    }
    if (x1 == p.w) {
        uint last = base + p.w - 1;
        uint root = uf_find(parent, last);
        atomic_store_explicit(&parent[last], root, memory_order_relaxed);
        atomic_fetch_add_explicit(&size[root], 1, memory_order_relaxed);
        if (mark) atomic_fetch_or_explicit(&size[root], 0x80000000u, memory_order_relaxed);
    }
}

/* For each pixel: 1 if its component is big enough, 2 if that is not
   known (a small component that can go on above the region), else 0. */
kernel void k_ok(constant P &p [[buffer(0)]], device const uchar *t [[buffer(1)]],
                 device const uint *size [[buffer(2)]], device const uint *lab [[buffer(3)]],
                 device uchar *okb [[buffer(4)]], device const uchar *soff [[buffer(5)]],
                 uint2 gg [[thread_position_in_grid]]) {
    uint2 g = uint2(gg.x, gg.y + p.ya);
    if (g.x >= p.w || g.y > p.yb) return;
    uint i = g.y * p.w + g.x;
    uint o = 0;
    if (t[i] != 127 && g.y < p.yend) {
        uint s = size[lab[i - soff[i]]];
        o = (s & 0x7fffffffu) >= p.minpx ? 1u : (s >> 31) * 2u;
    }
    okb[i] = uchar(o);
}

static uint opp(uchar a, uchar b) { return (a ^ b) == 255 ? 1u : 0u; }

/* The point mask of each pixel (point_masks of clusters.c): bit 0 is the
   neighbor (1,0), bit 1 (0,1), bit 2 (-1,1), bit 3 (1,1). A component
   that is not known to be big enough counts as big enough: the CPU
   removes its points later if it is not. The (-1,1) point is left out
   if the pixel to the left has the (1,1) point of the same pixel pair;
   where that depends on such a component, bit 4 is set with bit 2, and
   the CPU decides. */
kernel void k_mask(constant P &p [[buffer(0)]], device const uchar *t [[buffer(1)]],
                   device const uchar *okb [[buffer(2)]], device uchar *mask [[buffer(3)]],
                   uint2 gg [[thread_position_in_grid]]) {
    uint2 g = uint2(gg.x, gg.y + p.ya);
    if (g.x >= p.w || g.y >= p.yb) return;
    uint x = g.x;
    uint i = g.y * p.w + x;
    uint m = 0;
    if (g.y >= 1 && g.y + 1 < p.h && x + 1 < p.w && okb[i] != 0) {
        uint j = i + p.w;
        uchar v = t[i];
        uint c10 = opp(v, t[i + 1]) & uint(okb[i + 1] != 0);
        if (x == 0) {
            m = c10;
        } else {
            uint c01 = opp(v, t[j]) & uint(okb[j] != 0);
            uint c11 = opp(v, t[j + 1]) & uint(okb[j + 1] != 0);
            uint cm1 = opp(v, t[j - 1]) & uint(okb[j - 1] != 0);
            /* the (1,1) point of the pixel to the left is the same pair
               (the first pixel has none) */
            uint p11 = x == 1 ? 0u : (opp(t[i - 1], t[j]) & uint(okb[i - 1] != 0) & uint(okb[j] != 0));
            uint open = p11 & uint(okb[i - 1] == 2 || okb[j] == 2);
            m = c10 | (c01 << 1) | ((cm1 & ((p11 ^ 1) | open)) << 2) | (c11 << 3) | ((cm1 & open) << 4);
        }
    }
    mask[g.y * p.nseg * SEG + x] = uchar(m);
}

/* The slot of the pair (a, b) in the table; a new slot if there is none.
   A slot is claimed by its first key, then by its second: a thread with
   another second key goes on to the next slot. */
static uint slot_of(device atomic_uint *ka, device atomic_uint *kb, device atomic_uint *state, device uint *slots,
                    uint a, uint b, uint tmask, bool full) {
    uint s = (a * 0x9E3779B1u ^ b * 0x85EBCA77u);
    s = (s ^ (s >> 15)) & tmask;
    for (uint n = 0; n < 4096;) {
        uint ea = atomic_load_explicit(&ka[s], memory_order_relaxed);
        if (ea == 0) {
            uint e = 0;
            if (atomic_compare_exchange_weak_explicit(&ka[s], &e, a + 1, memory_order_relaxed, memory_order_relaxed))
                ea = a + 1;
            else
                ea = e;
            if (ea == 0) continue;
        }
        if (ea == a + 1) {
            uint eb = atomic_load_explicit(&kb[s], memory_order_relaxed);
            if (eb == 0) {
                uint e = 0;
                if (atomic_compare_exchange_weak_explicit(&kb[s], &e, b + 1, memory_order_relaxed,
                                                          memory_order_relaxed)) {
                    eb = b + 1;
                    uint k = atomic_fetch_add_explicit(&state[2], 1, memory_order_relaxed);
                    if (full && k <= tmask) slots[k] = s;
                } else {
                    eb = e;
                }
                if (eb == 0) continue;
            }
            if (eb == b + 1) return s;
        }
        s = (s + 1) & tmask;
        n++;
    }
    atomic_store_explicit(&state[1], 1, memory_order_relaxed);
    return 0;
}

/* A thread a segment: the points of the segment in pixel order, then
   neighbor order (emit_row of clusters.c). Each run of points of one
   cluster is an entry: its first point, and its slot (23 bits), a flag
   (bit 23) and the count less one (8 high bits). The flag is for a
   single (-1,1) point that the CPU must decide on (see k_mask). seginfo
   gets where the entries of the segment are, and how many. */
kernel void k_emit(constant P &p [[buffer(0)]], device const uchar *t [[buffer(1)]],
                   device const uint *lab [[buffer(2)]], device const uint *mask [[buffer(3)]],
                   device atomic_uint *ka [[buffer(4)]], device atomic_uint *kb [[buffer(5)]],
                   device atomic_uint *cnt [[buffer(6)]], device atomic_uint *ecnt [[buffer(7)]],
                   device atomic_uint *state [[buffer(8)]], device uint *slots [[buffer(9)]],
                   device uint2 *pts [[buffer(10)]], device uint2 *entries [[buffer(11)]],
                   device uint *seginfo [[buffer(12)]], device const uchar *soff [[buffer(13)]],
                   uint2 gg [[thread_position_in_grid]]) {
    uint2 g = uint2(gg.x, gg.y + p.ya);
    uint x0 = g.x * SEG;
    uint y = g.y;
    if (y >= p.yb || x0 >= p.w) return;
    uint si = 2 * (y * p.nseg + g.x);
    seginfo[si] = 0;
    seginfo[si + 1] = 0;
    if (y < 1 || y + 1 >= p.h) return;
    uint base = y * p.w;
    uint x1 = min(x0 + SEG, p.w);
    const uint offs[4] = {1u, p.w, p.w - 1u, p.w + 1u};
    /* the masks of the segment: SEG / 4 words (the mask rows are a
       multiple of SEG bytes). The segment has no more entries than
       points. */
    uint mbase = (y * p.nseg + g.x) * (SEG / 4);
    uint n = 0;
    for (uint k = 0; k < SEG / 4; k++) n += popcount(mask[mbase + k] & 0x0f0f0f0fu);
    if (n == 0) return;
    uint at = atomic_fetch_add_explicit(&state[0], n, memory_order_relaxed);
    uint ea = atomic_fetch_add_explicit(&state[3], n, memory_order_relaxed);
    if (at + n > p.cap || ea + n > p.cap) {
        atomic_store_explicit(&state[1], 1, memory_order_relaxed);
        return;
    }
    seginfo[si] = ea;
    const uint dxy[4] = {1u, 65536u, 65535u, 65537u};
    /* gx | gy << 16 for neighbor j and pixel value a (0 or 255), at
       2j + (a & 1) */
    const uint gtab[8] = {0x000000ffu, 0x0000ff01u, 0x00ff0000u, 0xff010000u,
                          0x00ffff01u, 0xff0100ffu, 0x00ff00ffu, 0xff01ff01u};
    uint la = 0xffffffffu, lb = 0xffffffffu;
    uint lslot = 0, lcnt = 0, lat = at, ea0 = ea, lflag = 0;
    bool full = p.full != 0;
    for (uint x = x0; x < x1; x++) {
        uint mw = mask[mbase + (x - x0) / 4];
        if (mw == 0) {
            x |= 3;
            continue;
        }
        uint m = (mw >> (8 * ((x - x0) & 3))) & 31u;
        if (m == 0) continue;
        uint i = base + x;
        uint r0 = lab[i - soff[i]];
        uint a = t[i] & 1u;
        uint xy = ((2 * y) << 16) + 2 * x;
        for (uint j = 0; j < 4; j++) {
            if ((m & (1u << j)) == 0) continue;
            uint k = i + offs[j];
            uint r1 = lab[k - soff[k]];
            uint ia = min(r0, r1), ib = max(r0, r1);
            uint flag = (j == 2 && (m & 16u) != 0) ? 1u : 0u;
            if (ia != la || ib != lb || (flag | lflag) != 0) {
                if (lcnt != 0) {
                    if (full) {
                        atomic_fetch_add_explicit(&cnt[lslot], lcnt, memory_order_relaxed);
                        atomic_fetch_add_explicit(&ecnt[lslot], 1, memory_order_relaxed);
                    }
                    entries[ea++] = uint2(lat, lslot | (lflag << 23) | ((lcnt - 1) << 24));
                }
                lcnt = 0;
                lat = at;
                if (ia != la || ib != lb) lslot = slot_of(ka, kb, state, slots, ia, ib, p.tmask, full);
                la = ia;
                lb = ib;
                lflag = flag;
            }
            lcnt++;
            pts[at++] = uint2(xy + dxy[j], gtab[(j << 1) | a]);
        }
    }
    if (full) {
        atomic_fetch_add_explicit(&cnt[lslot], lcnt, memory_order_relaxed);
        atomic_fetch_add_explicit(&ecnt[lslot], 1, memory_order_relaxed);
    }
    entries[ea] = uint2(lat, lslot | (lflag << 23) | ((lcnt - 1) << 24));
    seginfo[si + 1] = ea + 1 - ea0;
}
);

enum { K_COPY, K_DEC2, K_TILE, K_TTHR, K_THRESH, K_ROWS, K_UNION, K_LABEL, K_OK, K_MASK, K_EMIT, K_COUNT };
static const char *const kernel_names[K_COUNT] = {"k_copy", "k_dec2",  "k_tile", "k_tthr", "k_thresh", "k_rows",
                                                  "k_union", "k_label", "k_ok",   "k_mask", "k_emit"};

typedef struct {
    uint32_t w, h, sstride, soff;
    int32_t diff;
    uint32_t minpx, yend, tw, th, tmask, cap, nseg, ya, yb, touch, cy0, ty0, hy0, full;
} KParams;

@interface ATMetal : NSObject {
  @public
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pso[K_COUNT];
    /* per image size */
    int64_t w, h, src_cap, tsize, cap, nseg;
    id<MTLBuffer> src, qim, tmm, tth, t, parent, size, soff, okb, mask;
    id<MTLBuffer> ka, kb, cnt, ecnt, slots, state, pts, entries, seginfo;
    id<MTLCommandBuffer> pending;
    id<MTLBuffer> src_now; /* the image of the pending work */
    int64_t ya, yb, ts_now;
    double gpu_start, gpu_end; /* of the last work that at_metal_finish waited for */
}
@end

@implementation ATMetal
@end

struct at_metal {
    void *obj; /* the retained ATMetal */
    /* the work: 0 none, 1 in progress, 2 done, 3 failed (set by Metal's
       thread at the end of the command buffer) */
    _Atomic int state;
    /* for traces: the times of the submit, the commit and the end */
    double t_submit, t_commit, t_done;
};

at_metal *at_metal_create(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) return NULL;
        ATMetal *m = [ATMetal new];
        m->dev = dev;
        m->queue = [dev newCommandQueue];
        NSError *err = nil;
        MTLCompileOptions *opt = [MTLCompileOptions new];
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kernel_source]
                                               options:opt
                                                 error:&err];
        if (!lib) {
            fprintf(stderr, "apriltags-fast: Metal kernels: %s\n", err.localizedDescription.UTF8String);
            return NULL;
        }
        for (int k = 0; k < K_COUNT; k++) {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:kernel_names[k]]];
            if (fn) m->pso[k] = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!m->pso[k]) {
                fprintf(stderr, "apriltags-fast: Metal kernel %s: %s\n", kernel_names[k],
                        err.localizedDescription.UTF8String);
                return NULL;
            }
        }
        at_metal *out = (at_metal *)calloc(1, sizeof *out);
        if (!out) abort();
        out->obj = (void *)CFBridgingRetain(m);
        return out;
    }
}

void at_metal_destroy(at_metal *mm) {
    if (!mm) return;
    @autoreleasepool {
        at_metal_finish(mm);
        CFBridgingRelease(mm->obj);
    }
    free(mm);
}

static id<MTLBuffer> new_buffer(ATMetal *m, int64_t bytes) {
    return [m->dev newBufferWithLength:(NSUInteger)(bytes > 0 ? bytes : 1) options:MTLResourceStorageModeShared];
}

/* The buffers for a quad detection image of w x h. */
static bool fit_buffers(ATMetal *m, int64_t w, int64_t h) {
    if (m->w == w && m->h == h) return true;
    int64_t n = w * h;
    int64_t tiles = (w / 4) * (h / 4);
    /* the table: a slot for each 8 pixels (a power of two); the points:
       one for each pixel. Images with more use the CPU stage. */
    int64_t ts = 4096;
    while (ts < n / 8) ts *= 2;
    m->tsize = ts;
    m->cap = n;
    m->w = m->h = 0;
    m->qim = new_buffer(m, n);
    m->tmm = new_buffer(m, 2 * tiles);
    m->tth = new_buffer(m, 2 * tiles);
    m->t = new_buffer(m, n);
    m->parent = new_buffer(m, 4 * n);
    m->size = new_buffer(m, 4 * n);
    m->soff = new_buffer(m, n);
    m->okb = new_buffer(m, n);
    m->mask = new_buffer(m, ((w + AT_METAL_SEG - 1) / AT_METAL_SEG) * AT_METAL_SEG * h);
    m->ka = new_buffer(m, 4 * ts);
    m->kb = new_buffer(m, 4 * ts);
    m->cnt = new_buffer(m, 4 * ts);
    m->slots = new_buffer(m, 4 * ts);
    m->state = new_buffer(m, 16);
    /* 3 points more: gather_points reads past a segment */
    m->pts = new_buffer(m, 8 * (m->cap + 3));
    m->entries = new_buffer(m, 8 * m->cap);
    m->ecnt = new_buffer(m, 4 * ts);
    m->nseg = (w + AT_METAL_SEG - 1) / AT_METAL_SEG;
    m->seginfo = new_buffer(m, 8 * h * m->nseg);
    if (!m->qim || !m->tmm || !m->tth || !m->t || !m->parent || !m->size || !m->soff || !m->okb || !m->mask ||
        !m->ka || !m->kb || !m->cnt || !m->slots || !m->state || !m->pts || !m->entries || !m->ecnt || !m->seginfo)
        return false;
    m->w = w;
    m->h = h;
    return true;
}

static void run2d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso, int64_t gw, int64_t gh) {
    NSUInteger tw = pso.threadExecutionWidth;
    NSUInteger th = pso.maxTotalThreadsPerThreadgroup / tw;
    [enc setComputePipelineState:pso];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)gw, (NSUInteger)gh, 1) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
}

static void set_buffers(id<MTLComputeCommandEncoder> enc, NSArray<id<MTLBuffer>> *bufs) {
    for (NSUInteger i = 0; i < bufs.count; i++) [enc setBuffer:bufs[i] offset:0 atIndex:i + 1];
}

void at_metal_finish(at_metal *mm) {
    @autoreleasepool {
        ATMetal *m = (__bridge ATMetal *)mm->obj;
        if (m->pending) {
            [m->pending waitUntilCompleted];
            m->gpu_start = m->pending.GPUStartTime;
            m->gpu_end = m->pending.GPUEndTime;
        }
        m->pending = nil;
        m->src_now = nil;
    }
}

uint8_t *at_metal_threshold_buffer(at_metal *mm, int64_t w, int64_t h) {
    @autoreleasepool {
        ATMetal *m = (__bridge ATMetal *)mm->obj;
        /* work that still reads the buffer: not now */
        if (atomic_load(&mm->state) == 1) return NULL;
        at_metal_finish(mm);
        atomic_store(&mm->state, 0);
        if (w < 8 || h < 8 || w > 32767 || h > 32767 || w * h > (INT64_C(1) << 26)) return NULL;
        if (!fit_buffers(m, w, h)) return NULL;
        return (uint8_t *)m->t.contents;
    }
}

bool at_metal_submit(at_metal *mm, const uint8_t *pixels, const at_metal_params *ap) {
    @autoreleasepool {
        ATMetal *m = (__bridge ATMetal *)mm->obj;
        at_metal_finish(mm);
        atomic_store(&mm->state, 0);
        mm->t_submit = CACurrentMediaTime();
        if (ap->decimate != 1 && ap->decimate != 2) return false;
        if (ap->min_cluster_pixels < 2) return false;
        int64_t w = ap->width / ap->decimate, h = ap->height / ap->decimate;
        if (w < 8 || h < 8 || w > 32767 || h > 32767 || w * h > (INT64_C(1) << 26)) return false;
        if (ap->stride * ap->height > (INT64_C(1) << 31)) return false;
        int64_t ya = ap->y0, yb = ap->y1;
        if (ya < 0 || yb >= h || yb - ya < 2) return false;
        if (!fit_buffers(m, w, h)) return false;
        id<MTLBuffer> src = nil;
        uint32_t soff = 0;
        if (pixels) {
            /* the caller's image: its pages as a buffer without a copy (they
               stay mapped until at_metal_finish), or a copy if Metal refuses */
            int64_t sbytes = ap->stride * (ap->height - 1) + ap->width;
            uintptr_t page = (uintptr_t)getpagesize();
            uintptr_t p0 = (uintptr_t)pixels & ~(page - 1);
            uintptr_t p1 = ((uintptr_t)pixels + (uintptr_t)sbytes + page - 1) & ~(page - 1);
            static int nocopy = -1;
            if (nocopy < 0) {
                const char *e = getenv("APRILTAG_GPU_COPY");
                nocopy = !(e && e[0] == '1');
            }
            if (nocopy)
                src = [m->dev newBufferWithBytesNoCopy:(void *)p0
                                                length:p1 - p0
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
            soff = (uint32_t)((uintptr_t)pixels - p0);
            if (!src) {
                if (m->src_cap < sbytes) {
                    m->src = new_buffer(m, sbytes);
                    m->src_cap = m->src ? sbytes : 0;
                    if (!m->src) return false;
                }
                memcpy(m->src.contents, pixels, (size_t)sbytes);
                src = m->src;
                soff = 0;
            }
        }
        m->src_now = src;

        /* the tile rows that the thresholds of rows ya..yb need, and the
           image rows of those tiles */
        int64_t tw = w / 4, th = h / 4;
        int64_t hy0 = ya / 4 < th ? ya / 4 : th - 1;
        int64_t hy1 = yb / 4 < th ? yb / 4 : th - 1;
        int64_t ty0 = hy0 > 0 ? hy0 - 1 : 0;
        int64_t ty1 = hy1 + 1 < th ? hy1 + 1 : th - 1;
        int64_t cy0 = 4 * ty0 < ya ? 4 * ty0 : ya;
        int64_t cy1 = 4 * ty1 + 3 > yb ? 4 * ty1 + 3 : yb;
        int64_t nr = yb - ya + 1;
        /* the table: a slot for each 8 pixels of the region */
        bool full = ya == 0;
        int64_t ts = 4096;
        while (ts < nr * w / 8) ts *= 2;
        if (ts > m->tsize) ts = m->tsize;
        m->ts_now = ts;
        KParams kp = {
            .w = (uint32_t)w,
            .h = (uint32_t)h,
            .sstride = (uint32_t)ap->stride,
            .soff = soff,
            .diff = ap->min_white_black_diff,
            .minpx = (uint32_t)ap->min_cluster_pixels,
            .yend = (uint32_t)ap->row_end,
            .tw = (uint32_t)tw,
            .th = (uint32_t)th,
            .tmask = (uint32_t)(ts - 1),
            .cap = (uint32_t)m->cap,
            .nseg = (uint32_t)m->nseg,
            .ya = (uint32_t)ya,
            .yb = (uint32_t)yb,
            .touch = ya > 0,
            .cy0 = (uint32_t)cy0,
            .ty0 = (uint32_t)ty0,
            .hy0 = (uint32_t)hy0,
            .full = full,
        };
        m->ya = ya;
        m->yb = yb;
        memset(m->state.contents, 0, 16);

        id<MTLCommandBuffer> cb = [m->queue commandBufferWithUnretainedReferences];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        NSRange tr = NSMakeRange(0, (NSUInteger)(4 * ts));
        [blit fillBuffer:m->ka range:tr value:0];
        [blit fillBuffer:m->kb range:tr value:0];
        if (full) {
            [blit fillBuffer:m->cnt range:tr value:0];
            [blit fillBuffer:m->ecnt range:tr value:0];
        }
        [blit endEncoding];

        /* APRILTAG_GPU_PROF=1: each kernel in a command buffer of its own,
           with its GPU time on stderr */
        static int prof = -1;
        if (prof < 0) {
            const char *e = getenv("APRILTAG_GPU_PROF");
            prof = e && e[0] == '1';
        }
        int step = 0;
#define STEP()                                                                                     \
    do {                                                                                           \
        if (prof) {                                                                                \
            [enc endEncoding];                                                                     \
            [cb commit];                                                                           \
            [cb waitUntilCompleted];                                                               \
            fprintf(stderr, "  gpu step %d: %.3f ms\n", step, (cb.GPUEndTime - cb.GPUStartTime) * 1e3); \
            cb = [m->queue commandBufferWithUnretainedReferences];                                 \
            enc = [cb computeCommandEncoder];                                                      \
            [enc setBytes:&kp length:sizeof kp atIndex:0];                                         \
        }                                                                                          \
        step++;                                                                                    \
    } while (0)
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setBytes:&kp length:sizeof kp atIndex:0];
        if (pixels) {
            set_buffers(enc, @[src, m->qim]);
            run2d(enc, m->pso[ap->decimate == 2 ? K_DEC2 : K_COPY], w, (cy1 < h ? cy1 : h - 1) - cy0 + 1);
            STEP();
            set_buffers(enc, @[m->qim, m->tmm]);
            run2d(enc, m->pso[K_TILE], tw, ty1 - ty0 + 1);
            STEP();
            set_buffers(enc, @[m->tmm, m->tth]);
            run2d(enc, m->pso[K_TTHR], tw, hy1 - hy0 + 1);
            STEP();
            set_buffers(enc, @[m->qim, m->tth, m->t]);
            run2d(enc, m->pso[K_THRESH], w, nr);
            STEP();
        }
        set_buffers(enc, @[m->t, m->parent, m->size, m->soff]);
        run2d(enc, m->pso[K_ROWS], m->nseg, nr);
        STEP();
        set_buffers(enc, @[m->t, m->parent, m->soff]);
        run2d(enc, m->pso[K_UNION], w, nr - 1);
        STEP();
        set_buffers(enc, @[m->t, m->parent, m->size]);
        run2d(enc, m->pso[K_LABEL], m->nseg, nr);
        STEP();
        set_buffers(enc, @[m->t, m->size, m->parent, m->okb, m->soff]);
        run2d(enc, m->pso[K_OK], w, nr);
        STEP();
        set_buffers(enc, @[m->t, m->okb, m->mask]);
        run2d(enc, m->pso[K_MASK], w, nr - 1);
        STEP();
        set_buffers(enc, @[
            m->t, m->parent, m->mask, m->ka, m->kb, m->cnt, m->ecnt, m->state, m->slots, m->pts, m->entries,
            m->seginfo, m->soff
        ]);
        run2d(enc, m->pso[K_EMIT], m->nseg, nr - 1);
        STEP();
#undef STEP
        [enc endEncoding];
        atomic_store(&mm->state, 1);
        [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
            mm->t_done = CACurrentMediaTime();
            atomic_store(&mm->state, done.status == MTLCommandBufferStatusCompleted ? 2 : 3);
        }];
        mm->t_commit = CACurrentMediaTime();
        [cb commit];
        m->pending = cb;
        return true;
    }
}

bool at_metal_done(at_metal *mm) { return atomic_load(&mm->state) >= 2; }

bool at_metal_get(at_metal *mm, at_metal_out *out) {
    @autoreleasepool {
        ATMetal *m = (__bridge ATMetal *)mm->obj;
        if (atomic_load(&mm->state) != 2) return false;
        const uint32_t *state = (const uint32_t *)m->state.contents;
        if (state[1] != 0 || state[2] > (uint32_t)m->ts_now / 2) return false;
        out->w = m->w;
        out->h = m->h;
        out->y0 = m->ya;
        out->y1 = m->yb;
        out->pts = (const uint64_t *)m->pts.contents;
        out->entries = (const uint32_t *)m->entries.contents;
        out->nentries = state[3];
        out->ecounts = (const uint32_t *)m->ecnt.contents;
        out->seginfo = (const uint32_t *)m->seginfo.contents;
        out->nseg = m->nseg;
        out->key_a = (const uint32_t *)m->ka.contents;
        out->key_b = (const uint32_t *)m->kb.contents;
        out->counts = (const uint32_t *)m->cnt.contents;
        out->slots = (const uint32_t *)m->slots.contents;
        out->nslots = state[2];
        out->npoints = state[0];
        out->table_size = m->ts_now;
        id<MTLCommandBuffer> cb = m->pending;
        double gs = cb ? cb.GPUStartTime : m->gpu_start, ge = cb ? cb.GPUEndTime : m->gpu_end;
        out->gpu_seconds = ge - gs;
        out->submit_seconds = mm->t_commit - mm->t_submit;
        out->start_seconds = gs - mm->t_commit;
        out->done_seconds = mm->t_done - mm->t_submit;
        return true;
    }
}

bool at_metal_wait(at_metal *mm, at_metal_out *out) {
    @autoreleasepool {
        ATMetal *m = (__bridge ATMetal *)mm->obj;
        if (!m->pending) return false;
        [m->pending waitUntilCompleted];
        /* the handler sets the state at about the same time */
        while (atomic_load(&mm->state) == 1) usleep(10);
        return at_metal_get(mm, out);
    }
}
