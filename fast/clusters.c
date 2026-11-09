/* Boundary clusters: the points between each pair of adjacent black and
   white components, grouped by the pair.

   This mirrors the C library's run-driven do_gradient_clusters: the
   points are emitted in the same (y, x, neighbor) order with the same
   lazy component-size gates. The C library splits the rows into tasks
   and merges the per-task hash tables; the merged result is the same as
   one sweep over all rows, with the clusters ordered by (hash bucket,
   id) for the bucket count the task size selects. That order is
   reproduced here. */
#include "common.h"

/* The cluster table: a chained hash table with the C library's bucket
   function, the emitted points, and the segments of points that go to
   one cluster: (slot << 32) | count, in emission order. */
typedef struct {
    vec_u32 heads;
    vec_u64 ids;
    vec_u32 nexts;
    vec_u32 counts;
    vec_pt pts;
    vec_u64 segs;
    uint64_t mask;
    uint64_t last_id;
    int64_t last_slot;
} ClusterTable;

/* A table with nbuckets buckets (a power of two). The heads are bumped,
   the five growable arrays take a stack each, in cx. */
static void ct_init(at_ctx *cx, ClusterTable *ct, int64_t nbuckets) {
    ct->heads.data = STK_ALLOC(cx, uint32_t, nbuckets);
    ct->heads.len = nbuckets;
    memset(ct->heads.data, 0xff, (size_t)nbuckets * 4);
    svec_take(cx, ct->ids);
    svec_take(cx, ct->nexts);
    svec_take(cx, ct->counts);
    svec_take(cx, ct->pts);
    svec_take(cx, ct->segs);
    ct->mask = (uint64_t)(nbuckets - 1);
    ct->last_id = 0;
    ct->last_slot = -1;
}

AT_INLINE uint64_t u64hash_2(uint64_t x) { return (x * 2654435761u) >> 32; }

/* The slot of the cluster with this id; a new cluster if there is none. */
static int64_t cluster_find(ClusterTable *ct, uint64_t id) {
    int64_t bucket = (int64_t)(u64hash_2(id) & ct->mask);
    uint32_t e = ct->heads.data[bucket];
    while (e != 0xffffffffu && ct->ids.data[e] != id) e = ct->nexts.data[e];
    if (e != 0xffffffffu) return e;
    int64_t slot = ct->ids.len;
    svec_push(ct->ids, id);
    svec_push(ct->nexts, ct->heads.data[bucket]);
    svec_push(ct->counts, 0);
    ct->heads.data[bucket] = (uint32_t)slot;
    return slot;
}

static int64_t cluster_slot_id(ClusterTable *ct, uint64_t id) {
    if (ct->last_slot >= 0 && ct->last_id == id) return ct->last_slot;
    int64_t slot = cluster_find(ct, id);
    ct->last_id = id;
    ct->last_slot = slot;
    return slot;
}

/* Resolve nodes [lo, hi): the representative of each, and whether its
   component is big enough (size + 1 >= min_cluster_pixels). The nodes
   before lo must be resolved. The C library resolves these lazily; they
   do not change during clustering, so resolving them up front gives the
   same. */
static void resolve_range(const UfNode *uf, int64_t lo, int64_t hi, int64_t min_cluster_pixels, uint32_t *rep,
                          uint8_t *ok) {
    for (int64_t i = lo; i < hi; i++) {
        /* the nodes before i are resolved: stop at the first one */
        int64_t j = i;
        uint32_t root;
        for (;;) {
            if (j < i) {
                root = rep[j];
                break;
            }
            uint32_t p = uf[j].parent;
            if (p == j) {
                root = p;
                break;
            }
            j = p;
        }
        rep[i] = root;
        ok[i] = (int64_t)uf[root].size + 1 >= min_cluster_pixels ? 1 : 0;
    }
}

/* The bucket count of the C library's per-task cluster tables. */
static int64_t cluster_map_size(int64_t w, int64_t h, int64_t nthreads) {
    int64_t sz = h - 1;
    int64_t chunksize = 1 + sz / (TASKS_PER_THREAD * nthreads);
    int64_t n = 1024;
    while (n < chunksize * w / 64 && n < 65536) n <<= 1;
    return n;
}

AT_INLINE void fill8(uint32_t *lab, uint8_t *okb, uint32_t r, uint8_t o) {
#if defined(__ARM_NEON)
    /* clang makes the loop 16 scalar stores on arm64 */
    uint32x4_t vr = vdupq_n_u32(r);
    vst1q_u32(lab, vr);
    vst1q_u32(lab + 4, vr);
    vst1_u8(okb, vdup_n_u8(o));
#else
    for (int k = 0; k < 8; k++) {
        lab[k] = r;
        okb[k] = o;
    }
#endif
}

/* The labels of one row (w pixels) from its runs; pixels without a run
   (127) get ok 0. The last column is the virtual node. lab and okb have
   w + 16 entries. */
static void label_row(const Run *runs, int64_t nruns, const uint32_t *rep, const uint8_t *ok, uint32_t vr,
                      uint8_t vo, int64_t w, uint32_t *lab, uint8_t *okb) {
    /* The runs are short (camera noise), so a fill loop per run would
       mispredict at each run end. Each run instead writes a block of 8
       (more only if it is longer), then 8 zeros of ok after it, for a
       possible gap. The runs go left to right, so the next run writes
       over the extra entries. */
    memset(okb, 0, (size_t)w);
    for (int64_t i = 0; i < nruns; i++) {
        int64_t s = runs[i].start;
        int64_t e = (int64_t)runs[i].end + 1;
        uint32_t r = rep[i];
        uint8_t o = ok[i];
        fill8(lab + s, okb + s, r, o);
        for (int64_t x = s + 8; x < e; x += 8) fill8(lab + x, okb + x, r, o);
        memset(okb + e, 0, 8);
    }
    lab[w - 1] = vr;
    okb[w - 1] = vo;
}

/* 1 if one of a and b is black and the other white (0, 127 or 255). */
AT_INLINE uint8_t opp(uint8_t a, uint8_t b) { return (a ^ b) == 255 ? 1 : 0; }

/* The point mask of each pixel x in 1..w-2 of row a against row b: bit 0
   (1,0), bit 1 (0,1), bit 2 (-1,1), bit 3 (1,1). e[0] has only the (1,0)
   bit, and e[w-1] is 0. */
SIMD_VOID(point_masks,
          (const uint8_t *restrict a, const uint8_t *restrict b, const uint8_t *restrict oa,
           const uint8_t *restrict ob, uint8_t *restrict e, int64_t w),
          (a, b, oa, ob, e, w)) {
    int64_t n = w - 2;
    const uint8_t *a0 = a + 1, *ap = a + 2, *am = a;
    const uint8_t *b0 = b + 1, *bp = b + 2, *bm = b;
    const uint8_t *oa0 = oa + 1, *oap = oa + 2, *oam = oa;
    const uint8_t *ob0 = ob + 1, *obp = ob + 2, *obm = ob;
    uint8_t *out = e + 1;
    for (int64_t x = 0; x < n; x++) {
        uint8_t v = a0[x];
        uint8_t c10 = opp(v, ap[x]) & oap[x];
        uint8_t c01 = opp(v, b0[x]) & ob0[x];
        uint8_t c11 = opp(v, bp[x]) & obp[x];
        uint8_t cm1 = opp(v, bm[x]) & obm[x];
        /* the (1,1) point of the pixel to the left */
        uint8_t p11 = opp(am[x], b0[x]) & oam[x] & ob0[x];
        out[x] = (uint8_t)((c10 | (c01 << 1) | ((cm1 & (p11 ^ 1)) << 2) | (c11 << 3)) & (uint8_t)(oa0[x] * 15));
    }
    /* x = 1 has no pixel to the left in the C loop */
    if ((a[1] ^ b[0]) == 255 && oa[1] != 0 && ob[0] != 0) e[1] |= 4;
    e[0] = opp(a[0], a[1]) != 0 && oa[0] != 0 && oa[1] != 0 ? 1 : 0;
    e[w - 1] = 0;
}

/* p[i] = e[2i] | e[2i+1] << 4: the masks of 16 pixels in a u64. */
SIMD_VOID(pack_nibbles, (const uint8_t *restrict e, uint8_t *restrict p, int64_t n), (e, p, n)) {
    for (int64_t i = 0; i < n; i++) p[i] = (uint8_t)(e[2 * i] | (e[2 * i + 1] << 4));
}

AT_INLINE Pt pt_of_bits(uint64_t v) {
    Pt p;
    memcpy(&p, &v, sizeof p);
    return p;
}

/* Emit the points of row y in pixel order. lab holds the labels of row
   y from ca and of row y+1 from cb. e holds the masks of pixels 2i and
   2i+1 in byte i (pack_nibbles), with 8 bytes of zero slack. Each set
   bit is one point: bit j of pixel x is neighbor (dx, dy) = (1,0),
   (0,1), (-1,1), (1,1) for j = 0..3. The bits are taken one at a time
   (in pixel, then neighbor order), 16 pixels a word, with dx and dy from
   j, so that there is no branch for each neighbor. */
static void emit_row(ClusterTable *ct, const uint8_t *a, const uint32_t *lab, int64_t ca, int64_t cb,
                     const uint8_t *e, int64_t w, int64_t y) {
    /* the last cluster and its points since the last flush to counts */
    uint64_t lid = ct->last_id;
    int64_t lslot = ct->last_slot;
    uint32_t lcnt = 0;
    /* per neighbor j: the label offset dy * (cb - ca) + dx, and
       (2y + dy) << 16 | (2px + dx) less 2px (dx = -1 borrows from the y
       part, which dy = 1 gives back) */
    int64_t dd = cb - ca;
    const int64_t offs[4] = {1, dd, dd - 1, dd + 1};
    int64_t rowb = (2 * y) << 16;
    const int64_t dxy[4] = {rowb + 1, rowb + 65536, rowb + 65535, rowb + 65537};
    /* gx | gy << 16 (16-bit each) for neighbor j and pixel value a (0 or
       255) at index 2j + (a & 1): vdiff = 255 - 2a is 255 or -255 */
    static const uint32_t gtab[8] = {0x000000ff, 0x0000ff01, 0x00ff0000, 0xff010000,
                                     0x00ffff01, 0xff0100ff, 0x00ff00ff, 0xff01ff01};
    const uint32_t *la = lab + ca;
    /* the points and segments grow in place: their lengths stay in locals */
    Pt *pts = ct->pts.data;
    int64_t np = ct->pts.len;
    uint64_t *segs = ct->segs.data;
    int64_t ns = ct->segs.len;
    uint32_t *counts = ct->counts.data;
    for (int64_t x = 0; x < w; x += 16) {
        uint64_t d = load64(e + x / 2);
        while (d != 0) {
            int64_t bit = __builtin_ctzll(d);
            d &= d - 1;
            int64_t px = x + (bit >> 2);
            int64_t j = bit & 3;
            int64_t xy = 2 * px + dxy[j];
            uint32_t g = gtab[(j << 1) | (a[px] & 1)];
            uint32_t rep0 = la[px];
            uint32_t rep1 = la[px + offs[j]];
            uint64_t id = rep0 < rep1 ? ((uint64_t)rep1 << 32) + rep0 : ((uint64_t)rep0 << 32) + rep1;
            if (id != lid || lslot < 0) {
                if (lcnt != 0) {
                    counts[lslot] += lcnt;
                    segs[ns++] = ((uint64_t)lslot << 32) | lcnt;
                }
                lcnt = 0;
                lslot = cluster_find(ct, id);
                lid = id;
            }
            lcnt += 1;
            /* x, y in the low 32 bits, gx, gy in the high */
            pts[np++] = pt_of_bits(((uint64_t)xy & 0xffffffffu) | ((uint64_t)g << 32));
        }
    }
    ct->pts.len = np;
    if (lcnt != 0) {
        counts[lslot] += lcnt;
        segs[ns++] = ((uint64_t)lslot << 32) | lcnt;
    }
    ct->segs.len = ns;
    ct->last_id = lid;
    ct->last_slot = lslot;
}

/* Emit the boundary points of rows ybase..ybase+n-1 (each row against
   the row below it) into the table. t holds rows ybase..ybase+n of the
   thresholded image (w bytes a row). Row ybase+i has nrun[i] runs at
   runs[i], with representatives rep[i] and ok flags okr[i], for i in
   0..n+1; vrep and vok are the virtual nodes (last column) of rows
   ybase..ybase+n.

   The C library visits each pixel (x, y), x in 1..w-2, and its neighbors
   (1,0), (0,1), (-1,1), (1,1) in that order. A neighbor gives a point if
   the two pixels are black and white, and both components are big
   enough. The (-1,1) point is left out if the pixel to the left gave a
   (1,1) point (the same pixel pair). The run-driven C code of faster3
   also gives the (1,0) point of x = 0 (a run that ends at column 0).
   Here, a SIMD pass makes a 4-bit mask per pixel from per-pixel labels,
   and only pixels with a point are visited. */
typedef struct {
    const Run *runs;
    const uint32_t *rep;
    const uint8_t *ok;
    int64_t n;
} LabRow;

typedef LabRow (*LabRowFn)(void *ctx, int64_t i);

static void gradient_points(at_ctx *cx, ClusterTable *ct, const uint8_t *t, int64_t w, int64_t ybase, int64_t n,
                            LabRowFn rowfn, void *ctx, const uint32_t *vrep, const uint8_t *vok) {
    /* labels of rows y (a) and y+1 (b): the representative of the pixel
       and 1 if its component is big enough, else 0; each row has 16
       entries of slack for label_row */
    at_mark mk = stk_enter(cx);
    int64_t ws = w + 16;
    uint32_t *lab = STK_ZALLOC(cx, uint32_t, 2 * ws);
    uint8_t *okb = STK_ZALLOC(cx, uint8_t, 2 * ws);
    uint8_t *e = STK_ZALLOC(cx, uint8_t, w + 8);
    /* the masks of two pixels in one byte (low nibble first), with 8
       bytes of zero slack */
    int64_t np = (w + 1) / 2;
    uint8_t *ep = STK_ALLOC(cx, uint8_t, np + 8);
    memset(ep + np, 0, 8);
    LabRow r = rowfn(ctx, 0);
    label_row(r.runs, r.n, r.rep, r.ok, vrep[0], vok[0], w, lab, okb);
    for (int64_t ly = 0; ly < n; ly++) {
        /* row y is in half ly % 2, row y+1 in the other */
        int64_t ca = (ly & 1) * ws;
        int64_t cb = ws - ca;
        r = rowfn(ctx, ly + 1);
        label_row(r.runs, r.n, r.rep, r.ok, vrep[ly + 1], vok[ly + 1], w, lab + cb, okb + cb);
        const uint8_t *ra = t + ly * w;
        const uint8_t *rb = ra + w;
        point_masks(ra, rb, okb + ca, okb + cb, e, w);
        pack_nibbles(e, ep, np);
        emit_row(ct, ra, lab, ca, cb, ep, w, ybase + ly);
    }
    stk_leave(cx, mk);
}

/* The unused points after each cluster of order_clusters. */
#define PT_PAD 16

/* Put the points of each segment at pos[slot] of out, and move that
   position on. With pad, the clusters are followed by PT_PAD unused
   points: a segment of at most PT_PAD points copies PT_PAD points (no
   branch on its length). The extra points go to the later points of the
   cluster, which later segments write, or to the unused points. */
static void scatter_points(const ClusterTable *ct, uint32_t *pos, Pt *out, bool pad) {
    const Pt *src = ct->pts.data;
    int64_t nsrc = ct->pts.len;
    int64_t at = 0;
    for (int64_t i = 0; i < ct->segs.len; i++) {
        uint64_t sg = ct->segs.data[i];
        int64_t s = (int64_t)(sg >> 32);
        int64_t n = (int64_t)(sg & 0xffffffffu);
        int64_t p = pos[s];
        if (pad && n <= PT_PAD && at + PT_PAD <= nsrc)
            memcpy(out + p, src + at, PT_PAD * sizeof(Pt));
        else
            memcpy(out + p, src + at, (size_t)n * sizeof(Pt));
        pos[s] = (uint32_t)(p + n);
        at += n;
    }
}

/* The slots in output order: by hash bucket, then by id, into order
   (ids.len entries). */
static void cluster_order(const ClusterTable *ct, uint32_t *order) {
    int64_t n = 0;
    for (int64_t b = 0; b < ct->heads.len; b++) {
        int64_t first = n;
        for (uint32_t e = ct->heads.data[b]; e != 0xffffffffu; e = ct->nexts.data[e]) order[n++] = e;
        uint32_t *grp = order + first;
        int64_t ng = n - first;
        /* insertion sort by id */
        for (int64_t i = 1; i < ng; i++) {
            uint32_t v = grp[i];
            int64_t j = i;
            while (j > 0 && ct->ids.data[grp[j - 1]] > ct->ids.data[v]) {
                grp[j] = grp[j - 1];
                j--;
            }
            grp[j] = v;
        }
    }
}

/* Group the points by cluster and order the clusters as the C library
   does. cpts (a stack of its own) gets the points of all clusters, each
   followed by PT_PAD unused points, and cstart (a stack of its own) each
   cluster's start (one more entry than clusters). Clusters without
   points are dropped. */
static void order_clusters(at_ctx *cx, const ClusterTable *ct, vec_pt *cpts, vec_u32 *cstart) {
    at_mark mk = stk_enter(cx);
    int64_t nslots = ct->ids.len;
    uint32_t *order = STK_ALLOC(cx, uint32_t, nslots);
    cluster_order(ct, order);
    uint32_t *pos = STK_ALLOC(cx, uint32_t, nslots);
    uint32_t at = 0;
    cstart->len = 0;
    for (int64_t i = 0; i < nslots; i++) {
        uint32_t s = order[i];
        if (ct->counts.data[s] == 0) continue;
        svec_push(*cstart, at);
        pos[s] = at;
        at += ct->counts.data[s] + PT_PAD;
    }
    svec_push(*cstart, at);
    cpts->len = at;
    scatter_points(ct, pos, cpts->data, true);
    stk_leave(cx, mk);
}

/* As order_clusters, but rank (nslots entries) gets the output index of
   each slot (0xffffffff for an empty one), and cstart the starts without
   gaps. */
static void cluster_ranks(at_ctx *cx, const ClusterTable *ct, uint32_t *rank, vec_u32 *cstart) {
    at_mark mk = stk_enter(cx);
    int64_t nslots = ct->ids.len;
    uint32_t *order = STK_ALLOC(cx, uint32_t, nslots);
    cluster_order(ct, order);
    memset(rank, 0xff, (size_t)nslots * 4);
    uint32_t at = 0, n = 0;
    cstart->len = 0;
    for (int64_t i = 0; i < nslots; i++) {
        uint32_t s = order[i];
        if (ct->counts.data[s] == 0) continue;
        svec_push(*cstart, at);
        rank[s] = n++;
        at += ct->counts.data[s];
    }
    svec_push(*cstart, at);
    stk_leave(cx, mk);
}
