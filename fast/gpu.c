/* The clusters of the GPU stage (metal.h), in the order of the CPU
   stage.

   The GPU names a component by the index of one of its pixels. The CPU
   stage names it by the root of its union-find, and orders the clusters
   by a hash of the two roots. So the union-find still runs on the CPU
   (at the same time as the GPU); here each GPU label becomes a root, and
   the points are put in cluster order. The points of a cluster come from
   the GPU in emission order (row, pixel, neighbor), as on the CPU. */
#include "common.h"

/* The points of the segments s0..s1 of src into dst. The GPU's segments
   are a few points each: a segment of 4 points or less is copied as 4
   points (no branch on its length), so dst has room for 3 points more,
   and src can be read up to 3 points past a segment. */
AT_INLINE void gather_points(Pt *restrict dst, const Pt *restrict src, const uint64_t *ssrc, const uint32_t *scnt,
                             int64_t s0, int64_t s1) {
    int64_t at = 0;
    for (int64_t s = s0; s < s1; s++) {
        int64_t c = scnt[s];
        if (c <= 4)
            memcpy(dst + at, src + ssrc[s], 4 * sizeof(Pt));
        else
            memcpy(dst + at, src + ssrc[s], (size_t)c * sizeof(Pt));
        at += c;
    }
}

#if defined(AT_METAL)
#include "metal.h"

/* The union-find root of the component of the pixel `label`. */
static uint32_t label_root(const UfNode *uf, RunRows rr, int64_t w, int64_t vcol_base, uint32_t label) {
    int64_t y = label / w, x = label % w;
    int64_t node;
    if (x == w - 1) {
        node = vcol_base + y;
    } else {
        /* the last run that starts at or before x */
        const Run *r = row_runs(rr, y);
        int64_t lo = 0, hi = row_count(rr, y);
        while (hi - lo > 1) {
            int64_t mid = (lo + hi) / 2;
            if (r[mid].start <= x)
                lo = mid;
            else
                hi = mid;
        }
        node = rr.row_off[y] + lo;
    }
    uint32_t r = (uint32_t)node;
    while (uf[r].parent != r) r = uf[r].parent;
    return r;
}

/* APRILTAG_GPU_FORCE=<part> (tests): with threads, the GPU gets this
   part of the image (0.05 to 0.95), and the cluster tasks wait for it.
   0 without the variable. */
static double gpu_force(void) {
    static double force = -1.0;
    if (force < 0.0) {
        const char *e = getenv("APRILTAG_GPU_FORCE");
        double v = e ? atof(e) : 0.0;
        force = v < 0.05 ? 0.0 : v > 0.95 ? 0.95 : v;
    }
    return force;
}

AT_INLINE bool root_ok(const UfNode *uf, uint32_t root, int64_t min_px) {
    return (int64_t)uf[root].size + 1 >= min_px;
}

/* The clusters of rows y0..y0+n-1 from the GPU stage into the table
   (ids and counts), and the entries to keep into ct->segs, in emission
   order: the first point in the GPU's points (32 bits), the count (9
   bits), the slot in the table (from bit 41). The GPU labeled its
   region on its own, so it could not know the size of each component:
   the entries of clusters with a component that is too small are left
   out here, and each flagged entry is tested (see k_mask of metal.m).
   Returns the number of points. */
static int64_t gpu_band_scan(at_ctx *cx, ClusterTable *ct, const at_metal_out *g, RunRows rr, const UfNode *uf,
                             int64_t vcol_base, int64_t min_px, int64_t y0, int64_t n) {
    int64_t w = g->w;
    at_mark mk = stk_enter(cx);
    /* the table slot of each GPU slot (-1: a component is too small), in
       a small cache */
    enum { SC = 2048 };
    uint32_t *sc_slot = STK_ALLOC(cx, uint32_t, SC);
    int32_t *sc_li = STK_ALLOC(cx, int32_t, SC);
    memset(sc_slot, 0xff, SC * 4);
    const Pt *gpts = (const Pt *)g->pts;
    uint64_t *segs = ct->segs.data;
    int64_t ns = ct->segs.len;
    int64_t total = 0;
    for (int64_t y = y0; y < y0 + n; y++) {
        const uint32_t *si = g->seginfo + 2 * y * g->nseg;
        for (int64_t sg = 0; sg < g->nseg; sg++, si += 2) {
            const uint32_t *e = g->entries + 2 * (int64_t)si[0];
            for (uint32_t i = 0; i < si[1]; i++, e += 2) {
                uint32_t slot = e[1] & 0x7fffffu;
                uint32_t c = (e[1] >> 24) + 1;
                uint32_t hsh = (slot * 2654435761u) >> 21;
                if (sc_slot[hsh] != slot) {
                    uint32_t ra = label_root(uf, rr, w, vcol_base, g->key_a[slot] - 1);
                    uint32_t rb = label_root(uf, rr, w, vcol_base, g->key_b[slot] - 1);
                    uint64_t id = ra < rb ? ((uint64_t)rb << 32) + ra : ((uint64_t)ra << 32) + rb;
                    sc_slot[hsh] = slot;
                    sc_li[hsh] = root_ok(uf, ra, min_px) && root_ok(uf, rb, min_px) ? (int32_t)cluster_find(ct, id) : -1;
                }
                int32_t li = sc_li[hsh];
                if (li < 0) continue;
                if (e[1] & 0x800000u) {
                    /* the (-1,1) point of pixel (x, py): not valid if the
                       pixel to the left has the (1,1) point of the pair */
                    Pt q = gpts[e[0]];
                    int64_t x = (q.x + 1) / 2, py = (q.y - 1) / 2;
                    uint32_t r1 = label_root(uf, rr, w, vcol_base, (uint32_t)(py * w + x - 1));
                    uint32_t r2 = label_root(uf, rr, w, vcol_base, (uint32_t)((py + 1) * w + x));
                    if (root_ok(uf, r1, min_px) && root_ok(uf, r2, min_px)) continue;
                }
                ct->counts.data[li] += c;
                total += c;
                segs[ns++] = (uint64_t)e[0] | ((uint64_t)c << 32) | ((uint64_t)li << 41);
            }
        }
    }
    ct->segs.len = ns;
    stk_leave(cx, mk);
    return total;
}

/* Put the points of the entries of gpu_band_scan at pos[slot] of out,
   and move that position on (scatter_points for the GPU's points). With
   pad, an entry of at most 4 points copies 4 points. */
static void gpu_band_copy(const ClusterTable *ct, const at_metal_out *g, uint32_t *pos, Pt *out, bool pad) {
    const Pt *gpts = (const Pt *)g->pts;
    for (int64_t i = 0; i < ct->segs.len; i++) {
        uint64_t sg = ct->segs.data[i];
        uint32_t src = (uint32_t)sg;
        int64_t c = (int64_t)((sg >> 32) & 0x1ff);
        int64_t li = (int64_t)(sg >> 41);
        int64_t p = pos[li];
        if (pad && c <= 4)
            memcpy(out + p, gpts + src, 4 * sizeof(Pt));
        else
            memcpy(out + p, gpts + src, (size_t)c * sizeof(Pt));
        pos[li] = (uint32_t)(p + c);
    }
}

/* The clusters of the GPU stage, in the order of cluster_order for
   nbuckets buckets. Cluster c has the points cstart[c]..cstart[c + 1]
   (cstart: a stack of its own); they are the segments sfirst[c]..
   sfirst[c + 1], each a start in the points of the GPU (ssrc) and a
   count (scnt). The three arrays are scratch of cx. Returns the number
   of clusters. */
static int64_t gpu_clusters(at_ctx *cx, const at_metal_out *g, RunRows rr, const UfNode *uf, int64_t vcol_base,
                            int64_t nbuckets, vec_u32 *cstart, uint32_t **sfirst_out, uint32_t **scnt_out,
                            uint64_t **ssrc_out) {
    int64_t w = g->w, h = g->h;
    int64_t n = g->nslots;
    uint32_t *sfirst = STK_ALLOC(cx, uint32_t, n + 1);
    uint32_t *scnt = STK_ALLOC(cx, uint32_t, g->nentries + 1);
    uint64_t *ssrc = STK_ALLOC(cx, uint64_t, g->nentries + 1);
    at_mark mk = stk_enter(cx);
    uint64_t *ids = STK_ALLOC(cx, uint64_t, n);
    uint32_t *bucket = STK_ALLOC(cx, uint32_t, n);
    uint32_t *bfirst = STK_ZALLOC(cx, uint32_t, nbuckets + 1);
    /* many clusters share a component (the background): a small cache of
       the roots */
    enum { RC = 1024 };
    uint32_t *rc_label = STK_ALLOC(cx, uint32_t, 2 * RC);
    uint32_t *rc_root = rc_label + RC;
    memset(rc_label, 0xff, RC * 4);
    for (int64_t i = 0; i < n; i++) {
        uint32_t s = g->slots[i];
        uint32_t ab[2] = {g->key_a[s] - 1, g->key_b[s] - 1};
        uint32_t r[2];
        for (int k = 0; k < 2; k++) {
            uint32_t hsh = (ab[k] * 2654435761u) >> 22;
            if (rc_label[hsh] != ab[k]) {
                rc_label[hsh] = ab[k];
                rc_root[hsh] = label_root(uf, rr, w, vcol_base, ab[k]);
            }
            r[k] = rc_root[hsh];
        }
        uint64_t id = r[0] < r[1] ? ((uint64_t)r[1] << 32) + r[0] : ((uint64_t)r[0] << 32) + r[1];
        ids[i] = id;
        bucket[i] = (uint32_t)(u64hash_2(id) & (uint64_t)(nbuckets - 1));
        bfirst[bucket[i] + 1] += 1;
    }
    for (int64_t b = 0; b < nbuckets; b++) bfirst[b + 1] += bfirst[b];
    /* by bucket, then by id (insertion sort: a bucket has few clusters) */
    uint32_t *order = STK_ALLOC(cx, uint32_t, n);
    uint32_t *fill = STK_ALLOC(cx, uint32_t, nbuckets);
    memcpy(fill, bfirst, (size_t)nbuckets * 4);
    for (int64_t i = 0; i < n; i++) {
        int64_t first = bfirst[bucket[i]];
        int64_t j = fill[bucket[i]]++;
        while (j > first && ids[order[j - 1]] > ids[i]) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = (uint32_t)i;
    }
    /* where the next segment of each slot goes */
    uint32_t *pos = STK_ALLOC(cx, uint32_t, g->table_size);
    uint32_t at = 0, eat = 0;
    cstart->len = 0;
    for (int64_t i = 0; i < n; i++) {
        uint32_t s = g->slots[order[i]];
        svec_push(*cstart, at);
        sfirst[i] = eat;
        pos[s] = eat;
        at += g->counts[s];
        eat += g->ecounts[s];
    }
    svec_push(*cstart, at);
    sfirst[n] = eat;
    const uint32_t *si = g->seginfo + 2 * g->nseg;
    for (int64_t k = g->nseg; k < (h - 1) * g->nseg; k++, si += 2) {
        const uint32_t *e = g->entries + 2 * (int64_t)si[0];
        for (uint32_t i = 0; i < si[1]; i++, e += 2) {
            uint32_t at2 = pos[e[1] & 0x7fffffu]++;
            ssrc[at2] = e[0];
            scnt[at2] = (e[1] >> 24) + 1;
        }
    }
    stk_leave(cx, mk);
    *sfirst_out = sfirst;
    *scnt_out = scnt;
    *ssrc_out = ssrc;
    return n;
}

/* fit_quads for clusters in segments (fit_task of pool.c). */
static void fit_quads_segments(at_ctx *cx, const at_quad_thresh_params *qtp, const Pt *pts, const uint32_t *cstart,
                               int64_t nclusters, const uint32_t *sfirst, const uint32_t *scnt, const uint64_t *ssrc,
                               const uint8_t *im, int64_t w, int64_t h, int64_t tag_width, bool normal_border,
                               bool reversed_border, vec_quad *quads) {
    int64_t min_px = qtp->min_cluster_pixels;
    int64_t max_px = 2 * (2 * w + 2 * h);
    for (int64_t c = 0; c < nclusters; c++) {
        int64_t n = (int64_t)cstart[c + 1] - cstart[c];
        if (n < min_px || n > max_px) continue;
        at_mark mk = stk_enter(cx);
        int64_t s0 = sfirst[c], s1 = sfirst[c + 1];
        const Pt *cp = pts + ssrc[s0];
        if (s1 - s0 > 1) {
            Pt *gp = STK_ALLOC(cx, Pt, n + 3);
            gather_points(gp, pts, ssrc, scnt, s0, s1);
            cp = gp;
        }
        Quad q;
        memset(&q, 0, sizeof q);
        if (fit_quad(cx, qtp, cp, n, im, w, h, tag_width, normal_border, reversed_border, &q)) svec_push(*quads, q);
        stk_leave(cx, mk);
    }
}

static bool gpu_choose(struct at_internal *in);

/* Start the GPU stage for this image, if the settings allow it. row_end
   is the end of the rows that the union-find tasks cover. */
static bool gpu_submit(at_detector *td, const uint8_t *pixels, int64_t width, int64_t height, int64_t stride,
                       int64_t row_end, int64_t y0, int64_t y1) {
    struct at_internal *in = td->in;
    if (!in->metal) return false;
    if (td->quad_decimate != 2.0f && td->quad_decimate > 1.0f) return false;
    if (blur_kernel_size(td->quad_sigma) > 1 || td->qtp.deglitch || td->debug) return false;
    at_metal_params mp = {
        .width = width,
        .height = height,
        .stride = stride,
        .decimate = td->quad_decimate == 2.0f ? 2 : 1,
        .min_white_black_diff = td->qtp.min_white_black_diff,
        .min_cluster_pixels = td->qtp.min_cluster_pixels,
        .row_end = row_end,
        .y0 = y0,
        .y1 = y1,
    };
    return at_metal_submit(in->metal, pixels, &mp);
}

/* With the pool, the GPU is one more worker of the cluster stage: it
   makes the points of the last bands (cluster tasks) of the image, on
   its own, while the threads do the threshold, the union-find and the
   bands before them. A band task that starts when the GPU is done takes
   the GPU's points; if the GPU is not done, the task makes the points
   itself. So the GPU can only take work away.

   gpu_frac is the part of the bands that the GPU gets. It grows while
   all its bands are used, and gets smaller when the threads were there
   first. If the GPU was not even done at the end of the image, the
   detector had to wait for it: the next images then go without the GPU
   (twice as many each time). */
static uint8_t *gpu_threshold_buffer(at_detector *td, int64_t qw, int64_t qh) {
    struct at_internal *in = td->in;
    in->gpu_pending = false;
    in->gpu_t = false;
    /* AT_GPU_AUTO: with threads the GPU does not make an image faster
       (see README.md), so only AT_GPU_ON uses it */
    if (!in->metal || in->gpu_mode != AT_GPU_ON || td->qtp.deglitch || td->debug) return NULL;
    if (td->qtp.min_cluster_pixels < 2 || qh < 8) return NULL;
    if (in->gpu_skip > 0 && gpu_force() == 0.0) {
        in->gpu_skip--;
        return NULL;
    }
    uint8_t *t = at_metal_threshold_buffer(in->metal, qw, qh);
    in->gpu_t = t != NULL;
    return t;
}

/* Start the GPU on the last bands, after the threshold stage. */
static void gpu_submit_region(at_detector *td, int64_t qw, int64_t qh, int64_t nthreads, int64_t row_end) {
    struct at_internal *in = td->in;
    if (!in->gpu_t) return;
    if (in->gpu_frac <= 0.0) in->gpu_frac = 0.35;
    if (gpu_force() > 0.0) in->gpu_frac = gpu_force();
    /* the bands of clusters_fit_parallel */
    int64_t nrows = qh - 2;
    int64_t chunk = 1 + nrows / (TASKS_PER_THREAD * nthreads);
    int64_t ntasks = (nrows + chunk - 1) / chunk;
    int64_t r = (int64_t)(in->gpu_frac * (double)ntasks + 0.5);
    r = imax(1, imin(r, ntasks - 1));
    if (ntasks < 2) return;
    in->gpu_k0 = ntasks - r;
    at_metal_params mp = {
        .width = qw,
        .height = qh,
        .stride = qw,
        .decimate = 1,
        .min_white_black_diff = td->qtp.min_white_black_diff,
        .min_cluster_pixels = td->qtp.min_cluster_pixels,
        .row_end = row_end,
        .y0 = 1 + in->gpu_k0 * chunk,
        .y1 = qh - 1,
    };
    in->gpu_pending = at_metal_submit(in->metal, NULL, &mp);
}

/* The end of the cluster stage: `used` of the GPU's `bands` took its
   points. */
static void gpu_region_result(at_detector *td, int64_t bands, int64_t used) {
    struct at_internal *in = td->in;
    td->timing.gpu_share = (double)used / (double)(bands + in->gpu_k0);
    in->gpu_used = used > 0;
    if (used == bands)
        in->gpu_frac = in->gpu_frac + 0.03 > 0.9 ? 0.9 : in->gpu_frac + 0.03;
    else
        in->gpu_frac -= 0.01 + 0.05 * (double)(bands - used) / (double)bands;
    if (in->gpu_frac < 0.04) in->gpu_frac = 0.04;
}

/* The end of an image. One thread: let go of the caller's image. With
   the pool the GPU can still work (it reads only its own buffers): the
   next image then goes without it. If no band took the GPU's points
   though it has the smallest part, the next images go without the GPU
   (twice as many each time). */
static void gpu_finish(at_detector *td, bool pooled) {
    struct at_internal *in = td->in;
    if (!in->metal) return;
    if (!pooled) {
        at_metal_finish(in->metal);
        in->gpu_pending = false;
        return;
    }
    if (!in->gpu_pending) return;
    if (in->gpu_used) {
        in->gpu_backoff = 0;
    } else if (in->gpu_frac <= 0.05) {
        in->gpu_backoff = in->gpu_backoff ? imin(2 * in->gpu_backoff, 256) : 2;
        in->gpu_skip = in->gpu_backoff;
    }
    at_metal_out g;
    if (at_metal_get(in->metal, &g)) {
        td->timing.gpu = g.gpu_seconds;
        static int trace = -1;
        if (trace < 0) trace = getenv("APRILTAG_GPU_TRACE") != NULL;
        if (trace && (in->gpu_frames & 63) == 63)
            fprintf(stderr, "region: rows %lld..%lld of %lld  submit %.3f  start %.3f  gpu %.3f  done at %.3f ms  share %.2f frac %.2f\n",
                    (long long)g.y0, (long long)g.y1, (long long)g.h, g.submit_seconds * 1e3, g.start_seconds * 1e3,
                    g.gpu_seconds * 1e3, g.done_seconds * 1e3, td->timing.gpu_share, in->gpu_frac);
    }
    in->gpu_frames++;
    in->gpu_pending = false;
}

bool at_use_gpu(at_detector *td, at_gpu_mode mode) {
    struct at_internal *in = td->in;
    if (mode == AT_GPU_OFF) {
        at_metal_destroy(in->metal);
        in->metal = NULL;
        in->gpu_mode = AT_GPU_OFF;
        return false;
    }
    if (!in->metal) in->metal = at_metal_create();
    in->gpu_mode = in->metal ? mode : AT_GPU_OFF;
    in->gpu_frames = 0;
    in->gpu_frac = 0.0;
    in->gpu_skip = in->gpu_backoff = 0;
    in->gpu_ema[0] = in->gpu_ema[1] = 0.0;
    return in->metal != NULL;
}

/* Whether this image uses the GPU stage. AT_GPU_AUTO uses the path that
   was faster for the last images (a moving average, see gpu_record),
   and tries the other path now and then: after GPU_PROBE_MIN images at
   first, then after more each time the choice stays the same
   (GPU_PROBE_MAX at most). The detections do not depend on the path. */
#define GPU_PROBE_MIN 16
#define GPU_PROBE_MAX 1024
static bool gpu_choose(struct at_internal *in) {
    if (in->gpu_mode != AT_GPU_AUTO) return in->gpu_mode == AT_GPU_ON;
    int64_t f = in->gpu_frames++;
    /* the first images: the GPU twice (its first image compiles and
       maps), then the CPU */
    if (f < 3) {
        in->gpu_best = false;
        in->gpu_probe = GPU_PROBE_MIN;
        in->gpu_next_probe = 3 + GPU_PROBE_MIN;
        return f < 2;
    }
    bool gpu = in->gpu_ema[1] < in->gpu_ema[0];
    if (gpu != in->gpu_best) {
        in->gpu_best = gpu;
        in->gpu_probe = GPU_PROBE_MIN;
        in->gpu_next_probe = f + in->gpu_probe;
    }
    if (f >= in->gpu_next_probe) {
        /* a path that is more than twice as slow is tried much less: its
           image takes long */
        double lo = in->gpu_ema[gpu ? 1 : 0], hi = in->gpu_ema[gpu ? 0 : 1];
        in->gpu_probe *= hi > 2.0 * lo ? 8 : 2;
        if (in->gpu_probe > GPU_PROBE_MAX) in->gpu_probe = GPU_PROBE_MAX;
        in->gpu_next_probe = f + in->gpu_probe;
        return !gpu;
    }
    return gpu;
}

/* Record the cluster stage of an image for AT_GPU_AUTO: its time for
   each boundary point (the time of an image depends too much on what is
   in it). The GPU path also costs the fit stage about 1.4 ns a point,
   on all threads together: its points are in many small segments. */
static void gpu_record(struct at_internal *in, bool gpu, double seconds, int64_t points, int64_t nthreads) {
    if (in->gpu_mode != AT_GPU_AUTO || points < 1000) return;
    /* the GPU's first image is not typical */
    if (gpu && in->gpu_frames == 1) return;
    double v = seconds / (double)points;
    if (gpu) v += 1.4e-9 / (double)nthreads;
    double *e = &in->gpu_ema[gpu ? 1 : 0];
    *e = *e == 0.0 ? v : 0.75 * *e + 0.25 * v;
}

#else

bool at_use_gpu(at_detector *td, at_gpu_mode mode) {
    (void)td;
    (void)mode;
    return false;
}

#endif
