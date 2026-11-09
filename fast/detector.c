/* The detector and the public API. */
#include "common.h"

at_detector *at_detector_create(void) {
    at_detector *td = (at_detector *)calloc(1, sizeof *td);
    struct at_internal *in = (struct at_internal *)calloc(1, sizeof *in);
    if (!td || !in) abort();
    td->nthreads = 1;
    td->quad_decimate = 2.0f;
    td->quad_sigma = 0.0f;
    td->refine_edges = true;
    td->decode_sharpening = 0.25;
    td->debug = false;
    td->qtp.min_cluster_pixels = 24;
    td->qtp.max_nmaxima = 10;
    td->qtp.critical_rad = 0.0f;
    td->qtp.cos_critical_rad = (float)cos(10.0 * M_PI / 180.0);
    td->qtp.max_line_fit_mse = 10.0f;
    td->qtp.min_white_black_diff = 5;
    td->qtp.deglitch = false;
    td->in = in;
    /* the detector's arrays keep their stacks while it lives */
    at_ctx_init(&in->dcx, AT_DETECTOR_NSTACKS, AT_DETECTOR_RESERVE);
    svec_take(&in->dcx, in->qd_offsets);
    svec_take(&in->dcx, in->qd_ids);
    svec_take(&in->dcx, in->qd_codes);
    svec_take(&in->dcx, in->dets);
    at_ctx_init(&in->mcx, AT_MATCH_NSTACKS, 0);
    qsm_kernel_init();
    return td;
}

void at_shutdown(at_detector *td) { pool_stop(td->in); }

void at_detector_destroy(at_detector *td) {
    if (!td) return;
    pool_stop(td->in);
    at_ctx_release(&td->in->dcx);
    at_ctx_release(&td->in->mcx);
    at_ctx_release(&td->in->icx);
    free(td->in);
    free(td);
}

bool at_add_family(at_detector *td, const at_family *fam, int bits_corrected) {
    struct at_internal *in = td->in;
    if (bits_corrected < 0 || bits_corrected > 3) return false;
    if (in->nfamilies == AT_MAX_FAMILIES) return false;
    if (fam->ncodes >= 65536 || fam->nbits <= 0 || fam->nbits > 64) return false;
    in->qd[in->nfamilies] =
        quick_decode_init(&in->mcx, fam, bits_corrected, &in->qd_offsets, &in->qd_ids, &in->qd_codes);
    in->families[in->nfamilies++] = *fam;
    return true;
}

void at_clear_families(at_detector *td) {
    struct at_internal *in = td->in;
    in->nfamilies = 0;
    in->qd_offsets.len = in->qd_ids.len = in->qd_codes.len = 0;
}

const char *at_family_name(const at_detector *td, const at_detection *d) {
    return td->in->families[d->family].name;
}

/* The family and quad-detection settings that fit_quads needs. */
static int64_t min_tag_width(const at_detector *td, bool *normal_border, bool *reversed_border) {
    const struct at_internal *in = td->in;
    *normal_border = false;
    *reversed_border = false;
    int32_t mtw = 1000000;
    for (int i = 0; i < in->nfamilies; i++) {
        const at_family *f = &in->families[i];
        if (f->width_at_border < mtw) mtw = f->width_at_border;
        *normal_border = *normal_border || !f->reversed_border;
        *reversed_border = *reversed_border || f->reversed_border;
    }
    if (td->quad_decimate > 1.0f) mtw = (int32_t)((float)mtw / td->quad_decimate);
    if (mtw < 3) mtw = 3;
    return mtw;
}

static at_detection make_detection(int64_t fi, int64_t id, int64_t hamming, int64_t rotation, float margin,
                                   const double qh[9]) {
    double theta = (double)rotation * M_PI / 2.0;
    double c = cos(theta), s = sin(theta);
    const double r[9] = {c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0};
    at_detection d;
    memset(&d, 0, sizeof d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double acc = 0.0;
            for (int k = 0; k < 3; k++) acc += qh[i * 3 + k] * r[k * 3 + j];
            d.H[i * 3 + j] = acc;
        }
    d.family = (int32_t)fi;
    d.id = (int32_t)id;
    d.hamming = (int32_t)hamming;
    d.decision_margin = margin;
    homography_project(d.H, 0.0, 0.0, &d.c[0], &d.c[1]);
    for (int i = 0; i < 4; i++) {
        double tcx = i == 1 || i == 2 ? 1.0 : -1.0;
        double tcy = i < 2 ? 1.0 : -1.0;
        homography_project(d.H, tcx, tcy, &d.p[i][0], &d.p[i][1]);
    }
    return d;
}

static void push_detection(struct at_internal *in, at_detection d) { svec_push(in->dets, d); }

/* Refine and decode every quad on this thread, appending detections in
   quad order. */
static void decode_quads(at_detector *td, Quad *quads, int64_t nquads, const ImView *view) {
    struct at_internal *in = td->in;
    at_ctx *cx = &in->mcx;
    int64_t max_tw = 1;
    for (int i = 0; i < in->nfamilies; i++) max_tw = imax(max_tw, in->families[i].total_width);
    at_mark mk = stk_enter(cx);
    double *scratch = STK_ALLOC(cx, double, 2 * max_tw * max_tw);
    QdTables tabs = {in->qd_offsets.data, in->qd_ids.data, in->qd_codes.data};
    for (int64_t qi = 0; qi < nquads; qi++) {
        Quad *q = &quads[qi];
        if (td->refine_edges) refine_edges(cx, td->quad_decimate, view, q);
        if (!quad_update_homography(q)) continue;
        for (int fi = 0; fi < in->nfamilies; fi++) {
            const at_family *fam = &in->families[fi];
            if (fam->reversed_border != q->reversed_border) continue;
            FamGeom g = fam_geom(fam);
            float margin;
            uint64_t rcode;
            if (quad_sample(&g, td->decode_sharpening, view, q, scratch, &margin, &rcode) != SAMPLE_OK) continue;
            FamLookup lk = {in->qd[fi], fam->ncodes, fam->nbits};
            int64_t id, hamming, rotation;
            unpack_lookup(quick_decode_codeword(&lk, tabs, rcode), &id, &hamming, &rotation);
            if (margin >= 0.0f && hamming < 255) push_detection(in, make_detection(fi, id, hamming, rotation, margin, q->H));
        }
    }
    stk_leave(cx, mk);
}

/* The detections of the quads that the fit jobs decoded, in quad order:
   fit task k has res->nq[k] quads from slot res->jfirst[k] on. */
static void decode_finish(at_detector *td, const DecodeOut *res) {
    struct at_internal *in = td->in;
    int64_t nf = in->nfamilies;
    for (int64_t k = 0; k < res->nfit; k++) {
        int64_t q0 = res->jfirst[k], q1 = q0 + res->nq[k];
        for (int64_t qi = q0; qi < q1; qi++) {
            if (res->hok[qi] == 0) continue;
            for (int64_t fi = 0; fi < nf; fi++) {
                int64_t i = qi * nf + fi;
                if (res->status[i] != SAMPLE_OK) continue;
                int64_t id, hamming, rotation;
                unpack_lookup(res->lookups[i], &id, &hamming, &rotation);
                float margin = res->margins[i];
                if (margin >= 0.0f && hamming < 255)
                    push_detection(in, make_detection(fi, id, hamming, rotation, margin, res->quads[qi].H));
            }
        }
    }
}

/* Stable sort of the detections by id (merge sort, scratch on cx). */
static void sort_detections(at_ctx *cx, at_detection *d, int64_t n) {
    if (n < 2) return;
    at_mark mk = stk_enter(cx);
    at_detection *tmp = STK_ALLOC(cx, at_detection, n);
    for (int64_t width = 1; width < n; width *= 2) {
        for (int64_t lo = 0; lo < n; lo += 2 * width) {
            int64_t mid = imin(lo + width, n), hi = imin(lo + 2 * width, n);
            int64_t i = lo, j = mid, o = lo;
            while (i < mid && j < hi) tmp[o++] = d[j].id < d[i].id ? d[j++] : d[i++];
            while (i < mid) tmp[o++] = d[i++];
            while (j < hi) tmp[o++] = d[j++];
        }
        memcpy(d, tmp, (size_t)n * sizeof *d);
    }
    stk_leave(cx, mk);
}

/* The rows of row y of the packed runs, its representatives and ok
   flags (for gradient_points on this thread). */
typedef struct {
    const Run *runs;
    const uint32_t *row_off;
    const uint32_t *rep;
    const uint8_t *ok;
} PackedRows;

static LabRow packed_row(void *ctx, int64_t i) {
    const PackedRows *p = (const PackedRows *)ctx;
    int64_t a = p->row_off[i], b = p->row_off[i + 1];
    return (LabRow){p->runs + a, p->rep + a, p->ok + a, b - a};
}

/* Threshold, segment, cluster and fit quads on the quad detection image
   im (w x h), into quads (a stack of its own), on this thread. */
static void quad_thresh_single(at_detector *td, const uint8_t *im, int64_t w, int64_t h, vec_quad *quads) {
    struct at_internal *in = td->in;
    at_ctx *cx = &in->mcx;
    double t0 = at_clock();
    uint8_t *t = in->buf_t;
    vec_run runs;
    svec_take(cx, runs);
    uint32_t *row_off = STK_ALLOC(cx, uint32_t, h + 1);
    threshold_image(cx, im, w, h, w, td->qtp.min_white_black_diff, t);
    if (td->qtp.deglitch) deglitch(cx, t, w, h);
    rle_image(cx, t, w, h, &runs, row_off);
    if (td->debug) at_write_pgm("debug_threshold.pnm", t, w, h);
    double t1 = at_clock();
    td->timing.threshold = t1 - t0;

    int64_t nruns = row_off[h];
    int64_t nnodes = nruns + h;
    UfNode *uf = in->buf_uf;
    uint8_t *edge = STK_ALLOC(cx, uint8_t, 2 * h);
    row_edges(t, w, h, edge);
    connected_components(cx, uf, edge, w, h, runs.data, row_off, 1);
    double t2 = at_clock();
    td->timing.unionfind = t2 - t1;

    ClusterTable ct;
    ct_init(cx, &ct, cluster_map_size(w, h, 1));
    bool normal_border, reversed_border;
    int64_t tag_width = min_tag_width(td, &normal_border, &reversed_border);
    uint32_t *rep = STK_ALLOC(cx, uint32_t, nnodes);
    uint8_t *ok = STK_ALLOC(cx, uint8_t, nnodes);
    resolve_range(uf, 0, nnodes, td->qtp.min_cluster_pixels, rep, ok);
    if (h >= 3) {
        /* rows 1..h-2, each against the row below */
        PackedRows pr = {runs.data, row_off + 1, rep, ok};
        gradient_points(cx, &ct, t + w, w, 1, h - 2, packed_row, &pr, rep + nruns + 1, ok + nruns + 1);
    }
    vec_pt cpts;
    vec_u32 cstart;
    svec_take(cx, cpts);
    svec_take(cx, cstart);
    order_clusters(cx, &ct, &cpts, &cstart);
    double t3 = at_clock();
    td->timing.clusters = t3 - t2;
    fit_quads(cx, &td->qtp, cpts.data, cstart.data, cstart.len - 1, PT_PAD, im, w, h, tag_width, normal_border,
              reversed_border, quads);
    td->timing.fit_quads = at_clock() - t3;
}

void at_detect(at_detector *td, const uint8_t *pixels, int64_t width, int64_t height, int64_t stride) {
    struct at_internal *in = td->in;
    at_ctx *cx = &in->mcx;
    in->dets.len = 0;
    td->detections = in->dets.data;
    td->ndetections = 0;
    td->nquads = 0;
    memset(&td->timing, 0, sizeof td->timing);
    if (in->nfamilies == 0) return;
    if (width < 8 || height < 8) return;
    int64_t nthreads = imax(1, td->nthreads);
    double t_start = at_clock();
    /* the quad detection image: decimated, and blurred or sharpened */
    int64_t qw = width, qh = height;
    if (td->quad_decimate > 1.0f) decimated_size(width, height, td->quad_decimate, &qw, &qh);
    /* the stacks, sized for this image (the image arrays hold nothing
       between calls) */
    size_t qpx = (size_t)(qw + 2) * (size_t)(qh + 2);
    at_ctx_fit(&in->mcx, AT_MATCH_NSTACKS, qpx * AT_MATCH_BYTES_PER_PIXEL);
    at_ctx_fit(&in->icx, AT_IMAGE_NSTACKS, qpx * AT_IMAGE_BYTES_PER_PIXEL);
    in->icx.sp = 1;
    in->buf_qim = STK_TAKE(&in->icx, uint8_t);
    in->buf_t = STK_TAKE(&in->icx, uint8_t);
    in->buf_uf = STK_TAKE(&in->icx, UfNode);
    if (nthreads > 1) pool_ensure(in, nthreads, qpx * AT_WORKER_BYTES_PER_PIXEL);
    bool pooled = nthreads > 1 && in->pool != NULL;
    at_mark mk = stk_enter(cx);
    uint8_t *qim = in->buf_qim;
    /* with the pool, the threshold jobs decimate by 2 themselves (when
       nothing else changes the decimated image first) */
    bool fuse_decimate = pooled && !td->qtp.deglitch && !td->debug && td->quad_decimate == 2.0f &&
                         blur_kernel_size(td->quad_sigma) <= 1;
    if (fuse_decimate) {
        /* done by the threshold jobs */
    } else if (td->quad_decimate > 1.0f) {
        decimate_rows(pixels, stride, td->quad_decimate, qim, qw, 0, qh);
    } else {
        for (int64_t y = 0; y < qh; y++) memcpy(qim + y * qw, pixels + y * stride, (size_t)qw);
    }
    double t_decimate = at_clock();
    td->timing.decimate = t_decimate - t_start;

    /* Without decimation the C library blurs the caller's image in place,
       so the decoder then samples the blurred image. The caller's image
       is left alone here, and the decoder samples the blurred copy. */
    bool decimated = td->quad_decimate > 1.0f;
    bool blurred = blur_or_sharpen(cx, qim, qw, qh, td->quad_sigma);
    double t_blur = at_clock();
    td->timing.blur = t_blur - t_decimate;
    if (td->debug) at_write_pgm("debug_preprocess.pnm", qim, qw, qh);

    /* the image that the decode samples */
    ImView view = {pixels, stride, width, height};
    if (blurred && !decimated) view = (ImView){qim, width, width, height};
    vec_quad quads;
    svec_take(cx, quads);
    double t_quads;
    if (pooled) {
        DecodeOut res;
        quad_thresh_pooled(td, qim, qw, qh, nthreads, fuse_decimate ? pixels : NULL, stride, &view, &quads, &res);
        t_quads = at_clock();
        /* the fit jobs scaled, refined and sampled the quads */
        td->nquads = quads.len;
        decode_finish(td, &res);
    } else {
        quad_thresh_single(td, qim, qw, qh, &quads);
        t_quads = at_clock();
        if (td->quad_decimate > 1.0f) {
            float f = td->quad_decimate;
            for (int64_t i = 0; i < quads.len; i++)
                for (int j = 0; j < 4; j++) {
                    quads.data[i].p[j][0] *= f;
                    quads.data[i].p[j][1] *= f;
                }
        }
        td->nquads = quads.len;
        decode_quads(td, quads.data, quads.len, &view);
    }
    double t_decode = at_clock();
    td->timing.decode = t_decode - t_quads;

    reconcile(in->dets.data, &in->dets.len);
    sort_detections(cx, in->dets.data, in->dets.len);
    stk_leave(cx, mk);
    td->detections = in->dets.data;
    td->ndetections = in->dets.len;
    double t_end = at_clock();
    td->timing.reconcile = t_end - t_decode;
    td->timing.total = t_end - t_start;
}
