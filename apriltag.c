/* The upstream AprilTag API (apriltag.h) over the fast detector
   (fast/apriltag_fast.h). The image is not copied: the fast detector
   reads im->buf in place. */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "apriltag.h"
#include "common/debug_print.h"
#include "fast/apriltag_fast.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* fast/families.c */
bool at_family_from_apriltag(const apriltag_family_t *src, at_family *dst);

apriltag_detector_t *apriltag_detector_create(void)
{
    apriltag_detector_t *td = calloc(1, sizeof(apriltag_detector_t));

    td->nthreads = 1;
    td->quad_decimate = 2.0;
    td->quad_sigma = 0.0;

    td->qtp.max_nmaxima = 10;
    td->qtp.min_cluster_pixels = 24;

    td->qtp.max_line_fit_mse = 10.0;
    td->qtp.cos_critical_rad = cos(10 * M_PI / 180);
    td->qtp.deglitch = false;
    td->qtp.min_white_black_diff = 5;

    td->tag_families = zarray_create(sizeof(apriltag_family_t *));
    td->family_bits = zarray_create(sizeof(int));

    pthread_mutex_init(&td->mutex, NULL);

    td->tp = timeprofile_create();

    td->refine_edges = true;
    td->decode_sharpening = 0.25;

    td->debug = false;

    td->fast = at_detector_create();

    return td;
}

void apriltag_detector_destroy(apriltag_detector_t *td)
{
    if (td == NULL)
        return;
    at_detector_destroy(td->fast);
    timeprofile_destroy(td->tp);
    zarray_destroy(td->tag_families);
    zarray_destroy(td->family_bits);
    pthread_mutex_destroy(&td->mutex);
    free(td);
}

apriltag_detector_t *apriltag_detector_copy(apriltag_detector_t *src)
{
    apriltag_detector_t *dst = malloc(sizeof(apriltag_detector_t));
    // the parameters; as upstream, no families
    *dst = *src;
    dst->tag_families = zarray_create(sizeof(apriltag_family_t *));
    dst->family_bits = zarray_create(sizeof(int));
    dst->tp = timeprofile_create();
    dst->wp = NULL;
    pthread_mutex_init(&dst->mutex, NULL);
    dst->fast = at_detector_create();
    dst->families_changed = false;
    return dst;
}

void apriltag_detector_add_family_bits(apriltag_detector_t *td, apriltag_family_t *fam, int bits_corrected)
{
    zarray_add(td->tag_families, &fam);
    zarray_add(td->family_bits, &bits_corrected);
    td->families_changed = true;
}

void apriltag_detector_remove_family(apriltag_detector_t *td, apriltag_family_t *fam)
{
    for (int i = 0; i < zarray_size(td->tag_families); i++) {
        apriltag_family_t *f;
        zarray_get(td->tag_families, i, &f);
        if (f == fam) {
            zarray_remove_index(td->tag_families, i, 0);
            zarray_remove_index(td->family_bits, i, 0);
            td->families_changed = true;
            return;
        }
    }
}

void apriltag_detector_clear_families(apriltag_detector_t *td)
{
    zarray_clear(td->tag_families);
    zarray_clear(td->family_bits);
    td->families_changed = true;
}

/* Give the families to the fast detector again (it makes the decode
   tables of each). */
static void sync_families(apriltag_detector_t *td)
{
    at_clear_families(td->fast);
    for (int i = 0; i < zarray_size(td->tag_families);) {
        apriltag_family_t *fam;
        int bits;
        zarray_get(td->tag_families, i, &fam);
        zarray_get(td->family_bits, i, &bits);
        at_family f;
        if (at_family_from_apriltag(fam, &f) && at_add_family(td->fast, &f, bits)) {
            i++;
            continue;
        }
        // bits_corrected above 3, or more than 16 families: remove it, so
        // that the family indices of the fast detector stay the same
        fprintf(stderr, "apriltag: cannot add family %s (bits_corrected %d), removed\n", fam->name, bits);
        zarray_remove_index(td->tag_families, i, 0);
        zarray_remove_index(td->family_bits, i, 0);
    }
    td->families_changed = false;
}

/* td->tp as upstream fills it, from the times of the stages. */
static void fill_timeprofile(timeprofile_t *tp, const at_timing *t)
{
    timeprofile_clear(tp);
    struct timeprofile_entry e;
    int64_t u = tp->utime;
    const char *names[] = {"init", "decimate", "blur/sharp", "threshold", "unionfind",
                           "make clusters", "fit quads to clusters", "decode+refinement",
                           "reconcile"};
    double secs[] = {0.0, t->decimate, t->blur, t->threshold, t->unionfind,
                     t->clusters, t->fit_quads, t->decode, t->reconcile};
    for (size_t i = 0; i < sizeof secs / sizeof secs[0]; i++) {
        u += (int64_t)(secs[i] * 1e6);
        memset(&e, 0, sizeof e);
        strncpy(e.name, names[i], sizeof e.name - 1);
        e.utime = u;
        zarray_add(tp->stamps, &e);
    }
}

bool apriltag_detector_use_gpu(apriltag_detector_t *td, int mode)
{
    return at_use_gpu(td->fast, mode == 1 ? AT_GPU_ON : mode == 2 ? AT_GPU_AUTO : AT_GPU_OFF);
}

void apriltag_detector_stage_ms(const apriltag_detector_t *td, double ms[11])
{
    const at_timing *t = &td->fast->timing;
    const double v[10] = {t->decimate, t->blur, t->threshold, t->unionfind, t->clusters,
                          t->fit_quads, t->decode, t->reconcile, t->total, t->gpu};
    for (int i = 0; i < 10; i++) ms[i] = v[i] * 1000.0;
    ms[10] = t->gpu_share;
}

zarray_t *apriltag_detector_detect(apriltag_detector_t *td, image_u8_t *im_orig)
{
    zarray_t *detections = zarray_create(sizeof(apriltag_detection_t *));

    if (zarray_size(td->tag_families) == 0) {
        debug_print("No tag families enabled\n");
        return detections;
    }

    if (im_orig->width < 8 || im_orig->height < 8) {
        debug_print("Image too small (%d x %d)\n", im_orig->width, im_orig->height);
        return detections;
    }

    if (td->families_changed)
        sync_families(td);

    at_detector *fd = td->fast;
    fd->nthreads = td->nthreads > 0 ? td->nthreads : 1;
    fd->quad_decimate = td->quad_decimate;
    fd->quad_sigma = td->quad_sigma;
    fd->refine_edges = td->refine_edges;
    fd->decode_sharpening = td->decode_sharpening;
    fd->debug = td->debug;
    fd->qtp.min_cluster_pixels = td->qtp.min_cluster_pixels;
    fd->qtp.max_nmaxima = td->qtp.max_nmaxima;
    fd->qtp.critical_rad = td->qtp.critical_rad;
    fd->qtp.cos_critical_rad = td->qtp.cos_critical_rad;
    fd->qtp.max_line_fit_mse = td->qtp.max_line_fit_mse;
    fd->qtp.min_white_black_diff = td->qtp.min_white_black_diff;
    fd->qtp.deglitch = td->qtp.deglitch != 0;

    td->tp->utime = utime_now();
    at_detect(fd, im_orig->buf, im_orig->width, im_orig->height, im_orig->stride);
    fill_timeprofile(td->tp, &fd->timing);

    td->nquads = (uint32_t)fd->nquads;
    td->nedges = 0;
    td->nsegments = 0;

    zarray_ensure_capacity(detections, (int)fd->ndetections);
    for (int64_t i = 0; i < fd->ndetections; i++) {
        const at_detection *d = &fd->detections[i];
        apriltag_detection_t *det = calloc(1, sizeof(apriltag_detection_t));
        zarray_get(td->tag_families, d->family, &det->family);
        det->id = d->id;
        det->hamming = d->hamming;
        det->decision_margin = d->decision_margin;
        det->H = matd_create(3, 3);
        memcpy(det->H->data, d->H, sizeof d->H);
        det->c[0] = d->c[0];
        det->c[1] = d->c[1];
        memcpy(det->p, d->p, sizeof det->p);
        zarray_add(detections, &det);
    }
    return detections;
}

void apriltag_detection_destroy(apriltag_detection_t *det)
{
    if (det == NULL)
        return;

    matd_destroy(det->H);
    free(det);
}

void apriltag_detections_destroy(zarray_t *detections)
{
    for (int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t *det;
        zarray_get(detections, i, &det);

        apriltag_detection_destroy(det);
    }

    zarray_destroy(detections);
}

void apriltag_detection_copy(apriltag_detection_t *src, apriltag_detection_t *dst)
{
    assert(src != NULL);
    assert(dst != NULL);

    if (dst->H) {
        matd_destroy(dst->H);
    }
    dst->H = matd_copy(src->H);

    dst->c[0] = src->c[0];
    dst->c[1] = src->c[1];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            dst->p[i][j] = src->p[i][j];
        }
    }

    dst->id = src->id;
    dst->family = src->family;
    dst->hamming = src->hamming;
    dst->decision_margin = src->decision_margin;
}

zarray_t *apriltag_detections_copy(zarray_t *detections)
{
    zarray_t *detections_copy = zarray_create(sizeof(apriltag_detection_t *));
    for (int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t *det;
        zarray_get(detections, i, &det);

        apriltag_detection_t *det_copy = calloc(1, sizeof(apriltag_detection_t));
        apriltag_detection_copy(det, det_copy);
        zarray_add(detections_copy, &det_copy);
    }

    return detections_copy;
}

image_u8_t *apriltag_to_image(apriltag_family_t *fam, uint32_t idx)
{
    assert(fam != NULL);
    assert(idx < fam->ncodes);

    uint64_t code = fam->codes[idx];

    image_u8_t *im = image_u8_create(fam->total_width, fam->total_width);

    int white_border_width = fam->width_at_border + (fam->reversed_border ? 0 : 2);
    int white_border_start = (fam->total_width - white_border_width) / 2;
    // Make 1px white border
    for (int i = 0; i < white_border_width - 1; i += 1) {
        im->buf[white_border_start * im->stride + white_border_start + i] = 255;
        im->buf[(white_border_start + i) * im->stride + fam->total_width - 1 - white_border_start] = 255;
        im->buf[(fam->total_width - 1 - white_border_start) * im->stride + white_border_start + i + 1] = 255;
        im->buf[(white_border_start + 1 + i) * im->stride + white_border_start] = 255;
    }

    int border_start = (fam->total_width - fam->width_at_border) / 2;
    for (uint32_t i = 0; i < fam->nbits; i++) {
        if (code & ((uint64_t)1 << (fam->nbits - i - 1))) {
            im->buf[(fam->bit_y[i] + border_start) * im->stride + fam->bit_x[i] + border_start] = 255;
        }
    }
    return im;
}
