/* A program of the upstream API: the detections of each image with all
   values in hex floats, and the median time per image on stderr. Build it
   against faster3 and against apriltags-fast to compare the two
   (test/compare_faster3.sh), or to time them.
   Usage: api_dump <threads> <iters> <family> <decimate> <image.pnm>... */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "apriltag.h"
#include "tag36h11.h"
#include "tagStandard52h13.h"
#include "tagStandard41h12.h"
#include "tag25h9.h"
#include "tag16h5.h"
#include "common/image_u8.h"
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
int main(int argc, char **argv) {
    int nt = atoi(argv[1]), iters = atoi(argv[2]);
    apriltag_family_t *tf = !strcmp(argv[3], "tagStandard52h13") ? tagStandard52h13_create()
        : !strcmp(argv[3], "tagStandard41h12") ? tagStandard41h12_create()
        : !strcmp(argv[3], "tag25h9") ? tag25h9_create() : !strcmp(argv[3], "tag16h5") ? tag16h5_create() : tag36h11_create();
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family_bits(td, tf, 1);
    td->nthreads = nt;
    td->quad_decimate = atof(argv[4]);
    int nimg = argc - 5;
    image_u8_t **ims = malloc(nimg * sizeof *ims);
    for (int i = 0; i < nimg; i++) ims[i] = image_u8_create_from_pnm(argv[5 + i]);
    double *ts = malloc(iters * sizeof *ts);
    for (int it = 0; it < iters; it++) {
        double t0 = now();
        for (int i = 0; i < nimg; i++) {
            zarray_t *d = apriltag_detector_detect(td, ims[i]);
            if (it == 0) {
                printf("%s nquads %u ndet %d\n", argv[5 + i], td->nquads, zarray_size(d));
                for (int k = 0; k < zarray_size(d); k++) {
                    apriltag_detection_t *det; zarray_get(d, k, &det);
                    printf("  %s %d h%d m%a c %a %a p", det->family->name, det->id, det->hamming, det->decision_margin, det->c[0], det->c[1]);
                    for (int j = 0; j < 4; j++) printf(" %a %a", det->p[j][0], det->p[j][1]);
                    printf(" H"); for (int j = 0; j < 9; j++) printf(" %a", det->H->data[j]);
                    printf("\n");
                }
            }
            apriltag_detections_destroy(d);
        }
        ts[it] = (now() - t0) / nimg * 1e3;
    }
    qsort(ts, iters, sizeof *ts, cmpd);
    fprintf(stderr, "ms_per_image %.3f (min %.3f)\n", ts[iters / 2], ts[0]);
    apriltag_detector_destroy(td);
    return 0;
}
