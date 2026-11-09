/* The native API of apriltags-fast. apriltag.h (the API of the upstream
   AprilTag library) is a layer over it. This API gives the detections in
   an array that the detector keeps, without an allocation per detection.

       at_detector *td = at_detector_create();
       at_add_family(td, at_tag36h11(), 2);
       at_detect(td, pixels, width, height, stride);
       for (int64_t i = 0; i < td->ndetections; i++) { ... td->detections[i] ... }
       at_detector_destroy(td);
*/
#ifndef APRILTAG_FAST_H
#define APRILTAG_FAST_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A tag family: the code words and the bit layout of a tag. The codes
   are static data (the built-in families) or an array that the caller
   keeps alive while the detector uses the family. */
typedef struct at_family {
    char name[32];
    const uint64_t *codes;
    int64_t ncodes;
    int32_t nbits;
    int32_t h;
    int32_t width_at_border;
    int32_t total_width;
    bool reversed_border;
    /* The bit locations, in tag coordinates, most significant bit first. */
    int32_t bit_x[64];
    int32_t bit_y[64];
} at_family;

/* Parameters of the quad detection stage. The defaults are those of the
   C library. */
typedef struct at_quad_thresh_params {
    int32_t min_cluster_pixels; /* 24 */
    int32_t max_nmaxima;        /* 10 */
    float critical_rad;         /* not used */
    float cos_critical_rad;     /* cos(10 degrees) */
    float max_line_fit_mse;     /* 10.0 */
    int32_t min_white_black_diff; /* 5 */
    bool deglitch;              /* false */
} at_quad_thresh_params;

/* One detected tag. `family` is the index of the family in the detector,
   in the order the families were added. */
typedef struct at_detection {
    int32_t family;
    int32_t id;
    int32_t hamming;
    float decision_margin;
    /* The 3x3 homography (row major) from tag coordinates ([-1, 1]) to
       image pixels. */
    double H[9];
    /* The center of the tag, in pixels. */
    double c[2];
    /* The corners, counter-clockwise, from tag coordinate (-1, -1). */
    double p[4][2];
} at_detection;

/* Time spent in each stage of the last detection, in seconds. */
typedef struct at_timing {
    double decimate, blur, threshold, unionfind, clusters, fit_quads, decode, reconcile, total;
} at_timing;

struct at_internal;

typedef struct at_detector {
    /* Threads to use. 1 runs everything on the calling thread. */
    int32_t nthreads;
    /* Detect quads on an image decimated by this factor (1.5, or an
       integer). Corners are still refined on the full image. */
    float quad_decimate;
    /* Gaussian blur of the decimated image (sigma), negative sharpens. */
    float quad_sigma;
    /* Refine the quad edges on the full-resolution image. */
    bool refine_edges;
    /* How much to sharpen the sampled bits before decoding. */
    double decode_sharpening;
    /* Write debug images (debug_preprocess.pnm, debug_threshold.pnm). */
    bool debug;
    at_quad_thresh_params qtp;

    /* The result of the last at_detect(), sorted by id. */
    at_detection *detections;
    int64_t ndetections;
    int64_t nquads;
    at_timing timing;

    struct at_internal *in;
} at_detector;

/* A detector with the default parameters and no families. */
at_detector *at_detector_create(void);
/* Stops the worker threads and frees the detector. */
void at_detector_destroy(at_detector *td);

/* Add a tag family. `bits_corrected` is the maximum number of bit errors
   to correct (0 to 3). Returns false if it is out of range or there are
   already 16 families. The detector keeps a copy of *fam (not of the
   codes). */
bool at_add_family(at_detector *td, const at_family *fam, int bits_corrected);
void at_clear_families(at_detector *td);
/* The name of the family a detection belongs to. */
const char *at_family_name(const at_detector *td, const at_detection *d);

/* Detect tags in a gray image: width x height pixels, rows `stride`
   bytes apart. The input is never changed. */
void at_detect(at_detector *td, const uint8_t *pixels, int64_t width, int64_t height, int64_t stride);

/* Stop the worker threads. They start again on the next at_detect. */
void at_shutdown(at_detector *td);

/* The built-in families (made from the upstream family files on the
   first call). */
const at_family *at_tag16h5(void);
const at_family *at_tag25h9(void);
const at_family *at_tag36h10(void);
const at_family *at_tag36h11(void);
const at_family *at_tagCircle21h7(void);
const at_family *at_tagCircle49h12(void);
const at_family *at_tagCustom48h12(void);
const at_family *at_tagStandard41h12(void);
const at_family *at_tagStandard52h13(void);
/* A built-in family by name ("tag36h11", ...), or NULL. */
const at_family *at_family_by_name(const char *name);

/* Read a binary PGM or PPM (color becomes gray as in the C library).
   Returns false on a bad file. *pixels is malloc'd (width bytes a row);
   free it with free(). */
bool at_read_pnm(const char *path, int64_t *width, int64_t *height, uint8_t **pixels);
/* at_read_pnm into a buffer that the caller keeps between images: *pixels
   holds *cap bytes (NULL and 0 at first); a larger image replaces it with
   a larger buffer. Free it with free(). */
bool at_read_pnm_into(const char *path, int64_t *width, int64_t *height, uint8_t **pixels, int64_t *cap);
bool at_write_pgm(const char *path, const uint8_t *pixels, int64_t width, int64_t height);

#ifdef __cplusplus
}
#endif

#endif
