/* The GPU (Metal) cluster stage. Apple platforms only.

   The GPU makes the boundary points of the thresholded image, grouped
   by pairs of components, from the caller's image: it decimates,
   thresholds, labels the connected components, and emits the points.
   at_metal_submit starts this and returns at once; the CPU does its own
   threshold and union-find at the same time. at_metal_wait gives the
   result. The CPU then orders the clusters with its union-find, so the
   output is the same as that of the CPU stage. */
#ifndef AT_METAL_H
#define AT_METAL_H

#include <stdbool.h>
#include <stdint.h>

typedef struct at_metal at_metal;

#define AT_METAL_SEG 64

typedef struct {
    int64_t width, height, stride; /* the caller's image */
    int32_t decimate;              /* 1 or 2 */
    int32_t min_white_black_diff;
    int32_t min_cluster_pixels;
    /* rows [1, row_end) are connected to the row above them (the rows
       that the union-find tasks cover) */
    int64_t row_end;
    /* The region: the points of rows y0..y1-1 of the quad detection
       image (each row against the row below it). All rows: 0 and its
       height - 1. With y0 > 0 the GPU labels rows y0..y1 on their own: a
       component can then go on above row y0, so the entries can have
       points of components that are too small, and flagged entries. */
    int64_t y0, y1;
} at_metal_params;

typedef struct {
    int64_t w, h; /* the quad detection image */
    int64_t y0, y1; /* the region */
    /* The points, and the entries: an entry is a run of points of one
       cluster, entries[2e] its first point and entries[2e + 1] its slot
       (23 low bits), a flag (bit 23) and its count less one (8 high
       bits). A flagged entry is one (-1,1) point of the pixel (x, y): it
       is valid only if the components of the pixels (x - 1, y) and (x,
       y + 1) are not both big enough. Segment k of
       row y (nseg segments of AT_METAL_SEG pixels a row) has
       seginfo[2i + 1] entries from seginfo[2i] on, i = y * nseg + k. The
       entries of a segment are in emission order, and the segments in
       order give the emission order of the image. */
    const uint64_t *pts;
    const uint32_t *entries;
    const uint32_t *seginfo;
    int64_t nseg, nentries;
    /* the cluster table: slot s is the pair of component labels key_a[s]
       - 1 and key_b[s] - 1 (a label is the index of a pixel of the
       component) with counts[s] points in ecounts[s] entries; slots
       lists the nslots slots in use */
    const uint32_t *key_a, *key_b, *counts, *ecounts, *slots;
    int64_t nslots, npoints, table_size;
    double gpu_seconds;
    /* for traces: the time of the submit call, from the commit to the
       start of the GPU, and from the submit to the end */
    double submit_seconds, start_seconds, done_seconds;
} at_metal_out;

/* NULL without a Metal device. */
at_metal *at_metal_create(void);
void at_metal_destroy(at_metal *m);
/* Start the stage for an image; it first waits for the work before it.
   Returns false if it cannot run (the caller then uses the CPU stage).
   The image must stay as it is until at_metal_finish. */
bool at_metal_submit(at_metal *m, const uint8_t *pixels, const at_metal_params *p);
/* Whether the work ended (any thread can ask). */
bool at_metal_done(at_metal *m);
/* The result of work that ended. Returns false if it failed or a buffer
   was too small. The result stays until the next at_metal_submit. */
bool at_metal_get(at_metal *m, at_metal_out *out);
/* Wait for the work, then at_metal_get. */
bool at_metal_wait(at_metal *m, at_metal_out *out);
/* Wait for the work, and let go of the image. */
void at_metal_finish(at_metal *m);

#endif
