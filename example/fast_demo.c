/* Command-line demo: detect tags in PGM images, with the options of the
   C library's apriltag_demo (and of the Goose port's demo). */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fast/apriltag_fast.h"

static void usage(void) {
    puts("Usage: apriltag_fast_demo [options] <input files>\n"
         "  -h, --help             Show this help\n"
         "  -d, --debug            Enable debugging output (slow)\n"
         "  -q, --quiet            Reduce output\n"
         "  -f, --family <name>    Tag family to use [tag36h11]\n"
         "  -i, --iters <n>        Repeat processing on input set this many times [1]\n"
         "  -t, --threads <n>      Use this many CPU threads [1]\n"
         "  -a, --hamming <n>      Detect tags with up to this many bit errors [1]\n"
         "  -x, --decimate <f>     Decimate input image by this factor [2.0]\n"
         "  -b, --blur <f>         Apply low-pass blur to input; negative sharpens [0.0]\n"
         "  -0, --refine-edges     Spend more time trying to align edges of tags [true]\n"
         "      --save-detections <file>  Save detections to this file as TSV\n"
         "      --hex-detections <file>   Save detections with exact hex floats\n"
         "      --timing                  Print the time of each stage\n"
         "      --deglitch, --min-cluster-pixels <n>, --max-nmaxima <n>, --min-white-black-diff <n>,\n"
         "      --max-line-fit-mse <f>, --sharpening <f>, --critical-angle <deg>: detector parameters");
}

static double parse_f64(const char *s) {
    char *end;
    double v = strtod(s, &end);
    if (*s == 0 || *end != 0) {
        printf("bad number: %s\n", s);
        exit(1);
    }
    return v;
}

static int64_t parse_i64(const char *s) {
    char *end;
    long long v = strtoll(s, &end, 10);
    if (*s == 0 || *end != 0) {
        printf("bad integer: %s\n", s);
        exit(1);
    }
    return v;
}

/* A growable output buffer (outside the timed region). */
typedef struct {
    char *data;
    size_t len, cap;
} Buf;

static void bprintf(Buf *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
#include <stdarg.h>
static void bprintf(Buf *b, const char *fmt, ...) {
    for (;;) {
        va_list ap;
        va_start(ap, fmt);
        int n = vsnprintf(b->data ? b->data + b->len : NULL, b->data ? b->cap - b->len : 0, fmt, ap);
        va_end(ap);
        if (n < 0) abort();
        if (b->data && b->len + (size_t)n < b->cap) {
            b->len += (size_t)n;
            return;
        }
        b->cap = (b->cap + (size_t)n + 1) * 2;
        b->data = realloc(b->data, b->cap);
        if (!b->data) abort();
    }
}

static void write_buf(const char *path, const Buf *b) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        printf("couldn't write %s\n", path);
        exit(1);
    }
    if (b->len) fwrite(b->data, 1, b->len, f);
    fclose(f);
}

int main(int argc, char **argv) {
    const char *family = "tag36h11";
    int64_t iters = 1, threads = 1, hamming = 1;
    double decimate = 2.0, blur = 0.0;
    bool refine = true, quiet = false, debug = false, deglitch = false, timing = false;
    const char *tsv_path = NULL, *hex_path = NULL;
    int64_t mcp = 24, maxnm = 10, mwbd = 5;
    double mse = 10.0, sharpening = 0.25, crit_deg = 10.0;
    const char **inputs = calloc((size_t)argc, sizeof *inputs);
    int ninputs = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (printf("missing value for %s\n", a), exit(1), ""))
#define IS(s) (strcmp(a, s) == 0)
        if (IS("-h") || IS("--help")) {
            usage();
            return 0;
        } else if (IS("-d") || IS("--debug")) debug = true;
        else if (IS("-q") || IS("--quiet")) quiet = true;
        else if (IS("-f") || IS("--family")) family = NEXT();
        else if (IS("-i") || IS("--iters")) iters = parse_i64(NEXT());
        else if (IS("-t") || IS("--threads")) threads = parse_i64(NEXT());
        else if (IS("-a") || IS("--hamming")) hamming = parse_i64(NEXT());
        else if (IS("-x") || IS("--decimate")) decimate = parse_f64(NEXT());
        else if (IS("-b") || IS("--blur")) blur = parse_f64(NEXT());
        else if (IS("-0") || IS("--refine-edges")) refine = false;
        else if (IS("--save-detections")) tsv_path = NEXT();
        else if (IS("--deglitch")) deglitch = true;
        else if (IS("--timing")) timing = true;
        else if (IS("--min-cluster-pixels")) mcp = parse_i64(NEXT());
        else if (IS("--max-nmaxima")) maxnm = parse_i64(NEXT());
        else if (IS("--min-white-black-diff")) mwbd = parse_i64(NEXT());
        else if (IS("--max-line-fit-mse")) mse = parse_f64(NEXT());
        else if (IS("--sharpening")) sharpening = parse_f64(NEXT());
        else if (IS("--critical-angle")) crit_deg = parse_f64(NEXT());
        else if (IS("--hex-detections")) hex_path = NEXT();
        else inputs[ninputs++] = a;
#undef NEXT
#undef IS
    }

    at_detector *td = at_detector_create();
    const at_family *fam = at_family_by_name(family);
    if (!fam) {
        puts("Unrecognized tag family name. Use e.g. \"tag36h11\".");
        exit(-1);
    }
    if (!at_add_family(td, fam, (int)hamming)) {
        puts("\"hamming\" parameter is out-of-range.");
        exit(-1);
    }
    td->quad_decimate = (float)decimate;
    td->quad_sigma = (float)blur;
    td->nthreads = (int32_t)threads;
    td->debug = debug;
    td->refine_edges = refine;
    td->qtp.deglitch = deglitch;
    td->qtp.min_cluster_pixels = (int32_t)mcp;
    td->qtp.max_nmaxima = (int32_t)maxnm;
    td->qtp.min_white_black_diff = (int32_t)mwbd;
    td->qtp.max_line_fit_mse = (float)mse;
    td->decode_sharpening = sharpening;
    td->qtp.cos_critical_rad = (float)cos(crit_deg * M_PI / 180.0);

    Buf tsv = {0}, hex = {0};
    bprintf(&tsv, "image\titer\tid\thamming\tmargin\tcx\tcy\tx0\ty0\tx1\ty1\tx2\ty2\tx3\ty3\n");
    double total_time = 0.0;
    int64_t total_quads = 0;
    at_timing st = {0};
    uint8_t *pixels = NULL;
    int64_t w = 0, h = 0, cap = 0;
    for (int64_t iter = 0; iter < iters; iter++) {
        for (int k = 0; k < ninputs; k++) {
            const char *path = inputs[k];
            /* one image buffer for all frames (as the Goose demo) */
            if (!at_read_pnm_into(path, &w, &h, &pixels, &cap)) {
                printf("couldn't load %s\n", path);
                continue;
            }
            if (!quiet) printf("loading %s\n", path);
            at_detect(td, pixels, w, h, w);
            total_time += td->timing.total;
            st.decimate += td->timing.decimate;
            st.blur += td->timing.blur;
            st.threshold += td->timing.threshold;
            st.unionfind += td->timing.unionfind;
            st.clusters += td->timing.clusters;
            st.fit_quads += td->timing.fit_quads;
            st.decode += td->timing.decode;
            st.reconcile += td->timing.reconcile;
            total_quads += td->nquads;
            for (int64_t j = 0; j < td->ndetections; j++) {
                const at_detection *d = &td->detections[j];
                if (!quiet)
                    printf("detection: id %d, hamming %d, margin %g\n", d->id, d->hamming, (double)d->decision_margin);
                bprintf(&tsv, "%s\t%lld\t%d\t%d\t%.4f\t%.4f\t%.4f", path, (long long)iter, d->id, d->hamming,
                        (double)d->decision_margin, d->c[0], d->c[1]);
                for (int q = 0; q < 4; q++) bprintf(&tsv, "\t%.4f\t%.4f", d->p[q][0], d->p[q][1]);
                bprintf(&tsv, "\n");
                bprintf(&hex, "%s %d %d %a %a %a", path, d->id, d->hamming, (double)d->decision_margin, d->c[0],
                        d->c[1]);
                for (int q = 0; q < 4; q++) bprintf(&hex, " %a %a", d->p[q][0], d->p[q][1]);
                for (int q = 0; q < 9; q++) bprintf(&hex, " %a", d->H[q]);
                bprintf(&hex, "\n");
            }
        }
    }
    free(pixels);
    if (tsv_path) write_buf(tsv_path, &tsv);
    if (hex_path) write_buf(hex_path, &hex);
    double n = (double)(iters * ninputs);
    printf("images %d iters %lld quads %lld ms_per_image %.3f\n", ninputs, (long long)iters,
           (long long)(total_quads / iters), total_time * 1000.0 / n);
    if (timing) {
        const char *names[8] = {"decimate", "blur", "threshold", "unionfind", "clusters", "fit_quads", "decode",
                                "reconcile"};
        double v[8] = {st.decimate, st.blur, st.threshold, st.unionfind, st.clusters, st.fit_quads, st.decode,
                       st.reconcile};
        for (int k = 0; k < 8; k++) printf("  %s %.3f ms\n", names[k], v[k] * 1000.0 / n);
    }
    at_detector_destroy(td);
    free(tsv.data);
    free(hex.data);
    free(inputs);
    return 0;
}
