/* Connected components of the thresholded image, over row runs.

   The union-find is indexed by global run index. The last image column
   holds no runs; it gets one virtual node per row at vcol_base + y. A
   node holds its parent and size together: a root's size is then in the
   cache line that the walk to the root loaded. The task split for a
   given thread count is reproduced exactly, since the representatives
   it produces decide the order of the clusters. */
#include "common.h"

#define TASKS_PER_THREAD 6

/* Connect two nodes: walk both root chains together, compress both
   paths fully, then link the smaller tree under the larger one. */
AT_INLINE void uf_connect(UfNode *uf, uint32_t a0, uint32_t b0) {
    uint32_t ra = a0, rb = b0;
    for (;;) {
        uint32_t pa = uf[ra].parent, pb = uf[rb].parent;
        if (pa == ra && pb == rb) break;
        ra = pa;
        rb = pb;
    }
    if (ra == rb) return;
    uint32_t a = a0;
    while (uf[a].parent != ra) {
        uint32_t t = uf[a].parent;
        uf[a].parent = ra;
        a = t;
    }
    uint32_t b = b0;
    while (uf[b].parent != rb) {
        uint32_t t = uf[b].parent;
        uf[b].parent = rb;
        b = t;
    }
    uint32_t asize = uf[ra].size + 1;
    uint32_t bsize = uf[rb].size + 1;
    if (asize > bsize) {
        uf[rb].parent = ra;
        uf[ra].size += bsize;
    } else {
        uf[ra].parent = rb;
        uf[rb].size += asize;
    }
}

/* Initialize the nodes of one row's runs (n runs, first node base). */
AT_INLINE void uf_init_row(UfNode *uf, const Run *runs, int64_t n, int64_t base) {
    for (int64_t i = 0; i < n; i++) {
        uf[base + i].parent = (uint32_t)(base + i);
        uf[base + i].size = (uint32_t)(runs[i].end - runs[i].start);
    }
}

/* A white run that ends at column w-2 joins the virtual node of the row
   above if that row ends in white at w-1 but not at w-2. Only the last
   run of a row can end there, so this runs after the run loop (the
   same connect order as the test at each run: it is the last connect). */
AT_INLINE void last_column(UfNode *uf, int64_t w, int64_t y, const Run *cur, int64_t ncur, int64_t cur_base,
                           int64_t vcol_base, uint8_t above_last, uint8_t above_prev) {
    if (ncur > 0 && cur[ncur - 1].end == w - 2 && cur[ncur - 1].v == 255 && above_last == 255 && above_prev != 255)
        uf_connect(uf, (uint32_t)(cur_base + ncur - 1), (uint32_t)(vcol_base + y - 1));
}

/* Whether a row has a gap between runs (where a run of 127 was). */
AT_INLINE bool has_gap(const Run *r, int64_t n) {
    int f = 0;
    for (int64_t i = 1; i < n; i++) f |= r[i].start != r[i - 1].end + 1;
    return f != 0;
}

/* connect_rows for rows with any run values. */
AT_INLINE void connect_rows_any(UfNode *uf, const uint8_t *edge, int64_t w, int64_t y, const Run *prev, int64_t nprev,
                             int64_t prev_base, const Run *cur, int64_t ncur, int64_t cur_base, int64_t vcol_base) {
    uint8_t above_last = edge[2 * y - 1];
    uint8_t above_prev = edge[2 * y - 2];
    int64_t j = 0;
    for (int64_t i = 0; i < ncur; i++) {
        int64_t a0 = cur[i].start;
        int64_t a1 = cur[i].end;
        uint8_t v = cur[i].v;
        uint32_t head_a = (uint32_t)(cur_base + i);
        while (j < nprev && (int64_t)prev[j].end + 1 < a0) j++;
        for (int64_t k = j; k < nprev && prev[k].start <= a1 + 1; k++) {
            Run pk = prev[k];
            if (pk.v != v) continue;
            int64_t b0 = pk.start, b1 = pk.end;
            uint32_t head_b = (uint32_t)(prev_base + k);
            int64_t lo = imax(imax(a0, b0), 1);
            int64_t hi = imin(a1, b1);
            if (lo <= hi) {
                uf_connect(uf, head_a, head_b);
            } else if (v == 255) {
                int64_t xl = imax(imax(a0, b0 + 1), 1);
                if (xl <= imin(a1, b1 + 1)) {
                    uf_connect(uf, head_a, head_b);
                } else {
                    int64_t xr = imax(imax(a0, b0 - 1), 1);
                    if (xr <= imin(a1, b1 - 1)) uf_connect(uf, head_a, head_b);
                }
            }
        }
    }
    last_column(uf, w, y, cur, ncur, cur_base, vcol_base, above_last, above_prev);
}

/* Union the runs of row y (cur, the first with node number cur_base)
   with the runs of row y-1 (prev, from prev_base). edge[2*y] and
   edge[2*y+1] are the last two pixels of row y of the thresholded
   image. */
AT_INLINE void connect_rows(UfNode *uf, const uint8_t *edge, int64_t w, int64_t y, const Run *prev, int64_t nprev,
                            int64_t prev_base, const Run *cur, int64_t ncur, int64_t cur_base, int64_t vcol_base) {
    if (has_gap(prev, nprev) || has_gap(cur, ncur)) {
        connect_rows_any(uf, edge, w, y, prev, nprev, prev_base, cur, ncur, cur_base, vcol_base);
        return;
    }
    /* The runs of 127 are not stored. Without them, the runs of a row
       follow each other with no gap and alternate between 0 and 255, so
       the runs above of the same value as a run are every second one.
       The same pairs are connected, in the same order, as in
       connect_rows_any. */
    uint8_t above_last = edge[2 * y - 1];
    uint8_t above_prev = edge[2 * y - 2];
    int64_t j = 0;
    for (int64_t i = 0; i < ncur; i++) {
        int64_t a0 = cur[i].start;
        int64_t a1 = cur[i].end;
        uint8_t v = cur[i].v;
        uint32_t head_a = (uint32_t)(cur_base + i);
        while (j < nprev && (int64_t)prev[j].end + 1 < a0) j++;
        int64_t k = j;
        if (k < nprev && prev[k].v != v) k++;
        if (v == 255) {
            /* white is 8-connected: [max(a0, 1), a1] must meet
               [b0 - 1, b1 + 1], and every run k here has b0 <= a1 + 1
               and b1 + 1 >= a0, so only a1 >= 1 is left to test */
            if (a1 >= 1) {
                while (k < nprev && prev[k].start <= a1 + 1) {
                    uf_connect(uf, head_a, (uint32_t)(prev_base + k));
                    k += 2;
                }
            }
        } else {
            int64_t a0c = imax(a0, 1);
            while (k < nprev && prev[k].start <= a1 + 1) {
                Run pk = prev[k];
                if (imax(a0c, pk.start) <= imin(a1, pk.end)) uf_connect(uf, head_a, (uint32_t)(prev_base + k));
                k += 2;
            }
        }
    }
    last_column(uf, w, y, cur, ncur, cur_base, vcol_base, above_last, above_prev);
}

/* The runs of row y: runs + row_off[y] in the packed layout, or
   runs + y * w in the layout of the pool (rstride = w). */
typedef struct {
    const Run *runs;
    const uint32_t *row_off;
    int64_t rstride; /* 0: packed */
} RunRows;

AT_INLINE const Run *row_runs(RunRows rr, int64_t y) {
    return rr.rstride ? rr.runs + y * rr.rstride : rr.runs + rr.row_off[y];
}

AT_INLINE int64_t row_count(RunRows rr, int64_t y) { return rr.row_off[y + 1] - rr.row_off[y]; }

/* Union the runs of row y with the runs of row y-1. */
AT_INLINE void connect_runs_to_prev(UfNode *uf, const uint8_t *edge, int64_t w, int64_t y, RunRows rr,
                                    int64_t vcol_base) {
    connect_rows(uf, edge, w, y, row_runs(rr, y - 1), row_count(rr, y - 1), rr.row_off[y - 1], row_runs(rr, y),
                 row_count(rr, y), rr.row_off[y], vcol_base);
}

/* Process rows [y0, y1): initialize the row above the range, then each
   row, connecting it to the row above. */
static void unionfind_task(UfNode *uf, const uint8_t *edge, int64_t w, RunRows rr, int64_t vcol_base, int64_t y0,
                           int64_t y1) {
    uf_init_row(uf, row_runs(rr, y0 - 1), row_count(rr, y0 - 1), rr.row_off[y0 - 1]);
    for (int64_t y = y0; y < y1; y++) {
        uf_init_row(uf, row_runs(rr, y), row_count(rr, y), rr.row_off[y]);
        connect_runs_to_prev(uf, edge, w, y, rr, vcol_base);
    }
}

/* The most union-find tasks for an image of h rows. */
static int64_t unionfind_max_tasks(int64_t h) { return h / 2 + 1; }

/* The row ranges of the union-find tasks for a thread count (pairs of
   [y0, y1) in out, unionfind_max_tasks pairs at most). A single thread
   processes [1, h) as one task. Returns the number of tasks. */
static int64_t unionfind_tasks(int64_t h, int64_t nthreads, int64_t *out) {
    if (nthreads <= 1) {
        out[0] = 1;
        out[1] = h;
        return 1;
    }
    int64_t chunksize = 1 + h / (TASKS_PER_THREAD * nthreads);
    if (chunksize < 2) chunksize = 2;
    int64_t n = 0;
    for (int64_t i = 1; i < h; i += chunksize) {
        out[2 * n] = i;
        out[2 * n + 1] = imin(h, i + chunksize - 1);
        n++;
    }
    return n;
}

/* The last two pixels of each row, for connect_runs_to_prev. */
static void row_edges(const uint8_t *t, int64_t w, int64_t h, uint8_t *edge) {
    for (int64_t y = 0; y < h; y++) {
        edge[2 * y] = t[y * w + w - 2];
        edge[2 * y + 1] = t[y * w + w - 1];
    }
}

/* Set up the virtual last-column nodes, and with more than one task the
   runs below the last task (never connected: singletons). */
static void uf_prepare(UfNode *uf, int64_t h, const uint32_t *row_off, const int64_t *tasks, int64_t ntasks,
                       int64_t nthreads) {
    int64_t vcol_base = row_off[h];
    for (int64_t y = 0; y < h; y++) uf[vcol_base + y] = (UfNode){(uint32_t)(vcol_base + y), 0};
    if (nthreads > 1) {
        int64_t covered_end = ntasks > 0 ? tasks[2 * ntasks - 1] : 1;
        for (int64_t ri = row_off[covered_end]; ri < vcol_base; ri++) uf[ri] = (UfNode){(uint32_t)ri, 0};
    }
}

/* Build the union-find over all runs plus the virtual last-column nodes.
   uf must hold row_off[h] + h entries. */
static void connected_components(at_ctx *cx, UfNode *uf, const uint8_t *edge, int64_t w, int64_t h, const Run *runs,
                                 const uint32_t *row_off, int64_t nthreads) {
    int64_t vcol_base = row_off[h];
    at_mark mk = stk_enter(cx);
    int64_t *tasks = STK_ALLOC(cx, int64_t, 2 * unionfind_max_tasks(h));
    int64_t ntasks = unionfind_tasks(h, nthreads, tasks);
    uf_prepare(uf, h, row_off, tasks, ntasks, nthreads);
    RunRows rr = {runs, row_off, 0};
    for (int64_t i = 0; i < ntasks; i++)
        unionfind_task(uf, edge, w, rr, vcol_base, tasks[2 * i], tasks[2 * i + 1]);
    /* join the tasks: connect the first row of each task (but the first)
       to the row above it, in task order */
    for (int64_t i = 1; i < ntasks; i++) connect_runs_to_prev(uf, edge, w, tasks[2 * i] - 1, rr, vcol_base);
    stk_leave(cx, mk);
}

/* connect_rows_any, but the pairs go to out (two node ids each). */
static void rows_pairs(const uint8_t *edge, int64_t w, int64_t y, const Run *prev, int64_t nprev, int64_t prev_base,
                       const Run *cur, int64_t ncur, int64_t cur_base, int64_t vcol_base, vec_u32 *out) {
    uint8_t above_last = edge[2 * y - 1];
    uint8_t above_prev = edge[2 * y - 2];
    int64_t j = 0;
    for (int64_t i = 0; i < ncur; i++) {
        int64_t a0 = cur[i].start;
        int64_t a1 = cur[i].end;
        uint8_t v = cur[i].v;
        uint32_t head_a = (uint32_t)(cur_base + i);
        while (j < nprev && (int64_t)prev[j].end + 1 < a0) j++;
        for (int64_t k = j; k < nprev && prev[k].start <= a1 + 1; k++) {
            Run pk = prev[k];
            if (pk.v != v) continue;
            int64_t b0 = pk.start, b1 = pk.end;
            uint32_t head_b = (uint32_t)(prev_base + k);
            int64_t lo = imax(imax(a0, b0), 1);
            int64_t hi = imin(a1, b1);
            bool join = false;
            if (lo <= hi) {
                join = true;
            } else if (v == 255) {
                int64_t xl = imax(imax(a0, b0 + 1), 1);
                if (xl <= imin(a1, b1 + 1)) {
                    join = true;
                } else {
                    int64_t xr = imax(imax(a0, b0 - 1), 1);
                    join = xr <= imin(a1, b1 - 1);
                }
            }
            if (join) {
                svec_push(*out, head_a);
                svec_push(*out, head_b);
            }
        }
    }
    if (ncur > 0 && cur[ncur - 1].end == w - 2 && cur[ncur - 1].v == 255 && above_last == 255 &&
        above_prev != 255) {
        svec_push(*out, (uint32_t)(cur_base + ncur - 1));
        svec_push(*out, (uint32_t)(vcol_base + y - 1));
    }
}
