/* The worker pool.

   The structure is that of the Goose port's pool (pool.goose): a stage
   posts one job per worker, and each job (and the calling thread) claims
   tasks from an atomic counter until none are left. The union-find
   stitch and the cluster merge run on the calling thread in task order,
   while later tasks still run. The results do not depend on which
   thread runs which task, or when.

   Goose threads share no memory, so the Goose pool copies ranges in and
   out of shared views. Here the jobs work in place: they read the
   calling thread's arrays and write their results straight into them
   (or into the detector's stacks). The rules that make this safe are
   the same:
   - A shared array does not move while jobs of a stage can run (data
     stacks never move).
   - While jobs of a stage run, the calling thread reads only elements of
     tasks that are done, and writes only elements that no task of the
     stage reads or writes.
   - Two tasks never write the same element.
   - A task writes before it sets its done flag (release); the reader
     reads after it sees the flag (acquire). A job's end (its slot
     state, release) orders all of its writes before the end of the
     stage.

   A worker sleeps on a futex in its slot. Each wake-up costs the poster
   microseconds, so the calling thread wakes only POST_FAN workers, and
   each job first wakes the workers of its part of the jobs, two at a
   time. A job that no worker started when the calling thread is done is
   cancelled: it would find no task left. */
#include "common.h"

#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#define POST_FAN 4
#define AT_MAX_WORKERS 63
#define AT_NFLAGS 8192

enum { ST_POSTED = 0, ST_RUNNING = 1, ST_DONE = 2, ST_IDLE = 3 };
enum { STAGE_QUIT, STAGE_THRESH, STAGE_UF, STAGE_CLUSTERS, STAGE_FIT };
enum { C_THRESH, C_UF, C_CLUSTERS, C_FIT, C_COUNT };

struct at_pool;
static int64_t min_tag_width(const at_detector *td, bool *normal_border, bool *reversed_border);

/* A worker's slot: its state word ((job epoch << 2) | state), the end of
   the part of the jobs that it wakes, and its stacks. */
typedef struct {
    _Alignas(64) uint32_t word;
    int32_t sub_hi;
    int32_t start_hi; /* the end of the range of workers this one starts */
    struct at_pool *pool;
    int32_t index;
    at_ctx cx;
} Worker;

typedef struct {
    _Alignas(64) int64_t v;
} Counter;

struct at_pool {
    Worker workers[AT_MAX_WORKERS];
    pthread_t threads[AT_MAX_WORKERS];
    int32_t nworkers;
    int32_t started; /* atomic: threads created */
    uint32_t job_epoch;
    /* the stage: its kind and job, set before its jobs are posted */
    int kind;
    const void *job;
    Counter counter[C_COUNT];
    /* done flags of the tasks of a stage (by stage epoch), and the
       meeting points of two union-find tasks */
    int64_t epoch;
    int64_t flag[AT_NFLAGS];
    int64_t meet[AT_NFLAGS];
    /* buffers that the stages share, on the detector's stacks */
    Run *runs;       /* the runs of row y at y * w */
    Pt *clpts;       /* the points of the cluster tasks: band k at 4 * w * k * chunk */
    uint64_t *clids; /* the clusters of band k, at the same offsets */
    uint32_t *clcounts;
    uint32_t *join;  /* the pairs of the union-find joins (join_off) */
    uint32_t *troot; /* the roots of the first and last row of each task */
};

static long futex(uint32_t *addr, int op, uint32_t val) {
    return syscall(SYS_futex, addr, op, val, NULL, NULL, 0);
}

static void wake_worker(Worker *wk) { futex(&wk->word, FUTEX_WAKE_PRIVATE, 1); }

/* One step of a wait loop: a pause for the first 256 steps, then a
   yield (with more threads than cores, the thread that the loop waits
   for must get the core). */
static inline void wait_step(int64_t step) {
    if (step >= 256) {
        sched_yield();
        return;
    }
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

AT_INLINE void counter_set(struct at_pool *p, int c, int64_t v) {
    __atomic_store_n(&p->counter[c].v, v, __ATOMIC_RELAXED);
}

AT_INLINE int64_t counter_next(struct at_pool *p, int c) {
    return __atomic_fetch_add(&p->counter[c].v, 1, __ATOMIC_RELAXED);
}

AT_INLINE void flag_set(struct at_pool *p, int64_t i, int64_t epoch) {
    if ((uint64_t)i >= AT_NFLAGS) abort();
    __atomic_store_n(&p->flag[i], epoch, __ATOMIC_RELEASE);
}

AT_INLINE int64_t flag_get(struct at_pool *p, int64_t i) {
    if ((uint64_t)i >= AT_NFLAGS) abort();
    return __atomic_load_n(&p->flag[i], __ATOMIC_ACQUIRE);
}

/* Each of two tasks calls arrive(i, epoch) once, after its writes;
   exactly one call (the second) returns true, and it sees the writes of
   the first. */
AT_INLINE bool arrive(struct at_pool *p, int64_t i, int64_t epoch) {
    if ((uint64_t)i >= AT_NFLAGS) abort();
    return __atomic_exchange_n(&p->meet[i], epoch, __ATOMIC_ACQ_REL) == epoch;
}

static void run_stage(struct at_pool *p, at_ctx *cx);

/* Set the wake tree of the jobs lo..hi-1: job lo wakes lo + 1 and
   lo + 1 + a, each of which wakes the rest of its half. */
static void set_part(struct at_pool *p, int32_t lo, int32_t hi) {
    p->workers[lo].sub_hi = hi;
    int32_t n = hi - lo - 1;
    int32_t a = (n + 1) / 2;
    if (a > 0) set_part(p, lo + 1, lo + 1 + a);
    if (n - a > 0) set_part(p, lo + 1 + a, hi);
}

/* Wake the two workers below a job's worker in the wake tree. */
static void wake_part(struct at_pool *p, int32_t lo, int32_t hi) {
    int32_t n = hi - lo - 1;
    int32_t a = (n + 1) / 2;
    if (a > 0) wake_worker(&p->workers[lo + 1]);
    if (n - a > 0) wake_worker(&p->workers[lo + 1 + a]);
}

static void *worker_main(void *arg);

/* Start workers lo..hi-1: this thread starts lo, which starts the rest
   of the range (a start costs tens of microseconds). */
static void start_workers(struct at_pool *p, int32_t lo, int32_t hi) {
    if (lo >= hi) return;
    p->workers[lo].start_hi = hi;
    if (pthread_create(&p->threads[lo], NULL, worker_main, &p->workers[lo])) abort();
    __atomic_fetch_add(&p->started, 1, __ATOMIC_RELEASE);
}

static void *worker_main(void *arg) {
    Worker *wk = (Worker *)arg;
    struct at_pool *p = wk->pool;
    /* the rest of this worker's range, in two halves */
    int32_t lo = wk->index, hi = wk->start_hi;
    int32_t n = hi - lo - 1;
    int32_t a = (n + 1) / 2;
    start_workers(p, lo + 1, lo + 1 + a);
    start_workers(p, lo + 1 + a, hi);
    at_ctx_init(&wk->cx);
    for (;;) {
        uint32_t wv = __atomic_load_n(&wk->word, __ATOMIC_ACQUIRE);
        if ((wv & 3) == ST_POSTED) {
            uint32_t run = (wv & ~3u) | ST_RUNNING;
            if (__atomic_compare_exchange_n(&wk->word, &wv, run, false, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
                if (p->kind == STAGE_QUIT) {
                    __atomic_store_n(&wk->word, (wv & ~3u) | ST_DONE, __ATOMIC_RELEASE);
                    break;
                }
                wake_part(p, wk->index, wk->sub_hi);
                run_stage(p, &wk->cx);
                __atomic_store_n(&wk->word, (wv & ~3u) | ST_DONE, __ATOMIC_RELEASE);
            }
            continue;
        }
        futex(&wk->word, FUTEX_WAIT_PRIVATE, wv);
    }
    at_ctx_release(&wk->cx);
    return NULL;
}

/* Post the stage's jobs to workers 0..njobs-1 and wake the first
   POST_FAN of them; the others are woken by the jobs. */
static void post_jobs(struct at_pool *p, int kind, const void *job, int32_t njobs) {
    p->kind = kind;
    p->job = job;
    uint32_t e = ++p->job_epoch << 2;
    int32_t lo = 0, left = njobs;
    for (int i = 0; i < POST_FAN; i++) {
        int32_t part = (left + POST_FAN - 1 - i) / (POST_FAN - i);
        if (part > 0) set_part(p, lo, lo + part);
        lo += part;
        left -= part;
    }
    for (int32_t j = 0; j < njobs; j++) __atomic_store_n(&p->workers[j].word, e | ST_POSTED, __ATOMIC_RELEASE);
    lo = 0;
    left = njobs;
    for (int i = 0; i < POST_FAN; i++) {
        int32_t part = (left + POST_FAN - 1 - i) / (POST_FAN - i);
        if (part > 0) wake_worker(&p->workers[lo]);
        lo += part;
        left -= part;
    }
}

/* The end of a stage: cancel the jobs that no worker started, and wait
   for the others. */
static void finish_jobs(struct at_pool *p, int32_t njobs) {
    uint32_t e = p->job_epoch << 2;
    for (int32_t j = 0; j < njobs; j++) {
        Worker *wk = &p->workers[j];
        uint32_t wv = e | ST_POSTED;
        if (__atomic_compare_exchange_n(&wk->word, &wv, e | ST_IDLE, false, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
            continue;
        for (int64_t step = 0; __atomic_load_n(&wk->word, __ATOMIC_ACQUIRE) != (e | ST_DONE); step++) wait_step(step);
    }
}

static void pool_stop(struct at_internal *in) {
    struct at_pool *p = in->pool;
    if (!p) return;
    while (__atomic_load_n(&p->started, __ATOMIC_ACQUIRE) < p->nworkers) sched_yield();
    p->kind = STAGE_QUIT;
    p->job = NULL;
    uint32_t e = ++p->job_epoch << 2;
    for (int32_t j = 0; j < p->nworkers; j++) {
        __atomic_store_n(&p->workers[j].word, e | ST_POSTED, __ATOMIC_RELEASE);
        wake_worker(&p->workers[j]);
    }
    for (int32_t j = 0; j < p->nworkers; j++) pthread_join(p->threads[j], NULL);
    p->nworkers = 0;
    p->started = 0;
}

/* Start workers until there are nthreads - 1 of them. The pool never
   shrinks: idle workers sleep. This thread starts POST_FAN workers and
   does not wait for them: jobs posted before a worker runs wait in its
   slot, and the calling thread cancels them if it got there first. */
static void pool_ensure(struct at_internal *in, int64_t nthreads) {
    struct at_pool *p = in->pool;
    if (!p) {
        p = (struct at_pool *)aligned_alloc(64, (sizeof(struct at_pool) + 63) / 64 * 64);
        if (!p) abort();
        memset(p, 0, sizeof *p);
        p->runs = STK_TAKE(&in->dcx, Run);
        p->clpts = STK_TAKE(&in->dcx, Pt);
        p->clids = STK_TAKE(&in->dcx, uint64_t);
        p->clcounts = STK_TAKE(&in->dcx, uint32_t);
        p->join = STK_TAKE(&in->dcx, uint32_t);
        p->troot = STK_TAKE(&in->dcx, uint32_t);
        in->pool = p;
    }
    int32_t want = (int32_t)imin(nthreads - 1, AT_MAX_WORKERS);
    if (p->nworkers >= want) return;
    /* new workers' slots; the threads of an old range are all started
       before a new range starts (the slot ranges must not overlap) */
    while (__atomic_load_n(&p->started, __ATOMIC_ACQUIRE) < p->nworkers) sched_yield();
    int32_t lo = p->nworkers;
    for (int32_t j = lo; j < want; j++) {
        p->workers[j].pool = p;
        p->workers[j].index = j;
        p->workers[j].word = (p->job_epoch << 2) | ST_IDLE;
    }
    p->nworkers = want;
    int32_t left = want - lo;
    for (int i = 0; i < POST_FAN; i++) {
        int32_t part = (left + POST_FAN - 1 - i) / (POST_FAN - i);
        start_workers(p, lo, lo + part);
        lo += part;
        left -= part;
    }
}

/* ---------------------------------------------------------------------
   Threshold: threshold and run-length encode tile rows, in tasks of
   `chunk` tile rows. With src, decimate the rows by 2 first (into im). */
typedef struct {
    int64_t w, h, chunk, ntasks, stride;
    int32_t diff;
    uint8_t *im;        /* the quad detection image (w x h) */
    const uint8_t *src; /* the caller's image, or NULL */
    uint8_t *t;         /* the thresholded image */
    Run *runs;          /* row y's runs at y * w */
    uint32_t *counts;   /* the run count of each row */
} ThreshJob;

/* Threshold and run-length encode tile rows [ty0, ty1). */
static void thresh_task(at_ctx *cx, const ThreshJob *job, int64_t ty0, int64_t ty1) {
    int64_t w = job->w, h = job->h, th = h / 4;
    int64_t y0, y1;
    band_rows(h, ty0, ty1, &y0, &y1);
    at_mark mk = stk_enter(cx);
    /* the image rows that threshold_band reads */
    int64_t ra = 4 * imax(ty0 - 1, 0);
    int64_t rb = imax(4 * imin(ty1 + 1, th), y1);
    const uint8_t **rows = STK_ALLOC(cx, const uint8_t *, rb - ra);
    if (job->src) {
        /* decimate here: this task's rows go to the image, the rows of
           the tasks next to it (which they decimate too) to scratch */
        uint8_t *other = STK_ALLOC(cx, uint8_t, (rb - ra) * w);
        for (int64_t r = ra; r < rb; r++) {
            uint8_t *o = r >= y0 && r < y1 ? job->im + r * w : other + (r - ra) * w;
            decimate2_row(job->src + 2 * r * job->stride, job->src + (2 * r + 1) * job->stride, o, w);
            rows[r - ra] = o;
        }
    } else {
        for (int64_t r = ra; r < rb; r++) rows[r - ra] = job->im + r * w;
    }
    threshold_band(cx, rows, ra, w, h, job->diff, ty0, ty1, job->t + y0 * w);
    uint8_t *cb = STK_ALLOC(cx, uint8_t, w + 64);
    uint8_t *pb = STK_ALLOC(cx, uint8_t, w / 8 + 8);
    for (int64_t y = y0; y < y1; y++) job->counts[y] = (uint32_t)rle_row(job->t + y * w, w, job->runs + y * w, cb, pb);
    stk_leave(cx, mk);
}

static void thresh_tasks(struct at_pool *p, at_ctx *cx, const ThreshJob *job) {
    int64_t th = job->h / 4;
    for (;;) {
        int64_t k = counter_next(p, C_THRESH);
        if (k >= job->ntasks) break;
        thresh_task(cx, job, k * job->chunk, imin(th, (k + 1) * job->chunk));
    }
}

/* threshold_image and rle_image with the pool, in tile-row bands. t gets
   the thresholded image, row_off the first node of each row (h + 1
   entries); the runs of row y are in p->runs at y * w. */
static void threshold_parallel(at_ctx *cx, struct at_pool *p, uint8_t *im, int64_t w, int64_t h, int32_t diff,
                               int64_t nthreads, uint8_t *t, uint32_t *row_off, const uint8_t *src, int64_t stride) {
    at_mark mk = stk_enter(cx);
    uint32_t *counts = STK_ALLOC(cx, uint32_t, h);
    int64_t th = h / 4;
    int64_t chunk = 1 + th / (TASKS_PER_THREAD * nthreads);
    int64_t ntasks = (th + chunk - 1) / chunk;
    ThreshJob tj = {w, h, chunk, ntasks, src ? stride : 0, diff, im, src, t, p->runs, counts};
    counter_set(p, C_THRESH, 0);
    int32_t njobs = (int32_t)imin(nthreads - 1, ntasks);
    post_jobs(p, STAGE_THRESH, &tj, njobs);
    thresh_tasks(p, cx, &tj);
    finish_jobs(p, njobs);
    row_off[0] = 0;
    uint32_t at = 0;
    for (int64_t y = 0; y < h; y++) {
        at += counts[y];
        row_off[y + 1] = at;
    }
    stk_leave(cx, mk);
}

/* The threshold and the runs on this thread (deglitch), with the runs in
   the pool's layout (row y at y * w). */
static void threshold_single_pooled(at_ctx *cx, struct at_pool *p, const at_detector *td, const uint8_t *im,
                                    int64_t w, int64_t h, uint8_t *t, uint32_t *row_off) {
    threshold_image(cx, im, w, h, w, td->qtp.min_white_black_diff, t);
    if (td->qtp.deglitch) deglitch(cx, t, w, h);
    at_mark mk = stk_enter(cx);
    uint8_t *cb = STK_ALLOC(cx, uint8_t, w + 64);
    uint8_t *pb = STK_ALLOC(cx, uint8_t, w / 8 + 8);
    row_off[0] = 0;
    uint32_t at = 0;
    for (int64_t y = 0; y < h; y++) {
        at += (uint32_t)rle_row(t + y * w, w, p->runs + y * w, cb, pb);
        row_off[y + 1] = at;
    }
    stk_leave(cx, mk);
}

/* ---------------------------------------------------------------------
   Union-find: task k is over rows [tasks[2k], tasks[2k+1]), on the
   nodes of its rows (and the virtual nodes of the rows above them). */
typedef struct {
    int64_t w, vcol_base, epoch, ntasks;
    const int64_t *tasks;
    const uint32_t *row_off;
    const Run *runs;
    const uint8_t *edge;
    UfNode *uf;
} UfJob;

/* Point every node of [lo, hi) at its root. The nodes before i already
   point at their roots, so a parent before i gives the root in one load;
   three loads reach the root in almost all other cases, without a
   branch. */
static void compress_roots(UfNode *uf, int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; i++) {
        uint32_t p = uf[i].parent;
        uint32_t pp = uf[p].parent;
        uint32_t r = uf[pp].parent;
        if (p >= i && uf[r].parent != r)
            while (uf[r].parent != r) r = uf[r].parent;
        uf[i].parent = r;
    }
}

/* Where the pairs of the join of row y go in p->join (row y-1 starts at
   node prev_base). A join has at most 3 pairs per run of its two rows
   and one more: the next join's rows start at least 2 rows and all
   these runs later. */
AT_INLINE int64_t join_off(int64_t y, int64_t prev_base) { return 8 * (prev_base + y); }

/* The pairs of join k (task k's first row y with row y-1), by the roots
   in their tasks (p->troot; the virtual node of row y-1 stays), without
   the pairs that repeat one of the last two kept pairs: those are no-ops
   for uf_connect. Both tasks are done. The pairs go to p->join at
   join_off: the count, then the pairs; then join flag k is set. */
static void join_pairs(struct at_pool *p, const UfJob *job, int64_t k) {
    int64_t w = job->w;
    int64_t y = job->tasks[2 * k] - 1;
    const uint32_t *ro = job->row_off + y - 1;
    int64_t np = ro[1] - ro[0], nc = ro[2] - ro[1];
    uint32_t *dst = p->join + join_off(y, ro[0]);
    vec_u32 out = {dst + 1, 0};
    rows_pairs(job->edge, w, y, job->runs + (y - 1) * w, np, ro[0], job->runs + y * w, nc, ro[1], job->vcol_base,
               &out);
    const uint32_t *roots = p->troot;
    int64_t n = out.len / 2;
    /* drop a pair that repeats one of the last two kept pairs (the kept
       pairs are written over the read ones, behind them) */
    uint64_t l1 = UINT64_MAX, l2 = UINT64_MAX;
    int64_t kept = 0;
    for (int64_t q = 0; q < n; q++) {
        uint32_t a = roots[out.data[2 * q]];
        uint32_t b = out.data[2 * q + 1];
        if (b < ro[1]) b = roots[b];
        uint64_t ab = ((uint64_t)a << 32) | b;
        if (ab == l1 || ab == l2) continue;
        l2 = l1;
        l1 = ab;
        out.data[kept] = a;
        out.data[kept + 1] = b;
        kept += 2;
    }
    dst[0] = (uint32_t)(kept / 2);
    flag_set(p, 2 * k + 1, job->epoch);
}

/* Union-find task k, then its done flag. */
static void unionfind_task_job(struct at_pool *p, const UfJob *job, int64_t k) {
    int64_t y0 = job->tasks[2 * k], y1 = job->tasks[2 * k + 1];
    RunRows rr = {job->runs, job->row_off, job->w};
    unionfind_task(job->uf, job->edge, job->w, rr, job->vcol_base, y0, y1);
    /* point every node at its root: the stitch finds the same roots and
       sizes, and the lookups of the cluster tasks get short */
    int64_t rbase = job->row_off[y0 - 1], rend = job->row_off[y1];
    int64_t vbase = job->vcol_base + y0 - 1;
    compress_roots(job->uf, rbase, rend);
    compress_roots(job->uf, vbase, vbase + (y1 - y0));
    /* the roots of the first and the last row, for the joins */
    for (int64_t i = rbase; i < job->row_off[y0]; i++) p->troot[i] = job->uf[i].parent;
    for (int64_t i = job->row_off[y1 - 1]; i < rend; i++) p->troot[i] = job->uf[i].parent;
    flag_set(p, 2 * k, job->epoch);
    /* the second of two neighbor tasks makes the pairs of their join */
    if (k > 0 && arrive(p, k, job->epoch)) join_pairs(p, job, k);
    if (k + 1 < job->ntasks && arrive(p, k + 1, job->epoch)) join_pairs(p, job, k + 1);
}

static void unionfind_tasks_run(struct at_pool *p, const UfJob *job) {
    for (;;) {
        int64_t k = counter_next(p, C_UF);
        if (k >= job->ntasks) break;
        unionfind_task_job(p, job, k);
    }
}

/* Connect the pairs of a join, with prefetches some pairs ahead (the
   lines were written by the jobs, on other cores). */
static void join_connect(UfNode *uf, const uint32_t *pairs, int64_t n) {
    for (int64_t q = 0; q < imin(n, 8); q++) {
        __builtin_prefetch(uf + pairs[2 * q], 1);
        __builtin_prefetch(uf + pairs[2 * q + 1], 1);
    }
    for (int64_t q = 0; q < n; q++) {
        if (q + 8 < n) {
            __builtin_prefetch(uf + pairs[2 * (q + 8)], 1);
            __builtin_prefetch(uf + pairs[2 * (q + 8) + 1], 1);
        }
        if (q + 4 < n) __builtin_prefetch(uf + uf[pairs[2 * (q + 4) + 1]].parent, 1);
        uf_connect(uf, pairs[2 * q], pairs[2 * q + 1]);
    }
}

/* connected_components with the pool: the tasks touch disjoint nodes,
   so they run as jobs; the stitch runs here, in task order. Join i
   changes only nodes of tasks 0..i, and the tasks after it write only
   their own nodes, so join i runs as soon as tasks 0..i are done. The
   second of tasks i-1 and i to finish finds the pairs of join i and sets
   join flag i: here only uf_connect runs. */
static void connected_components_parallel(at_ctx *cx, struct at_pool *p, UfNode *uf, const uint8_t *edge, int64_t w,
                                          int64_t h, const uint32_t *row_off, int64_t nthreads) {
    at_mark mk = stk_enter(cx);
    int64_t vcol_base = row_off[h];
    int64_t *tasks = STK_ALLOC(cx, int64_t, 2 * unionfind_max_tasks(h));
    int64_t ntasks = unionfind_tasks(h, nthreads, tasks);
    uf_prepare(uf, h, row_off, tasks, ntasks, nthreads);
    int64_t ep = ++p->epoch;
    UfJob uj = {w, vcol_base, ep, ntasks, tasks, row_off, p->runs, edge, uf};
    counter_set(p, C_UF, 0);
    int32_t njobs = (int32_t)imin(nthreads - 1, ntasks);
    post_jobs(p, STAGE_UF, &uj, njobs);
    int64_t next = 1;
    bool claim = true;
    int64_t waits = 0;
    for (;;) {
        while (next < ntasks && flag_get(p, 2 * next + 1) == ep) {
            /* the task of next found the pairs */
            const uint32_t *jp = p->join + join_off(tasks[2 * next] - 1, row_off[tasks[2 * next] - 2]);
            join_connect(uf, jp + 1, jp[0]);
            next++;
        }
        if (next >= ntasks && (ntasks == 0 || flag_get(p, 2 * (ntasks - 1)) == ep)) break;
        if (claim) {
            int64_t k = counter_next(p, C_UF);
            if (k < ntasks)
                unionfind_task_job(p, &uj, k);
            else
                claim = false;
        } else {
            wait_step(waits++);
        }
    }
    finish_jobs(p, njobs);
    stk_leave(cx, mk);
}

/* ---------------------------------------------------------------------
   Clusters: task k is the gradient points of rows
   1 + k*chunk .. min(h-1, 1 + (k+1)*chunk) - 1, each against the row
   below. Its clusters (in slot order, without the empty ones) and their
   points go to the pool's buffers from 4*w*k*chunk on. */
typedef struct {
    int64_t w, h, chunk, ntasks, vcol_base, min_cluster_pixels, epoch;
    const uint8_t *t;
    const Run *runs;
    const uint32_t *row_off;
    const UfNode *uf;
    /* per task: the number of clusters and the gap after each */
    int64_t *ncl;
    int64_t *gap;
} ClusterJob;

/* The representative of a node of the finished union-find, and whether
   its component is big enough. */
AT_INLINE uint32_t shared_rep(const UfNode *uf, int64_t i, int64_t min_cluster_pixels, uint8_t *ok) {
    uint32_t r = uf[i].parent;
    for (;;) {
        uint32_t q = uf[r].parent;
        if (q == r) break;
        r = q;
    }
    *ok = (int64_t)uf[r].size + 1 >= min_cluster_pixels ? 1 : 0;
    return r;
}

/* The rows of a cluster task, for gradient_points: the runs in the
   pool's layout, the representatives of the task's runs. */
typedef struct {
    const Run *runs;
    const uint32_t *row_off;
    int64_t w, y0, r0;
    const uint32_t *rep;
    const uint8_t *ok;
} PoolRows;

static LabRow pool_row(void *ctx, int64_t i) {
    const PoolRows *p = (const PoolRows *)ctx;
    int64_t y = p->y0 + i;
    int64_t a = p->row_off[y] - p->r0;
    return (LabRow){p->runs + y * p->w, p->rep + a, p->ok + a, p->row_off[y + 1] - p->row_off[y]};
}

static void cluster_task(struct at_pool *p, at_ctx *cx, const ClusterJob *job, int64_t k) {
    int64_t w = job->w;
    int64_t y0 = 1 + k * job->chunk;
    int64_t n = imin(job->h - 1, y0 + job->chunk) - y0;
    at_mark mk = stk_enter(cx);
    const uint32_t *ro = job->row_off;
    int64_t r0 = ro[y0];
    int64_t nr = ro[y0 + n + 1] - r0;
    uint32_t *rep = STK_ALLOC(cx, uint32_t, nr);
    uint8_t *ok = STK_ALLOC(cx, uint8_t, nr);
    /* every node points at its task's root: runs with the same parent
       have the same representative */
    uint32_t lp = UINT32_MAX, lr = 0;
    uint8_t lo = 0;
    for (int64_t i = 0; i < nr; i++) {
        uint32_t par = job->uf[r0 + i].parent;
        if (par != lp) {
            lr = shared_rep(job->uf, par, job->min_cluster_pixels, &lo);
            lp = par;
        }
        rep[i] = lr;
        ok[i] = lo;
    }
    uint32_t *vrep = STK_ALLOC(cx, uint32_t, n + 1);
    uint8_t *vok = STK_ALLOC(cx, uint8_t, n + 1);
    for (int64_t j = 0; j <= n; j++) vrep[j] = shared_rep(job->uf, job->vcol_base + y0 + j, job->min_cluster_pixels, &vok[j]);
    ClusterTable ct;
    int64_t m = 64;
    while (m < nr) m *= 2;
    ct_init(cx, &ct, m);
    PoolRows pr = {job->runs, ro, w, y0, r0, rep, ok};
    gradient_points(cx, &ct, job->t + y0 * w, w, y0, n, pool_row, &pr, vrep, vok);
    /* the clusters in slot order, without the empty ones; their points
       grouped by cluster, padded when the points and the gaps fit in the
       band's 4w points a row */
    int64_t base = 4 * w * (y0 - 1);
    int64_t nslots = ct.ids.len, nc = 0;
    for (int64_t s = 0; s < nslots; s++) nc += ct.counts.data[s] != 0;
    int64_t gap = ct.pts.len + PT_PAD * nc <= 4 * w * n ? PT_PAD : 0;
    uint32_t *pos = STK_ALLOC(cx, uint32_t, nslots);
    uint64_t *ids = p->clids + base;
    uint32_t *counts = p->clcounts + base;
    uint32_t at = 0;
    nc = 0;
    for (int64_t s = 0; s < nslots; s++) {
        uint32_t c = ct.counts.data[s];
        if (c == 0) continue;
        ids[nc] = ct.ids.data[s];
        counts[nc] = c;
        nc++;
        pos[s] = at;
        at += c + (uint32_t)gap;
    }
    scatter_points(&ct, pos, p->clpts + base, gap == PT_PAD);
    job->ncl[k] = nc;
    job->gap[k] = gap;
    stk_leave(cx, mk);
    flag_set(p, k, job->epoch);
}

static void cluster_tasks_run(struct at_pool *p, at_ctx *cx, const ClusterJob *job) {
    for (;;) {
        int64_t k = counter_next(p, C_CLUSTERS);
        if (k >= job->ntasks) break;
        cluster_task(p, cx, job, k);
    }
}

/* ---------------------------------------------------------------------
   Fit: fit quads to clusters, and decode them. Task k is the clusters
   jfirst[o]..jfirst[o+1] for o = order[k]. Cluster c has points
   cstart[c]..cstart[c+1]; they are the segments sfirst[c]..sfirst[c+1],
   each a start in the pool's points (ssrc) and a count (scnt). The
   results go to per-cluster slots: task o's quads from jfirst[o] on,
   their count in nq[o]. */
typedef struct {
    float quad_decimate;
    bool refine;
    double decode_sharpening;
    ImView view; /* the image that the decode samples */
} DecodeParams;

/* The decode results of the fit tasks, read in place: task o has nq[o]
   quads from quads + jfirst[o], hok per quad and status, margin and
   code lookup per quad and family. */
typedef struct {
    const Quad *quads;
    const uint8_t *hok;
    const uint8_t *status;
    const float *margins;
    const uint32_t *lookups;
    const uint32_t *jfirst;
    const int64_t *nq;
    int64_t nfit;
} DecodeOut;

typedef struct {
    int64_t ntasks, w, h, tag_width, nf;
    bool normal_border, reversed_border;
    at_quad_thresh_params qtp;
    DecodeParams dp;
    const FamGeom *geoms;
    const FamLookup *looks;
    QdTables tabs;
    const uint8_t *im; /* the quad detection image */
    const Pt *clpts;
    const uint32_t *cstart, *sfirst, *scnt, *jfirst, *order;
    const uint64_t *ssrc;
    /* outputs */
    Quad *quads;
    uint8_t *hok, *status;
    float *margins;
    uint32_t *lookups;
    int64_t *nq;
} FitJob;

/* Scale the quads of a fit task to the full image, refine them
   (optionally), find their homographies and sample their bits. */
static void decode_quad_list(at_ctx *cx, const FitJob *job, Quad *qs, int64_t n, int64_t c0) {
    const DecodeParams *dp = &job->dp;
    int64_t nf = job->nf;
    int64_t tw = 1;
    for (int64_t i = 0; i < nf; i++) tw = imax(tw, job->geoms[i].total_width);
    at_mark mk = stk_enter(cx);
    double *scratch = STK_ALLOC(cx, double, 2 * tw * tw);
    if (dp->quad_decimate > 1.0f) {
        float f = dp->quad_decimate;
        for (int64_t i = 0; i < n; i++)
            for (int j = 0; j < 4; j++) {
                qs[i].p[j][0] *= f;
                qs[i].p[j][1] *= f;
            }
    }
    for (int64_t i = 0; i < n; i++) {
        Quad *q = &qs[i];
        if (dp->refine) refine_edges(cx, dp->quad_decimate, &dp->view, q);
        bool ok = quad_update_homography(q);
        job->hok[c0 + i] = ok;
        for (int64_t fi = 0; fi < nf; fi++) {
            int64_t k = (c0 + i) * nf + fi;
            int st = SAMPLE_REJECT;
            float margin = 0.0f;
            uint64_t rcode = 0;
            if (ok && job->geoms[fi].reversed_border == q->reversed_border)
                st = quad_sample(&job->geoms[fi], dp->decode_sharpening, &dp->view, q, scratch, &margin, &rcode);
            job->status[k] = (uint8_t)st;
            job->margins[k] = st == SAMPLE_OK ? margin : 0.0f;
            job->lookups[k] = st == SAMPLE_OK ? quick_decode_codeword(&job->looks[fi], job->tabs, rcode) : 0;
        }
    }
    stk_leave(cx, mk);
}

static void fit_task(at_ctx *cx, const FitJob *job, int64_t k) {
    int64_t o = job->order[k];
    int64_t c0 = job->jfirst[o], c1 = job->jfirst[o + 1];
    int64_t min_px = job->qtp.min_cluster_pixels;
    int64_t max_px = 2 * (2 * job->w + 2 * job->h);
    Quad *out = job->quads + c0;
    int64_t nq = 0;
    for (int64_t c = c0; c < c1; c++) {
        int64_t n = (int64_t)job->cstart[c + 1] - job->cstart[c];
        if (n < min_px || n > max_px) continue;
        at_mark mk = stk_enter(cx);
        /* a cluster of one segment is read in place; others are gathered */
        int64_t s0 = job->sfirst[c], s1 = job->sfirst[c + 1];
        const Pt *pts;
        if (s1 - s0 == 1) {
            pts = job->clpts + job->ssrc[s0];
        } else {
            Pt *g = STK_ALLOC(cx, Pt, n);
            int64_t at = 0;
            for (int64_t s = s0; s < s1; s++) {
                memcpy(g + at, job->clpts + job->ssrc[s], (size_t)job->scnt[s] * sizeof(Pt));
                at += job->scnt[s];
            }
            pts = g;
        }
        Quad q;
        memset(&q, 0, sizeof q);
        if (fit_quad(cx, &job->qtp, pts, n, job->im, job->w, job->h, job->tag_width, job->normal_border,
                     job->reversed_border, &q))
            out[nq++] = q;
        stk_leave(cx, mk);
    }
    decode_quad_list(cx, job, out, nq, c0);
    job->nq[o] = nq;
}

static void fit_tasks_run(struct at_pool *p, at_ctx *cx, const FitJob *job) {
    for (;;) {
        int64_t k = counter_next(p, C_FIT);
        if (k >= job->ntasks) break;
        fit_task(cx, job, k);
    }
}

static void run_stage(struct at_pool *p, at_ctx *cx) {
    switch (p->kind) {
    case STAGE_THRESH: thresh_tasks(p, cx, (const ThreshJob *)p->job); break;
    case STAGE_UF: unionfind_tasks_run(p, (const UfJob *)p->job); break;
    case STAGE_CLUSTERS: cluster_tasks_run(p, cx, (const ClusterJob *)p->job); break;
    case STAGE_FIT: fit_tasks_run(p, cx, (const FitJob *)p->job); break;
    default: break;
    }
}

/* The gradient clusters of rows 1..h-2 and the quads fit to them, with
   the pool. The cluster tasks run in row bands; their clusters are
   merged here in band order, which gives the same points in the same
   order as one sweep. The fit tasks then read the points of their
   clusters in the bands. Returns the time at the end of the clusters. */
static double clusters_fit_parallel(at_ctx *cx, struct at_pool *p, const at_detector *td, const uint8_t *t, int64_t w,
                                    int64_t h, const uint32_t *row_off, const UfNode *uf, const uint8_t *im,
                                    int64_t tag_width, bool normal_border, bool reversed_border, int64_t nthreads,
                                    const DecodeParams *dp, vec_quad *quads, DecodeOut *res) {
    const struct at_internal *in = td->in;
    int64_t vcol_base = row_off[h];
    int64_t nrows = h - 2;
    int64_t chunk = 1 + nrows / (TASKS_PER_THREAD * nthreads);
    int64_t ntasks = (nrows + chunk - 1) / chunk;
    int64_t *ncl = STK_ALLOC(cx, int64_t, ntasks);
    int64_t *gaps = STK_ALLOC(cx, int64_t, ntasks);
    int64_t ep = ++p->epoch;
    ClusterJob cj = {w,       h,       chunk, ntasks, vcol_base, td->qtp.min_cluster_pixels, ep, t, p->runs, row_off,
                     uf,      ncl,     gaps};
    counter_set(p, C_CLUSTERS, 0);
    int32_t njobs = (int32_t)imin(nthreads - 1, ntasks);
    post_jobs(p, STAGE_CLUSTERS, &cj, njobs);
    /* the merge of band k needs only bands 0..k-1: merge each band as
       soon as it is done; each band cluster is one segment of points */
    ClusterTable ct;
    ct_init(cx, &ct, cluster_map_size(w, h, nthreads));
    vec_u32 gslot;
    vec_u64 src;
    vec_u32 scount;
    svec_take(cx, gslot);
    svec_take(cx, src);
    svec_take(cx, scount);
    int64_t *cfirst = STK_ALLOC(cx, int64_t, ntasks + 1);
    int64_t next = 0, waits = 0;
    bool claim = true;
    while (next < ntasks) {
        while (next < ntasks && flag_get(p, next) == ep) {
            int64_t base = 4 * w * (next * chunk);
            const uint64_t *ids = p->clids + base;
            const uint32_t *counts = p->clcounts + base;
            int64_t at = base;
            cfirst[next] = gslot.len;
            for (int64_t j = 0; j < ncl[next]; j++) {
                int64_t slot = cluster_slot_id(&ct, ids[j]);
                ct.counts.data[slot] += counts[j];
                svec_push(gslot, (uint32_t)slot);
                svec_push(src, (uint64_t)at);
                svec_push(scount, counts[j]);
                at += counts[j] + gaps[next];
            }
            next++;
        }
        if (next >= ntasks) break;
        if (claim) {
            int64_t k = counter_next(p, C_CLUSTERS);
            if (k < ntasks) {
                cluster_task(p, cx, &cj, k);
                continue;
            }
            claim = false;
        }
        wait_step(waits++);
    }
    cfirst[ntasks] = gslot.len;
    finish_jobs(p, njobs);
    int64_t nseg = gslot.len;
    uint32_t *rank = STK_ALLOC(cx, uint32_t, ct.ids.len);
    vec_u32 cstart;
    svec_take(cx, cstart);
    cluster_ranks(cx, &ct, rank, &cstart);
    int64_t nclusters = cstart.len - 1;
    /* the segments by output cluster (band order within a cluster) */
    uint32_t *sfirst = STK_ZALLOC(cx, uint32_t, nclusters + 1);
    for (int64_t j = 0; j < nseg; j++) sfirst[rank[gslot.data[j]] + 1] += 1;
    for (int64_t c = 0; c < nclusters; c++) sfirst[c + 1] += sfirst[c];
    uint32_t *fill = STK_ALLOC(cx, uint32_t, nclusters + 1);
    memcpy(fill, sfirst, (size_t)nclusters * 4);
    uint64_t *ssrc = STK_ALLOC(cx, uint64_t, nseg);
    uint32_t *scnt = STK_ALLOC(cx, uint32_t, nseg);
    for (int64_t j = 0; j < nseg; j++) {
        uint32_t c = rank[gslot.data[j]];
        uint32_t at = fill[c]++;
        ssrc[at] = src.data[j];
        scnt[at] = scount.data[j];
    }
    double t_clusters = at_clock();

    /* fit tasks: queued largest first (by points), so that no large task
       starts last; the results go by task index. Twice the tasks of the
       other stages: a shorter tail (4 times was slower) */
    int64_t fchunk = 1 + nclusters / (2 * TASKS_PER_THREAD * nthreads);
    int64_t nfit = (nclusters + fchunk - 1) / fchunk;
    uint32_t *jfirst = STK_ALLOC(cx, uint32_t, nfit + 1);
    for (int64_t k = 0; k < nfit; k++) jfirst[k] = (uint32_t)(k * fchunk);
    jfirst[nfit] = (uint32_t)nclusters;
    uint32_t *order = STK_ALLOC(cx, uint32_t, nfit);
    for (int64_t k = 0; k < nfit; k++) order[k] = (uint32_t)k;
    /* stable insertion sort by points, descending */
    for (int64_t i = 1; i < nfit; i++) {
        uint32_t v = order[i];
        uint32_t pv = cstart.data[jfirst[v + 1]] - cstart.data[jfirst[v]];
        int64_t j = i;
        while (j > 0 && cstart.data[jfirst[order[j - 1] + 1]] - cstart.data[jfirst[order[j - 1]]] < pv) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = v;
    }
    int64_t nf = in->nfamilies;
    FamGeom *geoms = STK_ALLOC(cx, FamGeom, nf);
    FamLookup *looks = STK_ALLOC(cx, FamLookup, nf);
    for (int64_t fi = 0; fi < nf; fi++) {
        geoms[fi] = fam_geom(&in->families[fi]);
        looks[fi] = (FamLookup){in->qd[fi], in->families[fi].ncodes, in->families[fi].nbits};
    }
    FitJob fj = {
        .ntasks = nfit,
        .w = w,
        .h = h,
        .tag_width = tag_width,
        .nf = nf,
        .normal_border = normal_border,
        .reversed_border = reversed_border,
        .qtp = td->qtp,
        .dp = *dp,
        .geoms = geoms,
        .looks = looks,
        .tabs = {in->qd_offsets.data, in->qd_ids.data, in->qd_codes.data},
        .im = im,
        .clpts = p->clpts,
        .cstart = cstart.data,
        .sfirst = sfirst,
        .scnt = scnt,
        .jfirst = jfirst,
        .order = order,
        .ssrc = ssrc,
        .quads = quads->data,
        .hok = STK_ALLOC(cx, uint8_t, nclusters),
        .status = STK_ALLOC(cx, uint8_t, nclusters * nf),
        .margins = STK_ALLOC(cx, float, nclusters * nf),
        .lookups = STK_ALLOC(cx, uint32_t, nclusters * nf),
        .nq = STK_ALLOC(cx, int64_t, nfit),
    };
    counter_set(p, C_FIT, 0);
    njobs = (int32_t)imin(nthreads - 1, nfit);
    post_jobs(p, STAGE_FIT, &fj, njobs);
    fit_tasks_run(p, cx, &fj);
    finish_jobs(p, njobs);
    int64_t total = 0;
    for (int64_t k = 0; k < nfit; k++) total += fj.nq[k];
    quads->len = total;
    *res = (DecodeOut){fj.quads, fj.hok, fj.status, fj.margins, fj.lookups, jfirst, fj.nq, nfit};
    return t_clusters;
}

/* Threshold, segment, cluster, fit and decode quads with the pool. If
   src is not NULL, im is not made yet: the threshold jobs decimate src
   (rows stride bytes apart) by 2 into it. The scratch stays on cx until
   the caller leaves its mark (res points into it). */
static void quad_thresh_pooled(at_detector *td, uint8_t *im, int64_t w, int64_t h, int64_t nthreads,
                               const uint8_t *src, int64_t stride, const ImView *dview, vec_quad *quads,
                               DecodeOut *res) {
    struct at_internal *in = td->in;
    struct at_pool *p = in->pool;
    at_ctx *cx = &in->mcx;
    double t0 = at_clock();
    uint8_t *t = in->buf_t;
    uint32_t *row_off = STK_ALLOC(cx, uint32_t, h + 1);
    if (!td->qtp.deglitch)
        threshold_parallel(cx, p, im, w, h, td->qtp.min_white_black_diff, nthreads, t, row_off, src, stride);
    else
        threshold_single_pooled(cx, p, td, im, w, h, t, row_off);
    if (td->debug) at_write_pgm("debug_threshold.pnm", t, w, h);
    double t1 = at_clock();
    td->timing.threshold = t1 - t0;

    UfNode *uf = in->buf_uf;
    uint8_t *edge = STK_ALLOC(cx, uint8_t, 2 * h);
    row_edges(t, w, h, edge);
    connected_components_parallel(cx, p, uf, edge, w, h, row_off, nthreads);
    double t2 = at_clock();
    td->timing.unionfind = t2 - t1;

    *res = (DecodeOut){0};
    if (h < 3) return;
    bool normal_border, reversed_border;
    int64_t tag_width = min_tag_width(td, &normal_border, &reversed_border);
    DecodeParams dp = {td->quad_decimate, td->refine_edges, td->decode_sharpening, *dview};
    double t3 = clusters_fit_parallel(cx, p, td, t, w, h, row_off, uf, im, tag_width, normal_border, reversed_border,
                                      nthreads, &dp, quads, res);
    td->timing.clusters = t3 - t2;
    td->timing.fit_quads = at_clock() - t3;
}
