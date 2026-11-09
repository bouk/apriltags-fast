# Agent instructions

apriltags-fast: a fast AprilTag detector (`fast/`) under the API of the
AprilTag C library, branch `faster3` of BuildMonumental/apriltag
(`apriltag.h`, `apriltag.c`). The output must stay bit-identical to
`faster3` (compiled with `-ffp-contract=off`). Read README.md first.

## Files

- `common/`, `tag*.c`, `tag*.h`, `apriltag_pose.*`, `example/apriltag_demo.c`,
  `test/test_*.c`, `test/getline.*`, `test/data/` come from `faster3`
  without changes. Do not edit them; copy them again from `faster3`.
- `apriltag.h` is the `faster3` header with the cached buffers of the
  detector replaced by the fields of apriltags-fast. Keep all other types
  and declarations the same as `faster3`.
- `fast/families.c` makes the families of the native API from the
  upstream family files, so the code tables exist once.

## Build

- `cmake -B build -DBUILD_TESTING=ON && cmake --build build -j`, with
  clang (`CC=clang`) and with gcc (`CC=gcc`, another build directory).
  Both must have zero warnings.
- The flags of `fast/apriltag_all.c` are in CMakeLists.txt
  (`FAST_FLAGS`). Do not remove them.

## Verify

Do these after each change:

1. `ctest` in the build directory (upstream tests and `test_golden`), for
   both compilers.
2. For a change to `fast/`, `apriltag.c` or `apriltag.h`:
   `test/compare_faster3.sh <faster3-src> <faster3-lib> build` (see the
   script for how to build the reference: `-march=native
   -ffp-contract=off`, and `workerpool_run` must always call
   `workerpool_run_single`, or the `faster3` tasks race and its output
   changes from run to run). Also give it larger photos as images. It
   must pass all cases.
3. For a change to a hot loop, time both compilers (below): they must
   stay equally fast.

## Benchmark

`apriltag_fast_demo -q -t <n> -i 10 --timing <images>` prints the ms per
image and the time of each stage. `test/api_dump.c` times the upstream
API (`api_dump <threads> <iters> <family> <decimate> <images>`, median on
stderr). Run each side 3 times or more and alternate them: the noise on
a shared machine is 5% or more. Compare min and median per stage.

## Detector notes

- Memory: each thread has an `at_ctx` of data stacks (one reserved
  mapping, commit on touch, never freed between frames). A growable array
  takes a stack of its own (`svec_take`, `svec_push` with no capacity
  test); scratch is bumped from stack 0 (`STK_ALLOC`); `stk_enter` and
  `stk_leave` free both. No malloc and no copy per frame.
- Pool (`fast/pool.c`): workers sleep on a futex in their slot;
  `post_jobs` sets the whole wake tree before it posts. Rules: no write to
  an element that another task of the stage reads, done flags with
  release/acquire, results merged by task index (the result must not
  depend on which thread runs a task). The number of union-find tasks and
  the cluster hash size must follow the `faster3` task split for the
  thread count: they decide the representatives and the cluster order.
- `keysort_blocks` must give exactly the order of the `faster3` merge
  sort, also for keys with equal angles.
- `rle_row` does not store runs of 127.
- What made the hot loops fast (keep these):
  - Lane loops: an explicit vector type for the lane sums
    (`refine_lanes`, `point_keys`). A loop with `lanes[i % 8]` does not
    vectorize; a loop strip-mined by 8 vectorizes in clang, but gcc
    vectorizes the outer loop with thousands of permutes.
  - `at_sqrt`/`at_sqrtf`: no errno, so they vectorize.
  - `&` in place of `&&` of two loads (no branch).
  - Nested float selects as masks (`fsel`, `mask32`): gcc's jump
    threading makes nested selects into branches, which its vectorizer
    rejects.
  - `Run` is packed (5 bytes). `connect_rows_any` is inlined.
  - The scalar pixel loop of `refine_batch` stays scalar
    (`vectorize(disable)` / `GCC novector`).
  - Loops that must vectorize use `AT_VFOR`. Xcode builds the C code of a
    Swift package with `-Os` (Release), where clang vectorizes a loop only
    if it needs no scalar remainder loop; without `AT_VFOR` the detector
    is 1.5x slower there. Check with `clang -target arm64-apple-ios15.0
    -Os -Rpass=loop-vectorize`. Do not use `#pragma clang loop
    vectorize(enable)` for this: at `-Os` it masks the remainder (one
    branch per lane), and it lets clang reorder FP reductions.
  - Give each such loop `restrict` pointers (an `AT_INLINE` helper if the
    arrays share one buffer): at `-Os` clang makes no runtime alias checks.
- Tools: callgrind instruction counts per function on both compilers find
  a loop that one compiler does not vectorize; `-fopt-info-vec-missed`
  gives gcc's reason. Valgrind has no AVX-512: it runs the AVX2 versions.
  `-DAT_SIMD_DEBUG` with `AT_SIMD_V1=",name,"` runs named SIMD functions
  at AVX2; `AT_SIMD=0` or `1` lowers all of them.
