# apriltags-fast

A faster AprilTag detector with the API of the AprilTag C library. Use it in
place of [BuildMonumental/apriltag](https://github.com/BuildMonumental/apriltag/tree/faster3)
(branch `faster3`) or the upstream [AprilRobotics/apriltag](https://github.com/AprilRobotics/apriltag):
the headers, types, functions, CMake package (`apriltag::apriltag`) and
pkg-config file have the same names.

- The output is the same as `faster3`, bit for bit: ids, hamming, decision
  margin, center, corners, homography and the number of quad candidates.
  (`faster3` compiled with `-ffp-contract=off`; FMA contraction changes the
  last bits of the C library's output.)
- It takes 38-47% less time than `faster3` at 1 thread, and 71-74% less at
  16 threads (see [Performance](#performance)).
- Clang and gcc give the same speed.

## Use

The upstream API does not change:

```c
#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>

apriltag_detector_t *td = apriltag_detector_create();
apriltag_family_t *tf = tag36h11_create();
apriltag_detector_add_family(td, tf);
td->nthreads = 4;

zarray_t *detections = apriltag_detector_detect(td, im);   // im: image_u8_t *
for (int i = 0; i < zarray_size(detections); i++) {
    apriltag_detection_t *det;
    zarray_get(detections, i, &det);
    // det->id, det->c, det->p, det->H, ...
}
apriltag_detections_destroy(detections);

apriltag_detector_destroy(td);
tag36h11_destroy(tf);
```

`apriltag_pose.h` (`estimate_tag_pose`) and `common/` (`image_u8`, `zarray`,
`matd`, `pjpeg`, ...) are the files of `faster3`, without changes.

### Native API

`fast/apriltag_fast.h` is the API of the detector itself. It gives the
detections in an array that the detector keeps, without an allocation per
detection or a `matd_t` per homography:

```c
#include <apriltag/fast/apriltag_fast.h>

at_detector *td = at_detector_create();
td->nthreads = 4;
at_add_family(td, at_tag36h11(), 2);
at_detect(td, pixels, width, height, stride);
for (int64_t i = 0; i < td->ndetections; i++) {
    const at_detection *d = &td->detections[i];   // d->id, d->c, d->p, d->H
}
at_detector_destroy(td);
```

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build
```

In CMake: `find_package(apriltag REQUIRED)` and
`target_link_libraries(app apriltag::apriltag)`. With pkg-config: `apriltag`.

The CMake file sets the flags of the detector (`fast/apriltag_all.c`):
`-O2 -ffp-contract=off -fno-math-errno`, and for gcc also
`-fno-trapping-math -fvect-cost-model=dynamic`. Do not remove them: the
output must not depend on FMA contraction, and without them gcc does not
vectorize the hot loops.

Requirements: Linux (the worker threads sleep on futexes), and clang or gcc.
Use gcc 12 or later: older gcc versions build it, but do not vectorize the
corner sort keys. On x86-64, the SIMD functions get AVX2 and AVX-512
versions, and the detector selects one at run time. Other CPUs use the plain
C versions (not tested).

## Differences from the upstream API

- `td->nedges` and `td->nsegments` stay 0. `td->wp` stays NULL: the detector
  has its own worker threads (started on the first detection, stopped by
  `apriltag_detector_destroy`).
- `td->tp` has the time of each stage, as upstream.
- A family keeps no decode tables in `fam->impl`: each detector makes its own
  tables when you add a family (on the next detection).
- At most 16 families per detector, and `bits_corrected` at most 3. A family
  that breaks these limits is removed, with a message on stderr.
- `apriltag_detector_t` has no cached buffers (`cached_uf`, ...). It has three
  fields of its own at the end.
- No ArUco families (`faster3` has none), and no Python wrapper.

## How it is faster

The detector is a C port of apriltag-goose, a port of `faster3` to
[Goose](https://github.com/aardappel/goose). The
algorithms and the results are those of `faster3`. The time goes down because:

- Memory: each thread has a set of data stacks (one reserved mapping, pages
  committed on first touch, kept between frames). Growable arrays grow in
  place; nothing is allocated or copied per frame.
- Threads: a worker pool with a futex per worker, a wake-up tree, and task
  counters in place of a locked queue. The union-find stitch and the cluster
  merge run on the main thread while later tasks still run.
- SIMD: the hot loops are written so that clang and gcc vectorize them, in
  AVX2 and AVX-512 versions (threshold, run-length encoding, cluster points,
  sort keys, line fits, corner search, edge refinement).
- Run-length encoding and union-find over runs, clusters made from runs with a
  per-row segment table, radix sort of the boundary points, edge refinement on
  batches of sample points.

## Performance

Milliseconds per image (median of 3 runs of 15 iterations, the same
upstream-API program, `test/api_dump.c`), decimate 2. AMD EPYC Genoa VM,
16 vCPU. `faster3`: gcc 15 `-O3 -march=native -ffp-contract=off`.
apriltags-fast: clang 21 (gcc 15 gives the same within 3%).

| Images | Threads | faster3 | apriltags-fast | Time |
|---|---|---|---|---|
| 10 synthetic, 1920x1080, tag36h11 | 1 | 24.19 | 12.88 | -47% |
| | 4 | 8.88 | 3.78 | -57% |
| | 8 | 6.27 | 2.31 | -63% |
| | 16 | 6.25 | 1.60 | -74% |
| 2 photos, tag36h11 | 1 | 22.31 | 13.72 | -39% |
| | 4 | 8.28 | 4.35 | -48% |
| | 8 | 6.57 | 2.61 | -60% |
| | 16 | 6.06 | 1.67 | -72% |
| 1 photo, tagStandard52h13 | 1 | 33.87 | 20.91 | -38% |
| | 4 | 12.22 | 6.26 | -49% |
| | 8 | 8.45 | 3.53 | -58% |
| | 16 | 7.53 | 2.15 | -71% |

At 16 threads, one detector gives about 4 times the frames per second of
`faster3`, mostly because `faster3` is no faster at 16 threads than at 8.
With many detectors that run at the same time (one thread each), expect the
gain at 1 thread.

## Tests

```sh
cmake -B build -DBUILD_TESTING=ON && cmake --build build -j
(cd build && ctest)
```

- `test_detection`, `test_tag_pose_estimation`, `test_multiple_sizes`: the
  tests of `faster3`, without changes.
- `test_golden`: 18 parameter sets (decimation, blur, sharpening, deglitch,
  threads, families), output in hex floats, against `test/golden` (made with
  `faster3`).
- `test/compare_faster3.sh`: the upstream API of apriltags-fast against a
  `faster3` build, bit for bit, over 48 parameter sets (families, decimation,
  threads).

## Layout

| Path | Content |
|---|---|
| `apriltag.h`, `apriltag.c` | The upstream API over the fast detector. |
| `fast/` | The detector (`apriltag_all.c` compiles all of it as one unit) and its API, `apriltag_fast.h`. |
| `common/`, `tag*.c`, `apriltag_pose.c` | From `faster3`, without changes. |
| `example/apriltag_demo.c` | The demo of `faster3`, without changes. |
| `example/fast_demo.c` | A demo of the native API (`--timing` prints the stages). |
| `test/` | Tests (see above). |

## License

BSD 2-Clause, as the AprilTag library (`LICENSE.md`).

## GPU (Metal)

On Apple platforms the GPU can do work of the cluster stage
(`fast/metal.m`, `fast/gpu.c`). It is off by default. Turn it on with
`at_use_gpu` (`apriltag_detector_use_gpu` in the upstream API, `--gpu` or
`--gpu-auto` in `apriltag_fast_demo`, or `APRILTAG_GPU=1` / `auto` in the
environment). The detections are bit-identical with and without it.

The GPU decimates and thresholds the image, labels the connected
components (a lock-free union-find with atomics), and emits the boundary
points with their clusters in a hash table. Apple GPUs have no 64-bit
floats, so the quad fit, the edge refinement and the decode stay on the
CPU. The order of the clusters comes from the roots of the union-find of
the CPU, so the CPU still does its threshold and union-find, at the same
time as the GPU.

- One thread: the GPU does the whole cluster stage. `AT_GPU_AUTO`
  measures both paths and uses the faster one.
- More threads (`AT_GPU_ON` only): the GPU is one more worker. It makes
  the points of the last bands of the image, and a band task takes them
  if the GPU is done when the task starts.

Measured, ms for each photo (3196 x 2132, tag36h11, min of several runs):

| device | decimate | threads | CPU | with GPU |
| --- | --- | --- | --- | --- |
| iPhone 18 Pro | 2 | 1 | 9.95 | 9.24 |
| iPhone 18 Pro | 2 | 8 | 3.19 | 3.21 |
| iPhone 18 Pro | 1 | 1 | 27.67 | 24.33 |
| iPhone 18 Pro | 1 | 8 | 8.57 | 8.76 |
| M3 Pro | 2 | 1 | 14.5 | 13.2 |
| M3 Pro | 2 | 8 | 3.2 | 3.3 |

The GPU makes an image faster with one thread only. With all cores in
use it does not, for these measured reasons:

- The GPU can only take the cluster stage (25 to 30% of an image).
- The GPU gets about 3 times slower while all CPU cores work (the chip
  shares its power and memory between them): alone it does the stage of a
  6.8 MP image in 4.9 ms, the 8 threads do it in 2.5 ms.
- A band whose points come from the GPU still costs the CPU 75% of a
  band that it makes itself (181 against 240 us): most of the stage is
  the cluster tables, not the pixel work.
- A command buffer starts 0.1 to 1.3 ms after the commit.

Debug variables: `APRILTAG_GPU_PROF=1` (the GPU time of each kernel),
`APRILTAG_GPU_TRACE=1` (the timing of the region and of the band tasks),
`APRILTAG_GPU_FORCE=<part>` (tests: with threads, the GPU gets this part
of the image and the tasks wait for it), `APRILTAG_GPU_COPY=1` (copy the
image to the GPU; the default uses its pages in place).
