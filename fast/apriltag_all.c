/* All of the library but the family tables, as one unit: the compiler
   can inline across the stages (as Goose does with its one C file). */
/* No FP contraction (FMA): the output must not depend on it. This also
   holds where the build cannot pass -ffp-contract=off (SwiftPM). */
#pragma STDC FP_CONTRACT OFF
#ifdef __clang__
#pragma clang fp contract(off)
#endif

#include "common.h"

#include "util.c"
#include "image.c"
#include "threshold.c"
#include "unionfind.c"
#include "clusters.c"
#include "quads.c"
#include "decode.c"
#include "pool.c"
#include "detector.c"
