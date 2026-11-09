/* All of the library but the family tables, as one unit: the compiler
   can inline across the stages (as Goose does with its one C file). */
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
