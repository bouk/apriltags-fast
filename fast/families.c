/* The built-in families of the fast API, made once from the upstream
   family files (tag*.c), so that the code tables exist only
   once in the library. */
#include <pthread.h>
#include <string.h>

#include "apriltag_fast.h"
#include "../apriltag.h"
#include "../tag16h5.h"
#include "../tag25h9.h"
#include "../tag36h10.h"
#include "../tag36h11.h"
#include "../tagCircle21h7.h"
#include "../tagCircle49h12.h"
#include "../tagCustom48h12.h"
#include "../tagStandard41h12.h"
#include "../tagStandard52h13.h"

bool at_family_from_apriltag(const apriltag_family_t *src, at_family *dst) {
    if (src->nbits > 64) return false;
    memset(dst, 0, sizeof *dst);
    strncpy(dst->name, src->name ? src->name : "", sizeof dst->name - 1);
    dst->codes = src->codes;
    dst->ncodes = src->ncodes;
    dst->nbits = (int32_t)src->nbits;
    dst->h = (int32_t)src->h;
    dst->width_at_border = src->width_at_border;
    dst->total_width = src->total_width;
    dst->reversed_border = src->reversed_border;
    for (uint32_t i = 0; i < src->nbits; i++) {
        dst->bit_x[i] = (int32_t)src->bit_x[i];
        dst->bit_y[i] = (int32_t)src->bit_y[i];
    }
    return true;
}

typedef struct {
    const char *name;
    apriltag_family_t *(*create)(void);
    apriltag_family_t *upstream; /* kept: it owns the codes */
    at_family fam;
} Builtin;

#define B(n) {.name = #n, .create = n##_create}
static Builtin builtins[] = {
    B(tag16h5),          B(tag25h9),          B(tag36h10),          B(tag36h11),
    B(tagCircle21h7),    B(tagCircle49h12),   B(tagCustom48h12),    B(tagStandard41h12),
    B(tagStandard52h13),
};
#undef B

static pthread_mutex_t builtins_lock = PTHREAD_MUTEX_INITIALIZER;

static const at_family *builtin(Builtin *b) {
    pthread_mutex_lock(&builtins_lock);
    if (!b->upstream) {
        b->upstream = b->create();
        at_family_from_apriltag(b->upstream, &b->fam);
    }
    pthread_mutex_unlock(&builtins_lock);
    return &b->fam;
}

const at_family *at_family_by_name(const char *name) {
    for (size_t i = 0; i < sizeof builtins / sizeof builtins[0]; i++)
        if (!strcmp(builtins[i].name, name)) return builtin(&builtins[i]);
    return NULL;
}

const at_family *at_tag16h5(void) { return builtin(&builtins[0]); }
const at_family *at_tag25h9(void) { return builtin(&builtins[1]); }
const at_family *at_tag36h10(void) { return builtin(&builtins[2]); }
const at_family *at_tag36h11(void) { return builtin(&builtins[3]); }
const at_family *at_tagCircle21h7(void) { return builtin(&builtins[4]); }
const at_family *at_tagCircle49h12(void) { return builtin(&builtins[5]); }
const at_family *at_tagCustom48h12(void) { return builtin(&builtins[6]); }
const at_family *at_tagStandard41h12(void) { return builtin(&builtins[7]); }
const at_family *at_tagStandard52h13(void) { return builtin(&builtins[8]); }
