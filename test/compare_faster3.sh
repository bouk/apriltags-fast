#!/usr/bin/env bash
# Compare apriltags-fast with the faster3 C library, through the upstream
# API, bit for bit (hex floats, quad counts), over a matrix of parameters.
#
# Usage: test/compare_faster3.sh <faster3-src> <faster3-lib> <fast-lib-dir> [images...]
#   faster3-src   a checkout of BuildMonumental/apriltag, branch faster3
#   faster3-lib   its libapriltag.a, built with -march=native -ffp-contract=off
#                 and with workerpool_run always calling workerpool_run_single
#                 (with real threads, the faster3 tasks race and its output
#                 changes from run to run)
#   fast-lib-dir  the build directory of apriltags-fast (libapriltag.so)
set -u
cd "$(dirname "$0")/.."
F3SRC=$1; F3LIB=$2; FAST=$3; shift 3
IMAGES=("$@"); [ ${#IMAGES[@]} -eq 0 ] && IMAGES=(test/images/*.pgm)
CC=${CC:-cc}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
$CC -O2 -I"$F3SRC" test/api_dump.c -o "$TMP/ref" "$F3LIB" -lm -lpthread || exit 1
$CC -O2 -I. test/api_dump.c -o "$TMP/fast" -L"$FAST" -lapriltag -Wl,-rpath,"$(cd "$FAST" && pwd)" -lm -lpthread || exit 1
pass=0; fail=0
for fam in tag36h11 tag25h9 tagStandard41h12 tagStandard52h13; do
    for x in 1 1.5 2 3; do
        for t in 1 4 16; do
            "$TMP/ref" $t 1 $fam $x "${IMAGES[@]}" > "$TMP/r.txt" 2>/dev/null
            "$TMP/fast" $t 1 $fam $x "${IMAGES[@]}" > "$TMP/o.txt" 2>/dev/null
            if cmp -s "$TMP/r.txt" "$TMP/o.txt"; then
                pass=$((pass+1))
            else
                fail=$((fail+1)); echo "FAIL $fam -x $x -t $t"
                diff "$TMP/r.txt" "$TMP/o.txt" | head -4
            fi
        done
    done
done
echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]
