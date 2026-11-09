#!/usr/bin/env bash
# Golden test: the output of the fast demo, in hex floats, must be the
# same as test/golden (made with the C library and -ffp-contract=off).
# Usage: test/run_golden.sh <apriltag_fast_demo>
set -u
cd "$(dirname "$0")/.."
DEMO=$1
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
while IFS=$'\t' read -r name args; do
    [ -z "$name" ] && continue
    # shellcheck disable=SC2086
    "$DEMO" -q $args --hex-detections "$TMP/$name.hex" test/images/*.pgm > "$TMP/$name.out" 2>&1
    if cmp -s "test/golden/$name.hex" "$TMP/$name.hex"; then
        pass=$((pass+1)); echo "ok   $name"
    else
        fail=$((fail+1)); echo "FAIL $name ($args)"
        diff "test/golden/$name.hex" "$TMP/$name.hex" | head -5
    fi
done < test/cases.txt
echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]
