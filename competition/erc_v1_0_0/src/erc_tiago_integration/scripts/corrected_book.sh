#!/usr/bin/env bash
# Local validation aid — NOT part of the Phase 1 submission, and NOT used by
# solution.launch.py.
#
# Simulator release v1.0.0 ships erc_book.sdf with a 60 mm collision box, but
# the PAL Pro gripper's fingertip contact pads span only 59.72 mm at the
# gripper_left_finger_joint upper limit of 0.070 m. The jaws cannot enclose
# that book, so no grasp is physically possible on v1.0.0. The organizers
# corrected the book to 30 mm in a later simulator revision.
#
# This script swaps the corrected thickness in so the grasp pipeline can be
# validated end to end, and swaps it back out again. The committee evaluates
# against their own unmodified packages, so the tree must be restored before
# committing or submitting.
#
#   corrected_book.sh apply     # 60 mm -> 30 mm book
#   corrected_book.sh restore   # back to the shipped v1.0.0 book
#   corrected_book.sh status    # report which book is installed
set -euo pipefail

SHIPPED_THICKNESS=0.06
CORRECTED_THICKNESS=0.03

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOK="${HERE}/../../erc_description/models/book/sdf/erc_book.sdf"
[ -f "$BOOK" ] || { echo "cannot locate erc_book.sdf at $BOOK" >&2; exit 1; }

thickness() {
    sed -n 's|.*<size>[0-9.]* \([0-9.]*\) [0-9.]*</size>.*|\1|p' "$BOOK" | head -1
}

retarget() {
    local from="$1" to="$2" current
    current="$(thickness)"
    if [ "$current" = "$to" ]; then
        echo "already at ${to} m"
        return 0
    fi
    if [ "$current" != "$from" ]; then
        echo "unexpected book thickness ${current} m; refusing to edit" >&2
        exit 1
    fi
    sed -i "s|<size>\\(0.25\\) ${from} \\(0.16\\)</size>|<size>\\1 ${to} \\2</size>|g" "$BOOK"
    echo "book thickness ${from} m -> $(thickness) m"
}

case "${1:-status}" in
    apply)
        retarget "$SHIPPED_THICKNESS" "$CORRECTED_THICKNESS"
        echo "REMINDER: run 'corrected_book.sh restore' before committing."
        ;;
    restore)
        retarget "$CORRECTED_THICKNESS" "$SHIPPED_THICKNESS"
        ;;
    status)
        current="$(thickness)"
        echo "installed book thickness: ${current} m"
        if [ "$current" = "$SHIPPED_THICKNESS" ]; then
            echo "tree state: shipped v1.0.0 book (submission-clean)"
        else
            echo "tree state: MODIFIED — restore before committing or submitting"
        fi
        ;;
    *)
        echo "usage: $0 {apply|restore|status}" >&2
        exit 2
        ;;
esac
