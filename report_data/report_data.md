# Phase 1 report data — Team Libro (simulation, 2026-09-15)

Source: the solution's own per-trial reports (`erc_images/trial_*.json`) and annotated
evidence images, from development runs on the official v1.0.0 simulator with the
corrected 30 mm book. **These were iterative debugging runs with the code changing
between them — not a controlled 5-trial benchmark.** Times are wall-clock from launch,
at a simulation real-time factor of roughly 0.25–0.4.

## Detection accuracy

| Metric | Result |
|---|---|
| Trials that reached shelf identification | 15 of 17 (the others failed during arm stow) |
| Shelf column marker read as the requested number ("2") | 15/15 (100%) |
| Marker template-match confidence | mean 0.54, min 0.41, max 0.73 |
| Red target book located in the identified column | 15/15 (100%) |
| Column + row + colour confirmed by physical grip on the named book | 7/7 (100%) of the 7 grips |

A grip is only confirmed on finger contact with the Gazebo model named for the detected
column, row and colour (e.g. `book_col_4_row_2_red`), so each confirmed grip independently
verifies that detection. Annotated evidence images: `evidence_samples/`.

## Navigation and timing

| Metric | Result |
|---|---|
| Nav2 goals completed | 45 |
| Duration per goal | mean 37 s (min 14, max 132), n=45 |
| Arrival position error | mean 49 mm, max 80 mm |
| Arrival heading error | mean 0.096 rad (Nav2 goal tolerance 0.10 rad) |
| Heading error after in-place square-up | mean 0.0145 rad, n=7 |
| Launch → shelf column identified | mean 145 s (min 107, max 264), n=15 |
| Launch → target book identified | mean 207 s (min 159, max 369), n=15 |
| Launch → grip confirmed | mean 325 s (min 291, max 350), n=7 |
| Launch → book extracted | mean 461 s (min 380, max 678), n=4 |

Planned (Nav2 `/plan`, blue) vs executed (odometry, red) paths for each trial: `path_plots/`.

## Manipulation

| Stage | Result |
|---|---|
| Grasp attempted (book re-observed from the grasp pose) | 15 trials |
| Grip confirmed | 7/15 (47%) of attempts |
| Grip on the first closure, no re-alignment | 7/7 (100%) of grips |
| Book fully extracted from the shelf | 4/7 (57%) of grips |
| Book stowed for transport | 0/4 saved reports; plus 1 run stopped manually for debugging (no report saved) that stowed the book at 493 s and drove back to the start zone before the book was found to have dropped |
| Book placed in the collection bin | 0/17 (0%) of trials |

Note: trials 1–10 predate today's gripper fix (opening to 0.065 m instead of the jamming 0.070 m limit); every grip after that fix closed on the first attempt.

## Last failure per trial

| Cause | Trials |
|---|---|
| Action failed: status=6 | 5 |
| Unintended contact during travel/inspection | 2 |
| Target book was not retained in transport pose | 2 |
| Incomplete collision-free Cartesian path: 0.000 | 1 |
| Action timeout; cancelled | 1 |
| Sustained right-finger contact requires lateral alignment | 1 |
| Incomplete collision-free Cartesian path: 0.909 | 1 |
| Incomplete collision-free Cartesian path: 0.055 | 1 |
| Incomplete collision-free Cartesian path: 0.636 | 1 |
| Target book was not retained during extraction segment 2 | 1 |
| Action acknowledgement timeout | 1 |

## Per-trial table

| # | Marker (conf) | Column | Row | Grip (s) | Extracted (s) | Furthest stage | Outcome |
|---|---|---|---|---|---|---|---|
| 1 | 2 (0.55) | 3 | 4 | – | – | BOOK_IDENTIFIED | FAILED: Incomplete collision-free Cartesian path: 0.000 |
| 2 | – | – | – | – | – | STARTING | FAILED: Action timeout; cancelled |
| 3 | – | – | – | – | – | STARTING | FAILED: Action failed: status=6 |
| 4 | 2 (0.71) | 5 | 4 | – | – | BOOK_IDENTIFIED | FAILED: Action failed: status=6 |
| 5 | 2 (0.43) | 4 | 3 | – | – | BOOK_IDENTIFIED | FAILED: Sustained right-finger contact requires lateral alignment |
| 6 | 2 (0.55) | 1 | 3 | 323 | – | GRIP_CONFIRMED | FAILED: Incomplete collision-free Cartesian path: 0.909 |
| 7 | 2 (0.72) | 5 | 2 | – | – | BOOK_IDENTIFIED | FAILED: Action failed: status=6 |
| 8 | 2 (0.50) | 4 | 4 | – | – | BOOK_IDENTIFIED | FAILED: Action failed: status=6 |
| 9 | 2 (0.41) | 4 | 2 | – | – | BOOK_IDENTIFIED | FAILED: Incomplete collision-free Cartesian path: 0.055 |
| 10 | 2 (0.62) | 2 | 3 | – | – | BOOK_IDENTIFIED | FAILED: Unintended contact during travel/inspection |
| 11 | 2 (0.73) | 5 | 4 | 322 | – | GRIP_CONFIRMED | FAILED: Incomplete collision-free Cartesian path: 0.636 |
| 12 | 2 (0.42) | 4 | 2 | 350 | 396 | BOOK_GRASPED | FAILED: Target book was not retained in transport pose |
| 13 | 2 (0.42) | 4 | 4 | 291 | – | GRIP_CONFIRMED | FAILED: Target book was not retained during extraction segment 2 |
| 14 | 2 (0.55) | 1 | 4 | 342 | 678 | BOOK_GRASPED | FAILED: Unintended contact during travel/inspection |
| 15 | 2 (0.49) | 4 | 1 | 330 | 391 | BOOK_GRASPED | FAILED: Action failed: status=6 |
| 16 | 2 (0.41) | 4 | 2 | 320 | 380 | BOOK_GRASPED | FAILED: Target book was not retained in transport pose |
| 17 | 2 (0.61) | 2 | 1 | – | – | BOOK_IDENTIFIED | FAILED: Action acknowledgement timeout |
