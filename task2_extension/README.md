Task 2 extension — cross-validation, learning curves, and Fairlearn bias audit
This folder extends the Task 1 analysis in this repository (`code/analyze_oulad.py`,
`code/analyze_student_performance.py`) for Individual Task 2, Part 2 (Deliberation).
`verify_uci.py`, `verify_oulad.py` — re-run the original Task 1 pipelines unmodified,
to confirm the originally reported numbers before extending them.
`part2_analysis.py` — the extension itself: 5-fold stratified cross-validation,
learning curves, and a Fairlearn group-fairness audit (selection rate, demographic
parity difference, equalized odds difference) for both datasets.
`outputs/figures/` — the learning curve plots and the OULAD selection-rate-by-IMD-band
chart referenced in the Task 2 report.
`outputs/results/` — the Fairlearn per-group metric tables (CSV) underlying Table 2
and the fairness discussion in the report.
