# USW/ASW significance recovery log

| Date | Event | Evidence / decision |
|---|---|---|
| 2026-07-17 | Scope opened | Three gaps remain: FFSP50 USW has wrong/tiny effects with about 65% ties; JSSP15x15 USW is 2.31 worse than BOPO; CVRP50 ASW beats all hand-designed objectives but is 0.01322 worse than USW. |
| 2026-07-17 | Recovery candidates locked | Completed validation-selected recovery runs produced FFSP50 USW epoch117 (SHA `0b52e9de...fab1d`) and JSSP15x15 USW epoch13 (SHA `93b79af7...ac3ff`). Neither is selected from paper-test performance. |
| 2026-07-17 | CVRP diagnosis | Existing ASW LR5e-5 polish candidates have worse validation rewards (-10.443707 and -10.446414) than the paper epoch548 candidate (-10.442578), so they will not be promoted to the paper test. The next screen uses the stronger USW representation followed by the unchanged ASW objective and is explicitly labeled a variant. |
