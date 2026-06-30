# XGBoost insider-detection: fine-tuning history

All runs use the same source data (`cache/training_data.parquet`, 793 rows, 117
positives, 29 markets). Features come from the top-K of
`cb_insider_30feat_latest.meta.json` (ranked by CatBoost feature importance).

## Headline comparison (test = `us-strikes-iran-by-march-1-2026-492`, 245 rows / 35 pos)

| # | Run | Feats | CV PR-AUC | Test PR-AUC | Test ROC | P@0.5 | R@0.5 | F1@0.5 |
|---|---|---|---|---|---|---|---|---|
| 0a | xgb 14-feat (notebook 80/20) | 14 | – | 0.6744 | 0.8900 | – | – | – |
| 0b | xgb 30-feat (notebook 80/20) | 30 | – | 0.8362 | 0.9469 | – | – | – |
| 0c | xgb 50-feat (notebook 80/20) | 50 | – | 0.8305 | 0.9341 | – | – | – |
| 1 | xgb2 top-10, random split | 10 | 0.7085 | 0.7080 | 0.9218 | 0.622 | 0.800 | 0.700 |
| 2 | xgb2 top-10, pinned iran-mar-1 | 10 | 0.6616 | 0.9491 | 0.9944 | 0.745 | 1.000 | 0.854 |
| 3 | xgb2 + tightened Optuna (step 2a) | 10 | 0.6484 | 0.9722 | 0.9976 | 0.648 | 1.000 | 0.787 |
| 4 | xgb2 + OOF Platt cal (step 2b) | 10 | 0.6484 | 0.9722 | 0.9976 | 0.714 | 1.000 | 0.833 |
| 5 | xgb2 + StratifiedGroupKFold (step 2c) | 10 | 0.6589 | 0.9692 | 0.9972 | 0.761 | 1.000 | **0.864** |
| 6 | xgb2 top-30 (step 2d) | 30 | 0.6695 | 0.8391 | 0.9854 | 0.000 | 0.000 | 0.000 |
| 7 | xgb3 top-20 (with cal) | 20 | 0.6534 | **0.9969** | **0.9995** | 0.000 | 0.000 | 0.000 |
| 8 | xgb4 top-20 (NO cal) | 20 | 0.6534 | 0.9969 | 0.9995 | 0.000 | 0.000 | 0.000 |

## Reference (non-XGBoost models, same test market)

| Model | Test PR-AUC | Test ROC | F1@thr |
|---|---|---|---|
| **cb3** (catboost 10-feat v2) | 0.9661 | 0.9947 | **0.933** |
| cb4 (catboost 10-feat v2 d2) | 0.9314 | 0.9928 | 0.921 |
| cb2 (catboost 30-feat v2) | 0.8564 | 0.9827 | 0.244 |
| rf (random forest 10-feat v2) | 0.9702 | 0.9959 | 0.864 |

## What changed between runs

| # | Run | Change vs previous | Best params | Outcome |
|---|---|---|---|---|
| 1 | xgb2 random split | first Optuna run (100 trials, GroupKFold) | depth=7, lr=0.069, n_est=500 | overfit (train-test gap +0.27); test set wasn't comparable to cb3 |
| 2 | xgb2 pinned iran-mar-1 | only pinned test market (same code, same Optuna) | depth=5, lr=0.019, n_est=100 | PR-AUC jumped 0.708 → 0.949; proved diff was test, not model |
| 3 | xgb2 step 2a — tighter Optuna | depth ≤5, lr ∈ [0.005, 0.1], n_est ≥200, higher reg floors | depth=5, lr=0.045, n_est=1000 | PR-AUC → 0.972 |
| 4 | xgb2 step 2b — OOF Platt cal | added LogisticRegression on OOF raw probs | (same model as 3) | F1 0.787 → 0.833; PR-AUC unchanged (Platt is monotonic) |
| 5 | xgb2 step 2c — StratifiedGroupKFold | replaced GroupKFold with stratified+shuffle | depth=3, lr=0.014, n_est=1050 | **F1 → 0.864** (best); recall held at 1.000 |
| 6 | xgb2 step 2d — top-30 | only TOP_K=30 | depth=4, lr=0.006, n_est=300 | regression: PR-AUC 0.972 → 0.839, F1 collapsed → 0 |
| 7 | xgb3 top-20 (with cal) | new file, TOP_K=20, rest identical to 5 | depth=2, lr=0.007, n_est=300, gamma=3.4 | **PR-AUC 0.997** (best); F1=0 (same root cause as 6) |
| 8 | xgb4 top-20 (NO cal) | removed calibrator from xgb3 | (identical params) | identical test PR-AUC/F1; **proved calibrator wasn't the cause of F1=0** |

## Diagnostic data behind the F1=0 runs

| # | Run | Train raw prob range | Test raw prob range | Why F1@0.5 = 0 |
|---|---|---|---|---|
| 6 | xgb2 top-30 | (Platt-applied) | (Platt-applied) | calibrator + low raw probs → cal probs < 0.5 |
| 7 | xgb3 top-20 (cal) | (Platt-applied) | (Platt-applied) | sigmoid(4.62·raw − 3.51) ≥ 0.5 needs raw ≥ 0.76; no test row reaches that |
| 8 | xgb4 top-20 (nocal) | [0.071, 0.818] | **[0.071, 0.394]** | max test raw prob = 0.394 < 0.5; calibrator was a red herring |

## Key takeaways

| # | Lesson | Evidence |
|---|---|---|
| 1 | Pinned test market is essential for cross-model comparison | Run 1 vs Run 2: PR-AUC 0.708 → 0.949 from test split alone |
| 2 | Tighter Optuna ranges fix the overfit | Train-test gap +0.27 (Run 1) → near-0 from Run 3 onward |
| 3 | OOF Platt cal helps F1 only when raw probs straddle 0.5 | Run 4 (helps), Runs 6–8 (useless, raw probs all low) |
| 4 | StratifiedGroupKFold is a small but real CV stability win | CV PR-AUC 0.648 → 0.659 (Run 3 → Run 5) |
| 5 | More features ≠ better | Top-10 best F1, top-20 best PR-AUC, top-30 collapses both |
| 6 | F1=0 in 6/7/8 is a *thresholding* problem, not modelling | Ranking is near-perfect (PR-AUC 0.997, ROC 0.999); thr=0.5 just sits above all test raw probs |
| 7 | Best PR-AUC and best F1 are in different runs | Run 7 best PR-AUC (0.997); Run 5 best F1 (0.864) |

## Open follow-ups

| Idea | What it would do | Expected effect |
|---|---|---|
| Pick threshold from train OOF (e.g., ~0.2) | Re-evaluate Runs 6/7/8 at a data-driven threshold | Unlock F1 without changing model |
| Loosen Optuna ranges toward less conservative configs | Allow higher `scale_pos_weight`, drop `gamma` ceiling | Raw probs less compressed, F1 recovers at thr=0.5 |
| Switch to top-K precision/recall metric | Drop thresholded F1 entirely | Natural fit for insider-detection use case; PR-AUC-leading runs win |
