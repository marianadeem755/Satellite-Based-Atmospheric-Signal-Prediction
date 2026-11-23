# OBSCURA: Satellite AI Research Challenge

**Learning Events, Not Places: Predicting Five Atmospheric Targets at Unseen Industrial Sites**

**Kaggle team: Obscura Nexus Quartet**

A complete pipeline: Sentinel-2 style image chips become compact feature tables, small target-specific model teams are trained under strict site-level validation, weak predictions are shrunk toward zero, and a checked submission file is written.

| | |
|---|---|
| Team | Obscura Nexus Quartet |
| Targets | 5 (`t_ch4`, `t_no2`, `t_co`, `t_aai`, `t_firms`) |
| Training data | 3,218 rows from 81 sites |
| Test data | 1,221 rows from **new sites**, later in time |
| Inputs | 29 tabular columns + one 14-layer image chip per row |
| Metric | MCRMSE (mean of the 5 per-target RMSEs, lower is better) |
| Models | scikit-learn, CatBoost, SciPy (LightGBM and XGBoost are also imported). No pretrained models, no external data |

## Contents

1. [Problem](#1-problem)
2. [Data](#2-data)
3. [Approach in one picture](#3-approach-in-one-picture)
4. [Features](#4-features)
5. [Models](#5-models)
6. [Validation protocol](#6-validation-protocol)
7. [Results](#7-results)
8. [What worked and what did not](#8-what-worked-and-what-did-not)
9. [Limitations](#9-limitations)
10. [How to run](#10-how-to-run)
11. [Team and rules](#11-team-and-rules)

---

## 1. Problem

**Motivation.** Methane is one of the fastest ways to slow near-term warming, and many methane leaks come from oil, gas and coal sites. Those same sites burn and vent other things as well, so nitrogen dioxide, carbon monoxide, smoke and heat appear in the same places, and the same winds carry all of them. Satellites are the only tool that covers every site on a regular basis, but a satellite measurement is only useful on days when it works: clouds block the view, quality filters throw away unreliable pixels, and the signal from one facility is small next to the measurement noise.

**Sentinel-2** takes sharp pictures of the ground at 10 to 60 m, showing bright flares, bare ground and built-up areas. The gas instrument that provides the targets, **TROPOMI on Sentinel-5P**, sees pixels several kilometres wide (about 7 by 7 km for methane and CO). OBSCURA puts the two kinds of data in one task and asks how much a **Sentinel-2 image of the site and the weather and terrain around it** can say about the gases, for sites the model has never seen and in a later period.

**The task.** Each row is one **sample**: a site at a point in time. From weather, terrain, calendar columns and one 14-layer satellite chip, we predict five standardized quantities that other satellites measured over the same place and days:

| Target | Signal |
|---|---|
| `t_ch4` | Methane |
| `t_no2` | Nitrogen dioxide |
| `t_co` | Carbon monoxide |
| `t_aai` | Aerosol index |
| `t_firms` | Fire score |

Targets are scaled so that **predicting 0 gives an RMSE of about 1.0** on every target. Zero is therefore the safe baseline, and a prediction is only useful where it beats zero on unseen sites.

**What makes it hard**

- **New places, later period.** Test sites never appear in training. A model that memorizes sites looks excellent on a random split and fails on the test set.
- **Very few sites.** With about 40 rows per site, flexible models fit site-specific noise instead of real physics.
- **Weak signal for methane and CO.** These two make up 40% of the score and carry almost no learnable signal.
- **Missing labels.** A blank target means "not measured" (for example, cloud filtered). Blanks are skipped in training, evaluation and scoring, but all five columns are still predicted for every row.

## 2. Data

| File | Shape | Holds |
|---|---|---|
| `train.csv` | 3218 x 36 | `sample_id`, `site_key` (train only), 29 features, 5 targets |
| `test.csv` | 1221 x 30 | `sample_id` and the same 29 features |
| `sample_submission.csv` | 1221 x 6 | Required output format |
| `images/<sample_id>.tif` | one per row | 14-layer chip |

**Chip layers:** 12 spectral bands (B1 to B12; B11 and B12 are the key shortwave-infrared bands for heat and fire), a scene classification layer (SCL) and a cloud probability layer (CLDPRB).

**The 29 tabular columns** fall into four groups: weather (temperature, humidity, wind speed, precipitation, solar radiation), terrain (elevation, slope), calendar (cyclic month encoding) and supplied image summaries (cloud fraction and 18 image indices).

**Rules the code follows**

- `site_key` exists only in train. It is **never a feature**. It is used only to build site-grouped folds.
- Only **clear pixels** are used: cloud probability below 40 and no zero in B1 to B12, with a safe fallback when fewer than 500 pixels qualify.
- Elevation and slope are kept out of the tree models, because terrain values would help a tree identify a site. Ridge keeps them.
- Assertions stop the run early if anything is off: 29 tabular columns, unique ids, the same id order in test and sample submission, no infinite feature values.

## 3. Approach in one picture

```
 chips (14 layers) ─► clear-pixel mask ─► feature families ─┐
                                                            ├─► per-target matrices ─► model teams
 29 tabular columns ─► weather / season features ───────────┘            │
                                                                         ▼
 site-grouped out-of-fold predictions ─► shrinkage per target ─► candidate selection (adoption rule)
                                                                         │
                                          final fit on all rows ◄────────┘
                                                  │
                                                  ▼
                                   fire floor ─► submission.csv
```

Core idea: **features that describe a place** let a flexible model memorize a site, while **features that describe an event** (a hot pixel, a texture contrast, calm air) can carry over to new sites. Each target gets only the features and models that help on unseen sites.

## 4. Features

All chip statistics are computed from clear pixels and cached to disk, so each chip is processed once.

| Group | Size | What it describes |
|---|---|---|
| Weather and season | 8 | Supplied weather and cyclic month columns |
| Supplied cloud + image indices | 19 | Cloud fraction and 18 supplied indices |
| `img_` chip statistics | 83 | Per-band mean, spread, 10th and 90th percentile; NDVI, NDWI, NBR, SWIR ratio; scene-class and cloud shares |
| `x_` / `a_` ratios | 21 / 7 | Cross-band ratios and variation; ratios of neighboring bands |
| `v2_` haze and texture | 20 | Haze-style ratios and texture measures |
| `v3` hot-pixel base | 16 | Top-1, top-10 and top-100 brightness of B11 and B12, z-score pixel counts |
| `v4` / `v7` | 19 / 15 | Connected hot areas, hot-pixel band ratios and position; B12 thresholds and hot-area share |
| `v16` brightness | 36 | Brightness percentiles and bare-surface measures (hot part: 17 columns) |
| `v17` / `v23` | 10 / 13 | Smoothed hot-pixel ranks and local top values; finer B12 minus B8A rank features |
| `v19` urban texture | 18 | Local contrast of B8, B4 and B11 at three scales, edge density, local std, NDBI |

**Targeted interactions (chosen from physical reasoning, not tested as mechanisms)**

- **P1 (NO2, 5 columns):** 1/wind speed and its product with four urban-texture columns. Calm air keeps NO2 near where it is made.
- **A2 (aerosol, 5 columns):** solar radiation times four surface columns (bare-surface share, share of B12 above 0.30, share of B2 above 0.15, 90th percentile of B1/B2), plus temperature times bare-surface share. Heat over dry ground lifts dust.
- **F1 (fire):** finer rank features (1st, 5th, 25th hottest pixel, and so on).

## 5. Models

Models are small on purpose. With 81 sites a deep model learns each site's average.

| Model | Settings | Role |
|---|---|---|
| Histogram gradient boosting (HGB) | depth 3, 200 iterations, learning rate 0.05, min leaf 40, L2 = 1 | All targets; also random-subspace copies |
| Ridge | alpha 100, median imputation, quantile transform to normal | All targets |
| CatBoost | depth 4, 250 iterations, learning rate 0.06, L2 leaf 5 | Fire |
| SVR | C = 1, epsilon 0.1 | Fire, second member |
| Two-stage fire model | A classifier estimates the chance *p* that a row is in the hot group; prediction = *p* x high-fire HGB + (1 - *p*) x low-fire HGB | Fire |

**Per-target setup**

| Target | Columns | Team | Shrink |
|---|---|---|---|
| `t_ch4` | base set (174) | HGB + Ridge | 0.40 |
| `t_no2` | base + urban texture + P1 (197 full / 67 lean) | HGB + Ridge, full and lean averaged 50/50 | 0.938 |
| `t_co` | base set (174) | HGB + Ridge | 0.488 |
| `t_aai` | base + brightness + A2 (215 full / 85 lean) | HGB + Ridge, full and lean averaged 50/50 | 1.00 |
| `t_firms` | hot-pixel families, D1 pruning (118) | Two-stage + Ridge + CatBoost, averaged with an SVR version | 1.00 |

**D1 pruning for fire.** The fire model drops 130 whole-image columns (83 `img_`, 21 `x_`, and 26 "other" columns holding the 7 `a_` ratios, the cloud fraction and the 18 image indices; 132 for Ridge, which also drops elevation and slope). What stays: weather and season, haze and texture, and the hot-pixel families.

**Lean copies (v102).** For `t_no2` and `t_aai`, dropping the same columns outright hurt, so the final model averages a full and a lean copy.

**Training safeguards**

- Targets are clipped at the 1st and 99th percentile *inside each training fold only*.
- Imputation and transforms are fitted on the training part of each fold only.
- **Shrinkage:** one multiplier per target, picked from 21 values between 0 and 1 on out-of-fold predictions, blends the prediction with 0.
- **Fire floor:** `t_firms` cannot go below the no-detection level (the training minimum, -0.439). A value below it is farther from every possible truth than the floor itself, so clipping can only reduce error.

## 6. Validation protocol

Rows from one site share weather, terrain and surface, so ordinary cross-validation leaks site identity. **Every number that guides a decision here comes from splits that hold out whole sites.**

- **Out-of-fold predictions** for shrinkage come from site-grouped 5-fold splits, with a check that no site is in both train and validation. Sites are weighted by 1/sqrt(rows) so large sites do not dominate.
- **Half-site comparisons:** split the 81 sites into two random halves, choose settings on one half and score on the other. 4 partitions x 20 splits x 2 directions = **160 runs**. Every candidate sees identical splits.
- **Adoption rule, fixed before looking at results.** An idea is kept only if overall MCRMSE improves by at least **0.001**, in at least **75%** of runs, and by at least 0.001 in **each of two partition groups**. A pass is only a candidate and must repeat on fresh partitions.
- **Wider ensembles (v103).** Four candidates (W103a to W103d) with more members per target replace the v102 model only if they beat it by at least 0.0005.

## 7. Results

**Local, on held-out sites** (from the project write-up)

| Run | What it is | MCRMSE |
|---|---|---|
| Predict 0 | Baseline | 0.9967 |
| E1 | HGB + Ridge + two-stage fire + hot-pixel features + A2 | 0.8602 |
| E2 | E1 + D1 pruning for fire | 0.8549 |
| E3 | E2 + 50/50 lean average for `t_aai`, `t_no2` (v102) | 0.8548 |
| E4 | Final single model v138 (larger evaluation, predict-0 = 0.9971) | 0.8483 |

**Per target, final single model v138**

| Target | Predict 0 | v138 | Skill vs zero |
|---|---|---|---|
| `t_ch4` | 1.0001 | 0.9988 | about 0% |
| `t_no2` | 0.9964 | 0.8915 | 10.5% |
| `t_co` | 0.9971 | 0.9869 | 1.0% |
| `t_aai` | 0.9988 | 0.5760 | 42.3% |
| `t_firms` | 0.9931 | 0.7882 | 20.6% |
| **Overall** | 0.9971 | **0.8483** | 14.9% |

**Public leaderboard**

| Submission | Score |
|---|---|
| v102 (this notebook's model) | 1.01807 |
| v138 (final single model) | 1.01690 |
| Separate pipeline (v14) | 1.01761 |
| Per-target blend of the three files (best) | **1.01510** |

The blend's validation gain over v138 alone is only about 0.0001, so part of its leaderboard gain is probably noise. It is submitted next to v138 alone, never as the only submission.

## 8. What worked and what did not

**Worked**

- Target-specific features and models instead of one recipe for all five targets.
- Dropping 130 whole-image columns for fire (gain of 0.005 to 0.007 in four separate site-grouped checks).
- Shrinking weak targets toward 0, and clipping fire at its floor.

**Did not pass the adoption rule**

- Bigger or differently tuned tree models, other blend recipes, stacking, and feeding one target's prediction into another.
- 15 weather interactions, residual hot-spot features (v11 to v14), normalized-difference families (v13), anisotropy features (v28), wind direction and chip-centre statistics.
- Extra shrinking or gating by model disagreement, and isotonic calibration.
- Dropping the same columns for `t_aai`, `t_no2` or `t_co`.

## 9. Limitations

- All searches reuse the same 81 sites, and held-out sites come from the same period as training. The protocol **cannot measure the shift to the later test period**.
- Methane and CO stay near zero skill. The models themselves expect very little (strong shrinkage, small prediction spread).
- The "place vs event" explanation for D1 fits the numbers but is **not proven**; no site-scrambling test was run.
- Two candidates that passed the local rule (W3 and E13) did not help on the public leaderboard.
- The physical reasons behind P1 and A2 are our motivation for choosing the features, not tested mechanisms.

## 10. How to run

1. Open the notebook on Kaggle and attach the competition data (`train.csv`, `test.csv`, `sample_submission.csv`, `images/`).
2. Keep `FAST = False` (the default). `FAST` is only a smoke-test switch.
3. Run all cells. Chip feature tables are cached, so later runs are much faster.
4. The notebook writes `submission.csv`, using the best candidate by held-out score, with the fire floor applied.

Dependencies: Python, NumPy, pandas, SciPy, scikit-learn, CatBoost, joblib (LightGBM and XGBoost are imported by the notebook).

## 11. Team and rules

**Team:** Obscura Nexus Quartet (Maria Nadeem, Mehak Iftikhar, Anila Bukhari, Maria Aqdas)

We used competition data only: no outside satellite, weather, air-quality, methane, fire or elevation data. We did not try to identify facilities, locations or dates, and we did not probe the leaderboard. All libraries are open source and trained on competition data only. No pretrained model is used in any submission.
