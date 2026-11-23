# OBSCURA – Site-Aware Satellite Prediction| V02 

**Team: Obscura Nexus Quartet**

| Public LB | Honest validation (unseen sites) | Predict-0 baseline |
| --- | --- | --- |
| **1.01690** | **0.8483** OVERALL MCRMSE | 0.9971 |

This repository holds the notebook of our single  model (**V02**) for the OBSCURA satellite prediction competition on Kaggle. It also tells the story of how we got there: what we tried, why, what worked, and what did not.

> The model uses **only the competition data**. No pretrained models and no external data.

---

## Contents

| File | What it is |
| --- | --- |
| `Site-Aware Satellite Prediction| V02.ipynb` | The notebook: approach and the code of the V02 model |
| `README.md` | This file |

The competition data is **not** included. It is available on the Kaggle competition page, under the competition rules.

---

## The competition in short

* **Task:** for each sample we get one small satellite image (a *chip*, 14 layers: 12 Sentinel-2 bands, a scene-classification layer, a cloud-probability layer) and 29 table columns (weather, terrain, month). We predict **five targets**: `t_ch4`, `t_no2`, `t_co`, `t_aai`, `t_firms`.
* **Data:** 3218 training rows from 81 sites. 1221 test rows from **new sites**.
* **Score:** MCRMSE (mean of the five RMSE values, lower is better). Blank target values are skipped.
* **Difficulty:** only 81 sites, and rows from one site look alike, so a flexible model can memorise a site instead of the signal. `t_ch4` and `t_co` also have almost no learnable signal.

---

## Our approach

1. **Chips to numbers.** We compute simple statistics from clear pixels: band statistics, hot-pixel descriptors (brightest infrared pixels), and urban-texture descriptors.
2. **One feature table per target.** Each target gets only the feature groups that helped it in validation.
3. **Small, simple models.** Histogram Gradient Boosting, Ridge, CatBoost, and a two-stage "high / low" model for fires. Random-subspace copies are used for `t_no2`.
4. **Shrinkage.** Each prediction is multiplied by a factor between 0 and 1 (learned from site-wise out-of-fold predictions), so hard targets stay close to zero.
5. **Honest testing.** Whole sites stay together in validation. Every change is judged by a rule written before the run. The public leaderboard is only an independent check.

### Model per target (v02)

> **Note:** this repository shares our **approach**. v02 is our single model, but this README and notebook are not our final submission file or final selection.

| Target | Members (averaged) | Extra features | Shrinkage | Honest RMSE | Predict-0 |
| --- | --- | --- | --- | --- | --- |
| `t_ch4` | HGB + Ridge | none | 0.438 | 0.9988 | 1.0001 |
| `t_no2` | HGB + Ridge, 8 random-subspace copies each | urban texture + P1 | 0.944 | 0.8915 | 0.9964 |
| `t_co` | HGB + Ridge | none | 0.481 | 0.9869 | 0.9971 |
| `t_aai` | HGB + Ridge | v16 + A2 | 1.000 | 0.5760 | 0.9988 |
| `t_firms` | Two-stage + Ridge + CatBoost | hot-pixel families, 130 columns removed (D1) | 1.000 | 0.7882 | 0.9931 |
| **OVERALL** | | | | **0.8483** | **0.9971** |

---

## Notes and limitations

* **Validation vs leaderboard.** The validation score (0.8483) and the leaderboard score (1.01690) are measured on **different sites**, so their absolute sizes cannot be compared. We compare *changes*.
* **What is reproduced and what is imported.** Running the notebook reproduces the feature extraction, the feature matrices and the v02 test predictions. The honest validation numbers and the default shrinkage factors come from separate evaluation runs and are stored as fixed values. Setting `RECOMPUTE_FACTORS = True` recomputes the factors (about 1 to 2 hours).
* **Hypotheses.** Our explanations for why features help (wind, urban texture, hot pixels) are hypotheses, not proven mechanisms.
* **Limits.** `t_ch4` and `t_co` stay close to predicting zero. The validation score is an estimate and does not guarantee the private leaderboard result.

---

## How to run

1. Open the notebook on Kaggle and add the OBSCURA competition data as input.
2. Run all cells.
3. The notebook writes `submission_v138_model.csv` and prints the OVERALL MCRMSE. It never overwrites `submission.csv`.

**Runtime:** about 28 minutes on a Kaggle CPU session in our run (no GPU). Most of this is reading the chips. Chip features are saved as CSV files, so later runs are faster.

**Libraries:** Python 3, NumPy, pandas, SciPy, scikit-learn, CatBoost, joblib, threadpoolctl.

---

## Team Members

| Name | Kaggle |
| --- | --- |
| Maria Nadeem | [Kaggle](https://www.kaggle.com/mnkaggler) |
| Mehak Iftikhar | [Kaggle](https://www.kaggle.com/mehakiftikhar) |
| Maria Aqdas | [Kaggle](https://www.kaggle.com/mariaaqdas) |
| Anila Bukhari | [Kaggle](https://www.kaggle.com/anilabukhari) |

<p align="center">— Team Obscura Nexus Quartet</p>
