# SLTB vs Private Bus Preference Prediction

## Project Overview

This project develops a machine learning system to predict whether a passenger is more likely to prefer an SLTB bus or a Private bus in Sri Lanka. It uses a primary survey (724 responses) covering passenger demographics, trip characteristics, and service-quality ratings to train and evaluate several classification models.

## Objectives

- Predict passenger bus preference (SLTB vs. Private).
- Compare five machine learning algorithms under a robust, repeated cross-validation protocol.
- Identify and quantify a proxy-feature risk in the dataset before trusting the headline accuracy.
- Develop a working prediction application using Streamlit.

## Dataset

The dataset contains passenger-related information such as:

- Gender, Age Group, Occupation, Monthly Income
- Bus Usage Frequency, Bus Type Used, Trip Distance, Travel Purpose
- Wait Time, Service Frequency, Fare, Comfort, Crowding, Cleanliness, Safety, AC, Driver, Reliability

The target variable is `Bus_Preference` (SLTB bus / Private Bus).

**Known data-provenance limitation:** the 724-row modelling file (`data/`) was provided as an already-processed file. The exact transformation from the original 737-response raw survey export to this 724-row file could not be reconstructed — a row-matching check found only 13 of 724 rows line up directly with the raw export. This should be resolved (e.g. via form/sheet version history) before this dataset is used beyond a course exercise. See the accompanying paper, Section 3.1, for the full account.

## Machine Learning Models

Five models were implemented and evaluated: Decision Tree, Logistic Regression, Random Forest, XGBoost, and a Genetic-Algorithm-selected XGBoost (GA-XGBoost, 12 of 18 features).

## Model Performance

**Reported result (5-seed × 5-fold cross-validation, 25 runs per model — mean ± SD):**

| Model | Accuracy | F1 | AUC-ROC |
|---|---:|---:|---:|
| Decision Tree | 77.39% ± 3.98% | 79.44% ± 3.87% | 77.23% ± 4.00% |
| Logistic Regression | 82.14% ± 3.57% | 84.16% ± 3.21% | 89.28% ± 2.97% |
| **Random Forest** | **85.77% ± 2.66%** | **87.36% ± 2.52%** | **92.64% ± 1.92%** |
| XGBoost | 83.87% ± 3.30% | 85.53% ± 3.04% | 91.05% ± 2.33% |
| GA-XGBoost (12 features) | 82.88% ± 3.34% | 84.82% ± 3.04% | 90.12% ± 2.54% |

**Random Forest is the best-performing model under this protocol.** GA-based feature selection reduced the feature set by 33.3% (18→12) with no statistically significant change in performance vs. full-feature XGBoost (McNemar's test, p = 0.190).

> **Note on an earlier table.** A previous version of this README reported a single 70/30 train/test split (Decision Tree 78.44%, Logistic Regression 83.03%, Random Forest 83.94%, XGBoost 85.78%). Those numbers are real but came from one split only, which is a noisy, non-repeatable estimate. The 25-run cross-validated numbers above are the figures reported in the accompanying paper and should be treated as the project's actual result.

### Critical finding: `Bus_Type_Used` is a near-proxy for the target

`Bus_Type_Used` (the specific bus category a respondent reports using most) is the single most important feature in every model (Gini importance 0.276) and is collected on the same instrument as the target itself. Removing it drops accuracy by **6.3–7.9 percentage points** across all four non-GA models:

| Model | With `Bus_Type_Used` | Without | Δ |
|---|---:|---:|---:|
| Decision Tree | 77.39% | 70.16% | −7.23 |
| Logistic Regression | 82.14% | 75.85% | −6.29 |
| Random Forest | 85.77% | 77.87% | −7.90 |
| XGBoost | 83.87% | 76.40% | −7.47 |

This means a substantial share of the headline accuracy reflects this near-proxy relationship rather than genuine prediction from the nine independent service-quality ratings. See the paper's Discussion (Section 5.1) and Results (Section 4.3) for the full analysis.

## Working Model

A Streamlit application is included in `app.py`. It allows users to enter passenger information and receive a predicted bus preference and per-class probability.

## Reproducing the results

```bash
git clone https://github.com/fathiazn/sltb-private-bus-preference-ml.git
cd sltb-private-bus-preference-ml
pip install -r requirements.txt
```

Open the notebook in `notebooks/` in Google Colab or Jupyter and run top to bottom — it loads the data, performs preprocessing/encoding, trains all five models, runs the 25-run cross-validation, the GA feature selection, the `Bus_Type_Used` ablation, and the McNemar test.

To run the prediction app locally:
```bash
python -m streamlit run app.py
```

## Project Structure

```
SLTB-vs-Private-Bus-Preference/
├── architecture/
├── data/
├── models/
├── notebooks/
├── results/
├── src/
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

## License

See `LICENSE`. Survey data provided for academic/course use — see the data-provenance limitation above before any other use.
