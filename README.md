# Financial Stability Score Prediction

A machine learning project that predicts an individual's **Financial Stability Score** from financial, demographic, and behavioral data. The notebook builds and compares three regression models: **Linear Regression**, **Ridge Regression**, and **Lasso Regression**.

Developed by **Group 3 (IT3D.1)** as a regression course assignment.

## Dataset

`Financial Stability.csv` contains **1,000 records** with 27 input features and one numeric target, `Financial_Stability_Score` (ranging from about 1,246 to 2,505).

| Group | Features |
| ----- | -------- |
| Income and work | `Salary`, `Work_Experience`, `Hours_Worked_Per_Week`, `Job_Satisfaction`, `Commuting_Time`, `Vacation_Days_Used`, `Annual_Bonus` |
| Credit and debt | `Loan_Amount`, `Credit_Score`, `Previous_Loans`, `Loan_Default_History`, `Mortgage_Status` |
| Savings and investments | `Savings_Account_Balance`, `Investment_in_Stocks`, `Retirement_Fund`, `Health_Insurance`, `Monthly_Expenses` |
| Household | `Marital_Status`, `House_Ownership`, `Number_of_Children`, `Number_of_Dependents`, `Education_Level` |
| Lifestyle and behavior | `Exercise_Hours_Per_Week`, `Social_Activity_Score`, `Online_Shopping_Spend`, `Mobile_Usage_Hours`, `Internet_Usage_Hours` |
| **Target** | `Financial_Stability_Score` |

**Missing values:** 765 missing entries are spread across 16 numeric columns (roughly 40 to 56 per column).

**Categorical features:**

- `Education_Level`: High School, Associate Degree, Bachelor's, Master's, PhD
- `Marital_Status`: Single, Married, Divorced, Widowed
- `House_Ownership`: Owned, Rented, Mortgaged, Other
- `Mortgage_Status`: No Mortgage, Paying Mortgage, Fully Paid, Defaulted

## Methodology

1. **Data inspection:** columns, data types, and null counts
2. **Missing values:** numeric columns filled with the column mean, categorical columns filled with the mode
3. **Encoding:** one-hot encoding of `Education_Level`, `Marital_Status`, `House_Ownership`, and `Mortgage_Status` (first category dropped), giving 36 input features
4. **Feature and target split:** `Financial_Stability_Score` as the target
5. **Model training and evaluation**, using R² as the accuracy measure:
   - **Linear Regression:** test sizes of 0.20, 0.25, and 0.30, with scores averaged over 49 random states
   - **Ridge Regression:** test sizes of 0.20, 0.25, and 0.30, with alpha values of 0.1, 1, 10, and 100
   - **Lasso Regression:** test sizes of 0.20, 0.25, and 0.30, with alpha values of 0.01, 0.1, 1, 10, and 100, plus a count of how many features each alpha keeps
6. **Comparison and visualization:** a summary table of the three models and a plot of Lasso train and test R² against alpha

## Results

| Model | Best Test Size | Best Alpha | Train R² | Test R² |
| :---: | :---: | :---: | :---: | :---: |
| Linear Regression | 0.25 | N/A | 0.5587 | 0.5026 |
| Ridge Regression | 0.20 | 0.1 | 0.5444 | 0.5543 |
| Lasso Regression | 0.20 | 0.01 | 0.5444 | 0.5545 |

**Key findings**

- **Lasso Regression** had the highest test R² (0.5545), narrowly ahead of Ridge Regression (0.5543).
- Linear Regression had the lowest test R² (0.5026).
- At the best alpha (0.01), Lasso kept all 36 features. Larger alphas removed features (19 remained at alpha = 100) at a small cost in R².

## Repository Contents

```
.
├── Financial Stability.csv                                   Dataset (1,000 records)
└── Group 3 - Linear Regression Assignment IT3D.1.ipynb       Full analysis notebook
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/ycon4/Financial-Stability-Score-Prediction-Project.git
cd Financial-Stability-Score-Prediction-Project

pip install pandas numpy seaborn matplotlib scikit-learn jupyter
```

### Running the Notebook

```bash
jupyter notebook "Group 3 - Linear Regression Assignment IT3D.1.ipynb"
```

Run the cells from top to bottom. Keep `Financial Stability.csv` in the same folder as the notebook, since it is loaded with a relative path.

## Notes and Limitations

- The Ridge and Lasso results come from a single train-test split (`random_state=42`) with the best alpha chosen by test R², while the Linear Regression result is an average over 49 random splits. The models were therefore not compared under identical conditions, so the gap between them should be read with caution.
- Most numeric features are spread roughly evenly between 0 and 100 with no units given, which suggests the dataset is synthetic. Results may not carry over to real financial data.
- Moderate R² values (about 0.50 to 0.55) mean that roughly half of the variation in the score is not explained by these features.

## Authors

Group 3 (IT3D.1)

- Ryan S. Baguio
- Geff Kendra C. Gaviola
- Shane Daryl C. Maghinay
- Luzinda Niña M. Panong
- Andrei G. Raagas
- Kaye B. Villar
