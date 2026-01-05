
# 🫧🍏 Machine Learning Foundations Portfolio ☃️🎐

---

# 📊 [Government Confidence and Corruption Perception ML Model](https://github.com/maggieyung/ecornell-portfolio/blob/main/DefineAndSolveMLProblem.ipynb)



## 📑 Overview

This was the final project submitted for Break Through Tech's Machine Learning Foundations course, where I aimed to apply all machine learning skills and techniques learned into building an end-to-end ML regression pipeline from a real-life, relevant dataset. In this project, my goal was to use the World Happiness Report (WHR) 2018 dataset to develop two predictive models: one predicting Confidence in National Government and another predicting Perceptions of Corruption. Both models predict continuous scores based on socioeconomic features like social support, life expectancy, freedom to make choices, and life satisfaction. To build these models, I performed EDA, data preprocessing, feature engineering, model training, and validation - making use of Python ML libraries like Pandas, NumPy, Scikit-learn, Seaborn, and Matplotlib to train models and perform data visualizations.


- **Dataset:** World Happiness Report 2018 (1,400+ country-year records, 156 countries)
- **Problem Type:** Regression (Continuous value prediction)
- **Evaluation Framework:** Multi-metric evaluation (MAE, RMSE, R², Cross-validation)

**Contributors:** Maggie Yung ([@maggieyung](https://github.com/maggieyung)), Jose Cruz ([@Jose-Gael-Cruz-Lopez](https://github.com/Jose-Gael-Cruz-Lopez))



## 🌐 Relevance

This supervised ML regression problem holds real-world relevance because predicting government trust and corruption perceptions helps identify brewing social crises. The insights gained from these models could provide significant value to multiple stakeholders: NGOs could target interventions in regions where confidence is dropping, governments could monitor policy impacts in real-time and benchmark countries against global trends, and international organizations could assess region political stability and plan aid/support accordingly.


## 🔧 Toolkit
- **Languages:** Python
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (ensemble models, GridSearchCV)
- **Statistical Analysis:** SciPy (Winsorization)
- **Cross-Validation:** 5-fold CV for model selection
- **Development:** Jupyter Notebook

---

## ⚒️ Model Workflow

**Exploratory Data Analysis (EDA)**
- Inspected the dataset to locate outliers, missing values, and data distribution
  - Analyzed feature correlations with target variables
  - Visualized key relationships between socioeconomic factors and government trust/corruption

**Data Preprocessing**
- Handled missing values through median/mean imputation strategies
- Applied Winsorization to reduce the influence of extreme outliers
- Scaled numerical features (MinMaxScaler for Model 1, StandardScaler for Model 2)

**Feature Engineering**
- Created custom "Trust Factor" feature combining freedom and corruption scores for Model 1
- Removed irrelevant features with missing data or weak correlations

**Model Training**
- Model 1: Trained Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting
  - Used 70% training / 15% validation / 15% test split
- Model 2: Trained Linear Regression, Decision Tree, Stacking, Gradient Boosting, and Random Forest
  - Used 75% training / 25% test split with GridSearchCV hyperparameter tuning

**Model Validation**
- Performed 5-fold cross-validation on training data
- Compared validation performance across all model candidates

**Model Testing**
- Evaluated all models on test data, analyzing residuals and error distributions

**Model Evaluation**
- Model 1: Evaluated using MAE (Mean Absolute Error) and R² (coefficient of determination)
- Model 2: Evaluated using RMSE (Root Mean Square Error), R², and MAE
- Computed cross-validation metrics to assess generalization

**Model Selection**
- Model 1: Selected Random Forest based on lowest validation MAE
- Model 2: Selected Decision Tree based on best metrics across RMSE, R², and MAE


## 📊 Analysis

**Model 1 (Government Confidence) Findings:**
- Random Forest (non-linear relationships) performed the best
- Cross-validation MAE averaged 0.0759
- Top predictive features: Trust Factor (30.2%), Perceptions of Corruption (18.8%), Healthy Life Expectancy (17.2%)

**Model 2 (Corruption Perception) Findings:**
- Decision Tree outperformed all other approaches with lowest RMSE and MAE
- 65.7% improvement in RMSE compared to baseline Linear Regression
- Top predictive features: Social Support, Life Ladder, and Healthy Life Expectancy showed strong correlations


Across both models, we found consistent patterns regarding which features most influence predictions:
- **Perceptions of Corruption** and **Freedom to Make Choices** are inversely related to government confidence
- **Healthy Life Expectancy** is a reliable proxy for overall government effectiveness
- **Life Ladder** (overall life satisfaction) reflects satisfaction with government performance

### Model Performance Comparison

**Model 1: Government Confidence Prediction**

| Model | MAE | R² |
|-------|-----|-----|
| Linear Regression | 0.0578 | -0.5581 |
| Ridge Regression | 0.0468 | -1.0739 |
| **Random Forest** | **0.0438** | **-0.3474** |
| Gradient Boosting | 0.0539 | -0.3223 |

**Model 2: Corruption Perception Prediction**

| Model | RMSE | R² | MAE |
|-------|------|-----|-----|
| Linear Regression | 0.1060 | -9.2219 | 0.0757 |
| **Decision Tree** | **0.0364** | **-0.2060** | **0.0335** |
| Stacking | 0.1003 | -8.1449 | 0.0713 |
| Gradient Boosting | 0.0679 | -3.1885 | 0.0548 |
| Random Forest | 0.0383 | -0.3363 | 0.0353 |


## 📈 Key Metrics

**Model 1 Performance:**
- Linear baseline: MAE 0.0578
- Ridge regularization: MAE 0.0468 (modest improvement)
- Random Forest: MAE 0.0438 (best - 24% better than baseline)
- Gradient Boosting: MAE 0.0539

**Model 2 Performance:**
- Linear baseline: RMSE 0.1060
- Decision Tree: RMSE 0.0364 (best - 66% improvement)
- Random Forest: RMSE 0.0383 (very close second)
- Gradient Boosting: RMSE 0.0679

---

## 🎯 Main Takeaways

Both models achieve mean absolute errors small enough to be useful for identifying concerning trends in government trust/corruption perception (Model 1 MAE: 0.1663; Model 2 MAE: 0.0335). Random Forest (Model 1) and Decision Tree (Model 2) outperformed linear models, indicating non-linear relationships between socioeconomic factors and institutional trust/corruption perception. Tree-based methods with explicit hyperparameter tuning yielded the best outcomes. 5-fold cross-validation revealed consistent model performance across different data splits (Model 1 CV MAE: 0.0759 ± 0.0222; Model 2 CV RMSE: 0.0569 ± 0.0134),  suggesting models generalize reasonably well to unseen data.


### Best Performing Models

**Model 1: Random Forest for Government Confidence**
- Test MAE: 0.1663
- Test R²: -11.3260
- 5-Fold CV MAE Mean: 0.0759 (Std: 0.0222)
- Reflects variance in government trust across countries

**Model 2: Decision Tree for Corruption Perception**
- Test RMSE: 0.0364
- Test R²: -0.2060
- Test MAE: 0.0335
- 65.7% improvement in RMSE versus Linear Regression baseline
- Best performance at predicting corruption perception scores
