# Diabetes Prediction: KNN vs Logistic Regression

Predicting diabetes from patient health data (100k rows, ~8.5% positive) using scikit-learn pipelines, with a focus on handling class imbalance.

## Dataset
Kaggle diabetes prediction dataset. Features: `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, `blood_glucose_level`. Target: `diabetes` (0/1).

## Approach
- `ColumnTransformer` inside a `Pipeline`: `StandardScaler` for numeric columns, `OneHotEncoder` for `gender` and `smoking_history`. Preprocessing is fit only on training folds, which prevents data leakage.
- `GridSearchCV` (3-fold) scored on F1, recall and ROC AUC, since accuracy is misleading on imbalanced data.
- Stratified train/test split so both sets keep the class ratio.

## Results (3-fold CV on the training split)

| Model | F1 | Recall | ROC AUC |
|---|---|---|---|
| KNN (K=15) | 0.716 | 0.573 | 0.945 |
| Logistic regression, class weight 1:2 | 0.725 | 0.701 | 0.962 |
| Logistic regression, class weight 1:5 | 0.659 | 0.801 | 0.962 |
| Logistic regression, balanced | 0.571 | 0.881 | 0.962 |

## Key findings
- KNN reached 96% accuracy, but recall on diabetics was only 57%. Accuracy was inflated by the majority class.
- Logistic regression beat KNN on every metric and trains in about 1 second versus 90+ seconds for KNN scoring.
- Class weights moved the decision threshold, not the model's ranking ability: ROC AUC stayed at 0.962 while recall rose from 0.70 to 0.88 and precision fell.
- The right weight depends on cost. For screening, missing a diabetic is usually worse than a false alarm, so a higher weight is justified.

## Next steps
- Report final metrics on the held-out test set.
- Compare against `HistGradientBoostingClassifier` and tune the decision threshold.

## Run it
```bash
pip install pandas scikit-learn matplotlib
```
Then run the notebook.
