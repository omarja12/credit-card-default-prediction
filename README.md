# Credit Card Default Prediction

Which credit-card clients will default next month? A classification study on 30,000 clients with imbalanced classes (22% default), comparing plain classifiers with imbalance-aware ensembles on recall, F1 and ROC AUC.

The full analysis, with outputs, is in [`credit_card_default.ipynb`](credit_card_default.ipynb).

## Results on the held-out test set (20%, stratified)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Balanced random forest | 0.776 | 0.498 | **0.606** | **0.546** | **0.782** |
| Under-sampling + random forest | 0.763 | 0.476 | 0.624 | 0.540 | 0.780 |
| Under-sampling + gradient boosting | 0.756 | 0.465 | 0.634 | 0.537 | 0.782 |
| Balanced bagging | 0.780 | 0.507 | 0.551 | 0.528 | 0.760 |
| Random forest | **0.820** | 0.676 | 0.371 | 0.479 | 0.778 |
| Logistic regression | 0.807 | 0.674 | 0.263 | 0.378 | 0.749 |

The most accurate model misses most defaulters. Re-balancing the training data raises recall from 37% to about 61% at similar ROC AUC, trading precision for it, so the operating threshold should follow the cost of a missed default versus a false alarm.

## Method

1. **Cleaning:** undefined EDUCATION (0, 5, 6) and MARRIAGE (0) codes are dropped; repayment status -2 ("no consumption") is a valid state and kept.
2. **Features:** the six highly correlated monthly bill amounts are averaged into one; payment amounts enter as `log1p` to tame skew (skew 11-31 down to about -1).
3. **Split:** 80/20, stratified on the target. One-hot encoding is fitted on train only.
4. **Models:** logistic regression (standardised), random forest, balanced random forest, balanced bagging, and random under-sampling feeding a random forest or gradient boosting (`imbalanced-learn`).
5. **Metrics:** every scikit-learn metric is called as `metric(y_true, y_pred)`; ROC AUC uses predicted probabilities.

## Run it

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute credit_card_default.ipynb --output executed.ipynb
```

The GitHub Actions workflow executes the notebook top to bottom on every push; its last cell asserts that every model ran and that the best ROC AUC exceeds 0.7.

## Data

*Default of Credit Card Clients*, [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), CC BY 4.0. Yeh, I. C. and Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473-2480.
