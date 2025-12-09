# Credit Card Fraud Detection

Machine learning system to detect fraudulent credit card transactions with optimal balance between fraud detection and false alarms.

## Problem
Banks process millions of transactions daily and need automated fraud detection that catches fraudulent purchases while minimizing false alarms that frustrate customers. The dataset is severely imbalanced with only 0.17% fraud cases.

## Dataset
- 284,807 credit card transactions
- 492 fraud cases (0.17%)
- 30 features: V1-V28 (PCA anonymized), Time, Amount
- 70/30 train-test split with stratification

## Models Compared
1. Logistic Regression (baseline with top 5, top 10, and interaction features)
2. Ridge L2, Lasso L1, Elastic Net (regularized models)
3. Linear SVC L2 and L1 (support vector machines)
4. Random Forest (ensemble method)

## Results

**Best Model: Random Forest**
- F1-Score: 0.71
- Precision: 67%
- Recall: 76%
- ROC-AUC: 0.965
- False Alarms: 55 per 85,000 transactions
- Missed Fraud: 36 cases

**Comparison:**
- Logistic models: High precision (76-85%) but low recall (56-60%)
- Regularized models: High recall (86-89%) but extremely low precision (3-12%)
- Linear SVC: Moderate performance with high false alarm rates
- Random Forest: Best overall balance for real-world deployment

## Key Features
Feature importance analysis:
- V14: 39% (most critical)
- V10: 26%
- V12: 16%
- V4: 14%
- V11: 6%

## Business Application
Risk-based transaction routing:
- High Risk (>70%): Block immediately, alert customer
- Medium Risk (30-70%): Flag for review, require verification
- Low Risk (<30%): Approve automatically

Impact: Catches 76% of fraud with minimal customer disruption.

## Technical Details
- Cross-validation with StratifiedKFold
- GridSearchCV for hyperparameter tuning
- StandardScaler for feature normalization
- Metrics: F1-score, precision, recall, ROC-AUC, confusion matrix

## Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels
```

## Usage
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                               class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
```

## Key Findings
- Random Forest outperforms linear models for imbalanced fraud detection
- V14 dominates fraud prediction with 39% feature importance
- Model struggles with sophisticated fraud mimicking normal patterns
- Successfully detected fraud shows extreme feature values (V14 < -8)

## Future Work
- Implement SMOTE for improved class balance
- Add temporal features for pattern detection
- Deploy as real-time API
- Regular retraining for evolving fraud tactics

## Technologies
Python, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, statsmodels

## License
MIT

## Data Source
SecurePay Credit Card Fraud Detection Data (Kaggle - ESHUM_MALIK)
