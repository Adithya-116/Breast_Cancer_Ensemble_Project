
# Breast Cancer Ensemble Final Code

## Description
This Python script implements an ensemble-based breast cancer classification approach using the Wisconsin Breast Cancer Dataset (WBCD).
The workflow includes:
- Data preprocessing and scaling
- Feature selection using 16 stepwise LDA features
- Training base models: Logistic Regression, Random Forest, and XGBoost
- Building soft VotingClassifier and StackingClassifier
- Comparing both ensemble models and selecting the best one automatically
- Generating high-resolution confusion matrix and ROC curve plots

## Usage
1. Place `breast_cancer_dataset.csv` in the same folder.
2. Run `Breast_Cancer_Ensemble_Final_Code.py` in your Python environment.
3. Outputs:
   - Prints classification report and accuracy of each model
   - Saves `Confusion_Matrix_Final_HighRes.png`
   - Saves `ROC_Curve_Final_HighRes.png`

## Dependencies
- pandas
- scikit-learn
- xgboost
- matplotlib
- seaborn

## Author
Machiraju Adithya Vaibhav and Team
