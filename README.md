
# Breast Cancer Classification Using Ensemble Learning

This project presents a machine learning framework for early and accurate classification of breast cancer using classical and ensemble-based models. It applies robust preprocessing, feature selection using stepwise Linear Discriminant Analysis (LDA), and ensemble strategies (Voting and Stacking) to achieve high accuracy using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

---

## 📑 Project Overview

- **Dataset**: Wisconsin Diagnostic Breast Cancer (WDBC)
- **Goal**: Classify tumors as malignant or benign
- **Techniques Used**:
  - Stepwise LDA for feature selection
  - Z-score normalization
  - Stratified sampling
  - Ensemble methods: Voting & Stacking Classifiers
  - Hyperparameter tuning with GridSearchCV
- **Best Accuracy**: 98.25% using Stacking Classifier

---

## 📊 Dataset Description

- 569 records, each with 30 numerical features
- Target labels:  
  - `M` → Malignant (mapped to `1`)  
  - `B` → Benign (mapped to `0`)  
- Dataset is clean (no missing values)
- Features standardized using Z-score normalization

---

## 💻 How to Run the Project

Follow these steps to set up and run the project:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/breast-cancer-classification
cd breast-cancer-classification
```

### 2. Create and Activate Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

- Download the WDBC dataset from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- Save it as `breast_cancer_dataset.csv` in the `data/` folder

### 5. Run the Pipeline

```bash
python src/preprocessing.py          # Clean and normalize data
python src/train_models.py           # Train base models
python src/ensemble_models.py        # Train Voting and Stacking classifiers
python src/evaluation.py             # View evaluation metrics and plots
```

### 6. Make Predictions

```bash
python src/predict.py --input sample_input.csv
```

---

## 📈 Results

| Model             | Accuracy (%) | Precision | Recall | F1-Score | AUC-ROC |
|------------------|--------------|-----------|--------|----------|---------|
| Logistic Regression | 98.25     | 0.98      | 0.98   | 0.98     | 0.99    |
| Random Forest       | 97.00     | 0.97      | 0.97   | 0.97     | 0.98    |
| XGBoost             | 98.25     | 0.98      | 0.98   | 0.98     | 0.99    |
| LightGBM            | 98.25     | 0.98      | 0.98   | 0.98     | 0.99    |
| Voting Classifier   | 96.65     | 0.97      | 0.96   | 0.96     | 0.98    |
| Stacking Classifier | 98.25     | 0.98      | 0.98   | 0.98     | 0.99    |

---

## 📦 Project Structure

```
breast-cancer-classification/
├── data/
│   └── data.csv
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation_metrics.py
│   └── model_interpretation.py
├── results/
│   ├── classification_report.csv
│   ├── Confusion_Matrix.png
│   ├── Feature_vs_Accuracy.png
│   └── ROC_Curve.png
├── tasks/
│   ├── classification/
│       └── run_classification_main.py
├── requirements.txt
└── README.md
```

---

## 🔮 Future Improvements

- Test on external and imbalanced datasets  
- Add support for SHAP or LIME for interpretability  
- Use other feature selection techniques (e.g., RFE, mutual info)  
- Deploy using Flask or Streamlit  
- Explore neural network baselines

---

## 📚 References

- [1] WHO, “Global Cancer Observatory, 2022,” [Online](https://gco.iarc.fr/)
- [2] Biswas et al., “Multi-Model Machine Learning Approach,” *ICAC3*, 2024
- [3] Bardou et al., “Breast Cancer Classification Using CNNs,” *IEEE Access*, 2018
- [4] Reshan et al., “Ensemble Learning Techniques,” *Life*, 2023
- [5] Khan et al., “Improved ML Framework for Breast Cancer,” *Procedia CS*, 2023
- [6] Mishra et al., “Stepwise LDA & SVM for Breast Cancer,” *Measurement: Sensors*, 2023
- [7] Rahman et al., “Stratified Sampling in Medical ML,” *Informatics in Medicine Unlocked*, 2024
- [8] Jena & Sahoo, “Hybrid Feature & Ensemble Learning,” *Materials Today: Proceedings*, 2023
- [9] Jain & Arora, “ML in Breast Cancer Diagnosis: A Review,” *BSPC*, 2023

---

## 👩‍💻 Author

**Jitisha Khede**  
Medicaps University 

**Machiraju Adithya Vaibhav**  
Andra University 

**Majji Harsha Vardhan**  
Andra University 

---

## ⚖️ License

This project is intended for academic and research use only. Please cite the work if used.
