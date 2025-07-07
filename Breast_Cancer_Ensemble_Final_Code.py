
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("breast_cancer_dataset.csv")
df.drop(columns=['id', 'Unnamed: 32'], inplace=True, errors='ignore')
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Selected features
selected_features = [
    'concave points_worst', 'radius_worst', 'texture_worst',
    'concave points_mean', 'compactness_mean', 'area_worst',
    'perimeter_worst', 'concavity_mean', 'symmetry_worst',
    'fractal_dimension_mean', 'compactness_se', 'radius_se',
    'texture_se', 'area_se', 'smoothness_worst', 'symmetry_mean'
]
X = df[selected_features]
y = df['diagnosis']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
lr = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Voting classifier
voting_clf = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
    voting='soft'
)
voting_clf.fit(X_train_scaled, y_train)
voting_pred = voting_clf.predict(X_test_scaled)
voting_acc = accuracy_score(y_test, voting_pred)

# Stacking classifier
stacking_clf = StackingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
    final_estimator=LogisticRegression(),
    cv=5
)
stacking_clf.fit(X_train_scaled, y_train)
stacking_pred = stacking_clf.predict(X_test_scaled)
stacking_acc = accuracy_score(y_test, stacking_pred)

# Compare
if stacking_acc > voting_acc:
    best_model_name = "StackingClassifier"
    best_model = stacking_clf
    best_acc = stacking_acc
    best_pred = stacking_pred
else:
    best_model_name = "VotingClassifier"
    best_model = voting_clf
    best_acc = voting_acc
    best_pred = voting_pred

# Print results
print(f"\n🔸 VotingClassifier Accuracy : {voting_acc:.4f}")
print(f"🔹 StackingClassifier Accuracy: {stacking_acc:.4f}")
print(f"\n✅ Best Model: {best_model_name} with accuracy {best_acc:.4f}")

# Additional metrics
print("\nClassification Report:\n", classification_report(y_test, best_pred))

# Confusion matrix
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(8, 7), dpi=300)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("Confusion_Matrix_Final_HighRes.png")
plt.close()

# ROC curve
best_proba = best_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, best_proba)
auc = roc_auc_score(y_test, best_proba)
plt.figure(figsize=(8, 7), dpi=300)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='darkorange')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {best_model_name}")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("ROC_Curve_Final_HighRes.png")
plt.close()
