from src.data_preprocessing import load_data
from src.model_training import get_models, train_models
from src.evaluation_metrics import evaluate_model

X_train, X_test, y_train, y_test = load_data("data/breast_cancer_dataset.csv")
models = get_models()
name, model, y_pred, acc, y_proba = train_models(models, X_train, y_train, X_test, y_test)
evaluate_model(name, y_test, y_pred, y_proba)
