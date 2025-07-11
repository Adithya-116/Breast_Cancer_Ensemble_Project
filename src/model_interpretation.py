import shap
import matplotlib.pyplot as plt

def plot_shap_summary(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, show=False)
    plt.tight_layout()
    plt.savefig("results/shap_summary.png")
    plt.close()
