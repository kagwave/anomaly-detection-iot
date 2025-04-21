import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def explain_shap_tree(model, X, out_dir="outputs", prefix="shap"):
    os.makedirs(out_dir, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Save bar summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_summary_bar.png")
    plt.close()

    # Save detailed dot plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_summary_dot.png")
    plt.close()

    return shap_values