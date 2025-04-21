from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, accuracy_score
import numpy as np

def precision_at_recall_threshold(y_true, y_scores, recall_target=0.90):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    try:
        idx = np.where(recall >= recall_target)[0][-1]
        return precision[idx]
    except IndexError:
        return 0.0

def evaluate_models(y_true, rf_preds, svm_preds, lstm_preds, ensemble_preds):
    # Defensive cleanup of label data
    y_true = y_true.fillna(0)
    y_true.replace([np.inf, -np.inf], 0, inplace=True)
    y_true = y_true.astype(int)

    y_binary = y_true

    def print_metrics(name, preds):
        preds_binary = (preds >= 0.5).astype(int)
        f1 = f1_score(y_binary, preds_binary)
        try:
            auc = roc_auc_score(y_binary, preds)
        except:
            auc = float('nan')
        prec90 = precision_at_recall_threshold(y_binary, preds, recall_target=0.90)
        acc = accuracy_score(y_binary, preds_binary)

        print(f"{name}: F1={f1:.3f}, AUC={auc:.3f}, Precision@90%Recall={prec90:.3f}, Accuracy={acc:.3f}")

    print("\n--- MODEL EVALUATION ---")
    print_metrics("RF", rf_preds)
    print_metrics("SVM", svm_preds)
    print_metrics("LSTM", lstm_preds)
    print_metrics("Ensemble", ensemble_preds)