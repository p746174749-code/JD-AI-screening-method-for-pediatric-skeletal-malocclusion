import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def calculate_metrics_original(y_true, y_pred, class_labels):
    """Calculates per-class and average metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    sensitivity = []
    specificity = []
    accuracy = []
    f1_scores = []

    for i in range(len(class_labels)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FP - FN

        # Sensitivity (Recall)
        sens = TP / (TP + FN) if (TP + FN) != 0 else 0
        sensitivity.append(sens)

        # Specificity
        spec = TN / (TN + FP) if (TN + FP) != 0 else 0
        specificity.append(spec)

        # Accuracy
        acc = (TP + TN) / cm.sum()
        accuracy.append(acc)

        # Precision
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0

        # F1 Score
        f1 = (2 * precision * sens) / (precision + sens) if (precision + sens) != 0 else 0
        f1_scores.append(f1)

    # Weighted F1
    w_f1 = f1_score(y_true, y_pred, average='weighted', labels=class_labels)

    # Averages (Macro)
    avg_sensitivity = np.mean(sensitivity)
    avg_specificity = np.mean(specificity)
    avg_accuracy = np.mean(accuracy)
    avg_f1 = np.mean(f1_scores)

    return sensitivity, specificity, accuracy, f1_scores, avg_sensitivity, avg_specificity, avg_accuracy, avg_f1, w_f1


def main(file_path, model_columns, output_name):
    """Main execution logic."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Load Data
    df = pd.read_excel(file_path)
    class_labels = [1, 2, 3]
    true_labels = df['GT']
    all_subject_data = {}

    # --- Loop through each model fold ---
    for sub in model_columns:
        if sub in df.columns:
            predicted_labels = df[sub]

            # Calculate metrics
            sens, spec, acc, f1s, avg_sens, avg_spec, avg_acc, avg_f1, w_f1 = calculate_metrics_original(
                true_labels, predicted_labels, class_labels
            )
            overall_acc = (predicted_labels == true_labels).mean()

            # Store in dictionary
            all_subject_data[sub] = {
                'Recall (C1)': sens[0],
                'Recall (C2)': sens[1],
                'Recall (C3)': sens[2],
                'Macro recall': avg_sens,
                'Specificity (C1)': spec[0],
                'Specificity (C2)': spec[1],
                'Specificity (C3)': spec[2],
                'Macro Specificity': avg_spec,
                'F1 (C1)': f1s[0],
                'F1 (C2)': f1s[1],
                'F1 (C3)': f1s[2],
                'Macro F1': avg_f1,
                'Weighted F1': w_f1,
                'Overall Accuracy': overall_acc
            }

    # Convert to DataFrame
    output_df = pd.DataFrame(all_subject_data)

    # --- Calculate Mean ± Std ---
    model_df = output_df[model_columns]
    avg_col = model_df.mean(axis=1)
    std_col = model_df.std(axis=1, ddof=1)

    # Save the numeric values for display, but format them for the CSV/Excel
    output_df['Mean_±_Std'] = (avg_col * 100).map('{:.2f}'.format) + " ± " + (std_col * 100).map('{:.2f}'.format)

    # --- Terminal Printing ---
    print("\n" + "=" * 50)
    print(f"{'Performance Metric Summary (Mean ± Std)':^50}")
    print("=" * 50)
    # Print each metric's Mean ± Std
    for metric, val in output_df['Mean_±_Std'].items():
        print(f"{metric:<20} : {val}")
    print("=" * 50)

    # Save to Excel
    output_df.to_excel(output_name)
    print(f"Full results saved to: {output_name}")


if __name__ == "__main__":
    # ====== SETTINGS ======
    INPUT_FILE = 'data-test-pred-results.xlsx'
    MODELS = ['F1', 'F2', 'F3', 'F4', 'F5']
    OUTPUT_FILE = 'Detailed_Performance_Table.xlsx'

    main(INPUT_FILE, MODELS, OUTPUT_FILE)