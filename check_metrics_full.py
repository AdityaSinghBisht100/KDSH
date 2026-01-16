
import csv
import math

def read_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def calculate_metrics():
    try:
        test_rows = read_csv('files/test.csv')
        pred_rows = read_csv('submission_bdh.csv')

        # Map id -> prediction
        preds = {str(row['id']): int(row['prediction']) for row in pred_rows}

        # Confusion Matrix components
        # Class 1: Consistent (Positive)
        # Class 0: Contradiction (Negative)
        tp = 0 # Predicted Consistent, Actual Consistent
        tn = 0 # Predicted Contradict, Actual Contradict
        fp = 0 # Predicted Consistent, Actual Contradict
        fn = 0 # Predicted Contradict, Actual Consistent
        
        y_true = []
        y_pred = []

        for row in test_rows:
            row_id = str(row['id'])
            if row_id in preds:
                label_val = row.get('label')
                if not label_val:
                    continue
                label_str = label_val.strip().lower()
                ground_truth = 1 if label_str == 'consistent' else 0
                prediction = preds[row_id]
                
                y_true.append(ground_truth)
                y_pred.append(prediction)
                
                if ground_truth == 1:
                    if prediction == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if prediction == 0:
                        tn += 1
                    else:
                        fp += 1
        
        total = tp + tn + fp + fn
        if total == 0:
            print("No matching records found.")
            return

        print("\n=== Confusion Matrix ===")
        print(f"{'':<20} {'Pred Contradict (0)':<20} {'Pred Consistent (1)':<20}")
        print(f"{'Actual Contradict':<20} {tn:<20} {fp:<20}")
        print(f"{'Actual Consistent':<20} {fn:<20} {tp:<20}")

        # Metrics
        print("\n=== Performance Metrics ===")
        accuracy = (tp + tn) / total
        
        # Consistent (1) Metrics
        prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_1 = 2 * (prec_1 * rec_1) / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0
        
        # Contradict (0) Metrics
        prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_0 = 2 * (prec_0 * rec_0) / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0

        print(f"Overall Accuracy: {accuracy:.4f} ({tp+tn}/{total})")
        print(f"\nClass: Consistent (1)")
        print(f"  Precision: {prec_1:.4f}")
        print(f"  Recall:    {rec_1:.4f}")
        print(f"  F1-Score:  {f1_1:.4f}")
        
        print(f"\nClass: Contradiction (0)")
        print(f"  Precision: {prec_0:.4f}")
        print(f"  Recall:    {rec_0:.4f}")
        print(f"  F1-Score:  {f1_0:.4f}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    calculate_metrics()
