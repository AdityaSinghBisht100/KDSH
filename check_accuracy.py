import csv

def read_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

try:
    test_rows = read_csv('files/test.csv')
    pred_rows = read_csv('submission_bdh.csv')

    # Create map of id -> prediction
    preds = {str(row['id']): int(row['prediction']) for row in pred_rows}

    correct = 0
    total = 0
    consistent_correct = 0
    consistent_total = 0
    contradict_correct = 0
    contradict_total = 0
    
    for row in test_rows:
        row_id = str(row['id'])
        if row_id in preds:
            total += 1
            label_val = row.get('label')
            if not label_val:
                continue
            label_str = label_val.strip().lower()
            ground_truth = 1 if label_str == 'consistent' else 0
            
            if ground_truth == 1:
                consistent_total += 1
                if preds[row_id] == 1:
                    consistent_correct += 1
            else:
                contradict_total += 1
                if preds[row_id] == 0:
                    contradict_correct += 1
            
            if preds[row_id] == ground_truth:
                correct += 1

    if total > 0:
        print(f"\n=== Evaluation Metrics ===")
        print(f"Overall Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
        
        # Confusion Matrix
        tp = consistent_correct
        tn = contradict_correct
        fp = contradict_total - contradict_correct
        fn = consistent_total - consistent_correct
        
        print("\n=== Confusion Matrix ===")
        print(f"{'':<12} {'Pred Cons':<10} {'Pred Cont':<10}")
        print(f"{'True Cons':<12} {tp:<10} {fn:<10}")
        print(f"{'True Cont':<12} {fp:<10} {tn:<10}")
        
        # Consistent Class Metrics (Label 1)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        print("\n=== Detailed Metrics ===")
        print(f"Consistent (1): Precision={prec:.2%}, Recall={rec:.2%}, F1={f1:.4f}")
        
        # Contradict Class Metrics (Label 0)
        # Invert logic: TP_c = TN, FP_c = FN, etc.
        prec_c = tn / (tn + fn) if (tn + fn) > 0 else 0
        rec_c = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0
        
        print(f"Contradict (0): Precision={prec_c:.2%}, Recall={rec_c:.2%}, F1={f1_c:.4f}")
        
        # Macro F1
        macro_f1 = (f1 + f1_c) / 2
        print(f"\nMacro F1 Score: {macro_f1:.4f}")
    else:
        print("No matching IDs found.")

except Exception as e:
    print(f"Error: {e}")
