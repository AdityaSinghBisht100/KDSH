import csv

def read_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

try:
    test_rows = read_csv('files/test.csv')
    pred_rows = read_csv('submission_modal.csv')

    # Create map of id -> prediction
    preds = {row['id']: int(row['prediction']) for row in pred_rows}

    correct = 0
    total = 0
    
    for row in test_rows:
        row_id = row['id']
        if row_id in preds:
            total += 1
            # Ground truth
            label_val = row.get('label')
            if not label_val:
                continue
            label_str = label_val.strip().lower()
            ground_truth = 1 if label_str == 'consistent' else 0
            
            if preds[row_id] == ground_truth:
                correct += 1

    if total > 0:
        print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
    else:
        print("No matching IDs found.")

except Exception as e:
    print(f"Error: {e}")
