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
        print(f"Overall Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
        print(f"Consistent: {consistent_correct}/{consistent_total} = {consistent_correct/consistent_total*100:.2f}%")
        print(f"Contradict: {contradict_correct}/{contradict_total} = {contradict_correct/contradict_total*100:.2f}%")
    else:
        print("No matching IDs found.")

except Exception as e:
    print(f"Error: {e}")
