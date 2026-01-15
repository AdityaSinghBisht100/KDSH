import pandas as pd

test = pd.read_csv('files/test.csv')
pred = pd.read_csv('submission_modal.csv')

merged = test.merge(pred, on='id')

def to_binary(label):
    if isinstance(label, str):
        return 1 if label.strip().lower() == 'consistent' else 0
    return int(label)

merged['ground_truth'] = merged['label'].apply(to_binary)
correct = (merged['ground_truth'] == merged['prediction']).sum()
total = len(merged)

print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
print(f"\nPrediction distribution:")
print(pred['prediction'].value_counts())
print(f"\nConfusion Matrix:")
print(pd.crosstab(merged['ground_truth'], merged['prediction'], rownames=['Actual'], colnames=['Predicted']))
