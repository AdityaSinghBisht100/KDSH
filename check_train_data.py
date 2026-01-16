import csv

def read_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

try:
    # For training data, we need to generate predictions first
    # Let's check if we have training labels and can compare
    train_rows = read_csv('files/train.csv')
    
    print(f"Training data has {len(train_rows)} rows")
    
    # Check what columns we have
    if train_rows:
        print(f"Columns: {list(train_rows[0].keys())}")
        
        # Check label distribution
        labels = [row.get('label', '').strip().lower() for row in train_rows if row.get('label')]
        consistent_count = sum(1 for l in labels if l == 'consistent')
        contradict_count = sum(1 for l in labels if l == 'contradiction')
        
        print(f"\nTraining Label Distribution:")
        print(f"  Consistent: {consistent_count}")
        print(f"  Contradiction: {contradict_count}")
        print(f"  Total: {len(labels)}")

except Exception as e:
    print(f"Error: {e}")
