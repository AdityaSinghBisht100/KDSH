
import pandas as pd
import collections
import random

def diagnose():
    try:
        train_df = pd.read_csv("files/train.csv")
        test_df = pd.read_csv("files/test_custom.csv")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print("=== 1. Label Distribution Audit ===")
    print("Train Labels:", collections.Counter(train_df["label"]))
    print("Test Labels:", collections.Counter(test_df["label"]))
    
    train_counts = train_df["label"].value_counts()
    if "consistent" in train_counts:
        print(f"Train Consistent %: {train_counts['consistent'] / len(train_df) * 100:.1f}%")

    print("\n=== 2. Train-Test Leakage Check ===")
    train_pairs = set(zip(train_df["book_name"], train_df["char"]))
    test_pairs = set(zip(test_df["book_name"], test_df["char"]))
    overlap = train_pairs.intersection(test_pairs)
    print(f"Overlap (Book, Char) pairs: {len(overlap)}")
    if overlap:
        print("Leakage detected! Overlapping pairs:", overlap)

    print("\n=== 3. Context Validity / Sample Inspection ===")
    print("Random Contradiction Samples:")
    contradicts = train_df[train_df["label"] == "contradict"]
    if not contradicts.empty:
        for _, row in contradicts.sample(min(5, len(contradicts))).iterrows():
            print(f"- [{row['book_name']}] {row['char']}: {row['content']}")
            
    print("\nRandom Consistent Samples:")
    consistents = train_df[train_df["label"] == "consistent"]
    if not consistents.empty:
        for _, row in consistents.sample(min(5, len(consistents))).iterrows():
            print(f"- [{row['book_name']}] {row['char']}: {row['content']}")

if __name__ == "__main__":
    diagnose()
