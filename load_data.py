# load_data.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path="data/full_dataset.csv", language=None):
    """
    Loads the full dataset and splits it 80/10/10.
    Optionally filter by language: "zul", "xho", or "nso"
    """
    df = pd.read_csv(path)

    # Optional language filter
    if language:
        df = df[df["language"] == language]

    # 80% train, 10% dev, 10% test
    train, temp = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    dev, test = train_test_split(
        temp, test_size=0.5, stratify=temp["label"], random_state=42
    )

    print(f"Language: {language or 'all'}")
    print(f"Train: {len(train)} | Dev: {len(dev)} | Test: {len(test)}")

    return train, dev, test


if __name__ == "__main__":
    # Test it works — run this file directly to check
    train, dev, test = load_dataset()