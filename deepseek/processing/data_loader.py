import pandas as pd
import os
from typing import List, Dict

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_novels(self) -> Dict[str, str]:
        novels = {}
        for filename in os.listdir(self.config.NOVEL_DIR):
            if filename.endswith(".txt"):
                path = os.path.join(self.config.NOVEL_DIR, filename)
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    novels[filename] = f.read()
        return novels

    def load_train_data(self) -> pd.DataFrame:
        return pd.read_csv(self.config.TRAIN_CSV)

    def load_test_data(self) -> pd.DataFrame:
        return pd.read_csv(self.config.TEST_CSV)