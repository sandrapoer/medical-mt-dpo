import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Load en-fr TSV files
test_data = pd.read_csv(os.getenv("DATA_RAW_DIR") + "test.en-fr.tsv", sep="\t")
dev_data = pd.read_csv(os.getenv("DATA_RAW_DIR") + "dev.en-fr.tsv", sep="\t")


print("Number of test samples:", len(test_data))
print(f"\nColumn names: {list(test_data.columns)}")
print("Number of columns in test data:", len(test_data.columns))
print("Examples test data:\n", test_data.head())

print("Number of dev samples:", len(dev_data))
print(f"\nColumn names: {list(dev_data.columns)}")
print("Number of columns in dev data:", len(dev_data.columns))
print("Examples dev data:\n", dev_data.head())