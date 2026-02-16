from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
raw_path = os.getenv("DATA_RAW_DIR")
proc_path = os.getenv("DATA_PROCESSED_DIR")

os.makedirs(f"{proc_path}/test", exist_ok=True)

# Add TICO-19 test
test_df = pd.read_csv(f"{raw_path}/test.en-fr.tsv", sep="\t")
with open(f"{proc_path}/test/test.en", "w") as f:
    f.write("\n".join(test_df["sourceString"].tolist()))
with open(f"{proc_path}/test/test.fr", "w") as f:
    f.write("\n".join(test_df["targetString"].tolist()))

print(f"Test (TICO-19): {len(test_df):,}")