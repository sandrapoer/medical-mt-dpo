from dotenv import load_dotenv
import os

load_dotenv()
proc_path = os.getenv("DATA_PROCESSED_DIR")

def show_samples(split_name):
    with open(f"{proc_path}/{split_name}/{split_name}.en") as f_en:
        en = [next(f_en).strip() for _ in range(3)]
    with open(f"{proc_path}/{split_name}/{split_name}.fr") as f_fr:
        fr = [next(f_fr).strip() for _ in range(3)]
    
    print(f"=== {split_name.upper()} samples ===")
    for i, (e, f) in enumerate(zip(en, fr), 1):
        print(f"{i}. EN: {e[:80]}...")
        print(f"   FR: {f[:80]}...")
        print()

show_samples("train")
show_samples("val")
show_samples("test")
