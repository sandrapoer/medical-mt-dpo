import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import os

load_dotenv()
raw_path = os.getenv("DATA_RAW_DIR")

# EMEA TMX file parsing (over 100mb -> use iterparse)
context = ET.iterparse(f"{raw_path}/EMEA_en-fr.tmx", events=("end",))
tu_count = 0
first_example = None

for event, elem in context:
    if elem.tag.endswith("tu"):
        tu_count += 1
        
        # Save first example
        if tu_count == 1:
            tuvs = elem.findall(".//{*}tuv")
            en = tuvs[0].find(".//{*}seg").text
            fr = tuvs[1].find(".//{*}seg").text
            first_example = (en, fr)
        
        # Clear memory
        elem.clear()
        
        if tu_count % 100000 == 0:
            print(f"Processed {tu_count:,} pairs...")

print(f"\nTotal EMEA pairs: {tu_count:,}")
