import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import os

load_dotenv()
raw_path = os.getenv("DATA_RAW_DIR")

context = ET.iterparse(f"{raw_path}/ELRC_EMEA_en-fr.tmx", events=("end",))
examples = []

for event, elem in context:
    if elem.tag.endswith("tu"):
        tuvs = elem.findall(".//{*}tuv")
        en = tuvs[0].find(".//{*}seg").text
        fr = tuvs[1].find(".//{*}seg").text
        
        if en and fr and len(en.split()) > 5:
            examples.append((en, fr))
        
        elem.clear()
        
        if len(examples) >= 5:
            break

print("Sample sentences:")
for i, (en, fr) in enumerate(examples, 1):
    print(f"\n{i}. EN: {en}")
    print(f"   FR: {fr}")