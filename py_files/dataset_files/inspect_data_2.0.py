import xml.etree.ElementTree as ET

# Load the TMX file
def parse_tmx(tmx_path):
    """Extract all translation pairs from TMX file"""
    tree = ET.parse(tmx_path)
    root = tree.getroot()
    
    pairs = []
    for tu in root.findall('.//tu'):
        tuvs = tu.findall('tuv')
        if len(tuvs) >= 2:
            en_text = tuvs[0].find('seg').text
            fr_text = tuvs[1].find('seg').text
            if en_text and fr_text:  # Make sure neither is None
                pairs.append((en_text.strip(), fr_text.strip()))
    
    return pairs

# Load TSV files
def load_tsv_pairs(tsv_path):
    """Load translation pairs from TSV file"""
    pairs = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                source_lang = parts[0]
                target_lang = parts[1]
                source_text = parts[2].strip()
                target_text = parts[3].strip()
                
                # Only include en-fr pairs
                if source_lang == 'en' and target_lang == 'fr':
                    pairs.append((source_text, target_text))
    
    return pairs

# Main comparison
tmx_pairs = parse_tmx('/home/c2410843006/master-thesis/data/raw/tico19_en_fr.tmx')
dev_pairs = load_tsv_pairs('/home/c2410843006/master-thesis/data/raw/dev.en-fr.tsv')
test_pairs = load_tsv_pairs('/home/c2410843006/master-thesis/data/raw/test.en-fr.tsv')

print(f"TMX pairs: {len(tmx_pairs)}")
print(f"Dev pairs: {len(dev_pairs)}")
print(f"Test pairs: {len(test_pairs)}")
print(f"Dev + Test: {len(dev_pairs) + len(test_pairs)}")

# Check if dev is subset of TMX
tmx_set = set(tmx_pairs)
dev_in_tmx = sum(1 for pair in dev_pairs if pair in tmx_set)
test_in_tmx = sum(1 for pair in test_pairs if pair in tmx_set)

print(f"\nDev pairs found in TMX: {dev_in_tmx}/{len(dev_pairs)} ({dev_in_tmx/len(dev_pairs)*100:.1f}%)")
print(f"Test pairs found in TMX: {test_in_tmx}/{len(test_pairs)} ({test_in_tmx/len(test_pairs)*100:.1f}%)")

# Check if TMX = Dev + Test
combined_set = set(dev_pairs + test_pairs)
print(f"\nUnique pairs in Dev+Test: {len(combined_set)}")
print(f"TMX equals Dev+Test exactly: {tmx_set == combined_set}")

# Show some examples if they don't match
if tmx_set != combined_set:
    extra_in_tmx = tmx_set - combined_set
    extra_in_combined = combined_set - tmx_set
    print(f"\nExtra pairs in TMX (not in dev/test): {len(extra_in_tmx)}")
    print(f"Extra pairs in dev/test (not in TMX): {len(extra_in_combined)}")
    
    if len(extra_in_tmx) > 0:
        print("\nFirst 3 examples only in TMX:")
        for i, pair in enumerate(list(extra_in_tmx)[:3]):
            print(f"{i+1}. EN: {pair[0][:80]}...")
            print(f"   FR: {pair[1][:80]}...")