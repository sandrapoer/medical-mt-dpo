import xml.etree.ElementTree as ET

tree = ET.parse("data/raw/tico19_en_fr.tmx")
root = tree.getroot()

# Count translation units
translation_units = len(root.findall(".//{*}tu"))
print("Number of translation units in TMX file:", translation_units)