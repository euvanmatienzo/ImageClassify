import json

# ------------------ CONFIG ------------------
ei_json_file = "training_labels.json"   # Edge Impulse JSON
mapping_file = "class_names.json"       # human-readable label mapping
output_file = "labels.txt"              # Output labels.txt
# --------------------------------------------

# Load Edge Impulse JSON
with open(ei_json_file) as f:
    data = json.load(f)

# Load human-readable mapping
with open(mapping_file) as f:
    mapping = json.load(f)

# Extract unique label IDs from bounding boxes
label_ids = set()
for sample in data.get("samples", []):
    for box in sample.get("boundingBoxes", []):
        label_ids.add(box["label"])

label_ids = sorted(label_ids)

labels = []
missing_labels = []

for i in label_ids:
    key = str(i)
    if key in mapping:
        labels.append(mapping[key])
    else:
        # Handle missing label safely
        unknown_name = f"UNKNOWN_{i}"
        labels.append(unknown_name)
        missing_labels.append(i)

# Write labels.txt
with open(output_file, "w") as f:
    for label in labels:
        f.write(label + "\n")

print(f"? labels.txt created: {output_file}")
if missing_labels:
    print("? WARNING: Missing mappings for label IDs:", missing_labels)
    print("Please add them to class_names.json for correct human-readable names.")

print("Labels in order:")
for idx, name in zip(label_ids, labels):
    print(f"{idx}: {name}")
