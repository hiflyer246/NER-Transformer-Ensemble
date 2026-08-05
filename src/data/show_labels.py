from src.config.labels import ID2LABEL

print("=" * 50)
print("NER LABELS")
print("=" * 50)

for idx, label in ID2LABEL.items():
    print(f"{idx:<3} : {label}")