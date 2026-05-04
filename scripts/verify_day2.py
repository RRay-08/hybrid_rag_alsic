import os
import sqlite3
import json

print("🔍 Verifying Day 2 outputs...\n")

# Check files exist
files_to_check = [
    "../data/processed/materials.db",
    "../data/processed/chunks.json",
    "../data/processed/extracted_facts.json"
]

for file in files_to_check:
    exists = os.path.exists(file)
    status = "✅" if exists else "❌"
    print(f"{status} {file}")

# Check SQLite tables
print("\n🗄️  SQLite database contents:")
conn = sqlite3.connect("../data/processed/materials.db")
cursor = conn.cursor()

tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
for table in tables:
    table_name = table[0]
    count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"   • {table_name}: {count} records")

conn.close()

# Check JSON files
print("\n📄 JSON file summaries:")
for json_file in ["../data/processed/chunks.json", "../data/processed/extracted_facts.json"]:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   • {os.path.basename(json_file)}: {len(data)} items")

print("\n✅ Verification complete! Ready for Day 3.")