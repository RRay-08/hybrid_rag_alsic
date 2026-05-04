import pandas as pd
import sqlite3
import os
import csv

# Configuration
BASE_DIR = r"E:\hybrid_rag"
CSV_FOLDER = os.path.join(BASE_DIR, "data", "structured")
DB_PATH = os.path.join(BASE_DIR, "data", "processed", "materials.db")

# Expected columns (adjust if your files differ)
EXPECTED_COLS = ["material", "temperature_c", "property_name", "property_value", "unit", "source"]

def detect_delimiter(file_path):
    """Auto-detect if CSV uses comma or tab delimiter"""
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        # Count separators in first line
        commas = first_line.count(',')
        tabs = first_line.count('\t')
        return '\t' if tabs > commas else ','

def load_and_validate_csv(file_path):
    """Load CSV with correct delimiter and validate columns"""
    delimiter = detect_delimiter(file_path)
    print(f"   📄 Detected delimiter: '{repr(delimiter)}'")
    
    df = pd.read_csv(file_path, delimiter=delimiter, engine='python')
    
    # Clean column names (strip whitespace, lowercase)
    df.columns = df.columns.str.strip().str.lower()
    
    # Check if expected columns exist
    missing = set(EXPECTED_COLS) - set(df.columns)
    if missing:
        print(f"   ⚠️ Warning: Missing columns {missing} in {os.path.basename(file_path)}")
        print(f"   📋 Found columns: {list(df.columns)}")
        # Try to map common variants
        col_map = {
            'material': ['material', 'alloy', 'composite'],
            'temperature_c': ['temperature_c', 'temp_c', 'temperature', 'temp'],
            'property_name': ['property_name', 'property', 'metric'],
            'property_value': ['property_value', 'value', 'measurement'],
            'unit': ['unit', 'units'],
            'source': ['source', 'reference', 'dataset']
        }
        for target, variants in col_map.items():
            for v in variants:
                if v in df.columns and target not in df.columns:
                    df = df.rename(columns={v: target})
                    print(f"   🔁 Mapped '{v}' → '{target}'")
    
    # Keep only expected columns + drop rows with missing critical values
    available_cols = [c for c in EXPECTED_COLS if c in df.columns]
    if not available_cols:
        print(f"   ❌ No valid columns found in {os.path.basename(file_path)}")
        return None
        
    df = df[available_cols].dropna(subset=['material', 'property_name'])  # Keep rows with at least material+property
    return df

def build_database():
    """Main function: load all CSVs → validate → save to SQLite"""
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(CSV_FOLDER) if f.lower().endswith('.csv')]
    if not csv_files:
        print(f"❌ No CSV files found in {CSV_FOLDER}")
        return
    
    print(f"📁 Found {len(csv_files)} CSV file(s) to process")
    all_dfs = []
    
    for csv_file in csv_files:
        csv_path = os.path.join(CSV_FOLDER, csv_file)
        print(f"\n🔄 Processing: {csv_file}")
        df = load_and_validate_csv(csv_path)
        if df is not None and not df.empty:
            print(f"   ✅ Loaded {len(df)} valid rows")
            all_dfs.append(df)
        else:
            print(f"   ⚠️ Skipped (empty or invalid)")
    
    if not all_dfs:
        print("\n❌ No valid data loaded from any CSV file")
        return
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n📊 Combined dataset: {len(combined_df)} total rows")
    
    # Show preview
    print("\n🔍 Preview (first 3 rows):")
    print(combined_df.head(3).to_string(index=False))
    
    # Save to SQLite
    print(f"\n💾 Saving to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    combined_df.to_sql('material_data', conn, if_exists='replace', index=False)
    
    # Verify
    count = conn.execute("SELECT COUNT(*) FROM material_data").fetchone()[0]
    cols = [desc[1] for desc in conn.execute("PRAGMA table_info(material_data)")]
    conn.close()
    
    print(f"✅ Database created successfully!")
    print(f"   📦 Records: {count}")
    print(f"   🗂️ Columns: {cols}")
    print(f"\n💡 To inspect manually, run:")
    print(f"   python -c \"import sqlite3; c=sqlite3.connect('{DB_PATH}'); print(c.execute('SELECT * FROM material_data LIMIT 2').fetchall())\"")

if __name__ == "__main__":
    build_database()