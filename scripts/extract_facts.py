import json
import re
import sqlite3

# Configuration
CHUNKS_FILE = "../data/processed/chunks.json"
DB_PATH = "../data/processed/materials.db"
OUTPUT_FILE = "../data/processed/extracted_facts.json"

# Simple patterns to find material-property-temperature facts
# These are examples - you can expand them later
PATTERNS = {
    "material": r'\b(Steel_\w+|Aluminum_\w+|Titanium_\w+|Inconel_\w+|[A-Z][a-z]+-\d{3,4})\b',
    "temperature": r'(\d{2,4})\s*°?\s*C|(\d{2,4})\s*degrees?\s*(?:C|Celsius)',
    "property_value": r'(\d+(?:\.\d+)?)\s*(MPa|percent|%|GPa|°C|hours?)',
    "property_name": r'(tensile_strength|yield_strength|elongation|creep_rate|hardness|ductility)'
}

def extract_facts_from_text(text):
    """Find potential facts using regex patterns"""
    facts = []
    
    # Find all matches for each pattern
    materials = re.findall(PATTERNS["material"], text, re.IGNORECASE)
    temps = re.findall(PATTERNS["temperature"], text, re.IGNORECASE)
    values = re.findall(PATTERNS["property_value"], text, re.IGNORECASE)
    props = re.findall(PATTERNS["property_name"], text, re.IGNORECASE)
    
    # Simple combination: if we find at least one of each, create a fact
    if materials and temps and values and props:
        # Take first match of each (simplified)
        fact = {
            "material": materials[0] if isinstance(materials[0], str) else materials[0][0],
            "temperature_c": int(temps[0][0] or temps[0][1]),
            "property": props[0],
            "value": float(values[0][0]),
            "unit": values[0][1],
            "confidence": "low",  # Regex is not perfect
            "source_snippet": text[:100] + "..."  # First 100 chars for reference
        }
        facts.append(fact)
    
    return facts

def main():
    # Load chunks
    print(f"📖 Loading chunks from {CHUNKS_FILE}")
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    all_facts = []
    print(f"🔍 Scanning {len(chunks)} chunks for facts...")
    
    for chunk in chunks:
        facts = extract_facts_from_text(chunk["text"])
        for fact in facts:
            fact["source_chunk"] = chunk["chunk_id"]
            fact["source_file"] = chunk["source_file"]
            all_facts.append(fact)
    
    # Save extracted facts
    print(f"💾 Found {len(all_facts)} potential facts. Saving to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_facts, f, indent=2, ensure_ascii=False)
    
    # Also add to SQLite database (bridge between unstructured → structured)
    print(f"🔗 Adding facts to SQLite database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table for extracted facts if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS extracted_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material TEXT,
            temperature_c INTEGER,
            property TEXT,
            value REAL,
            unit TEXT,
            confidence TEXT,
            source_chunk TEXT,
            source_file TEXT,
            source_snippet TEXT
        )
    ''')
    
    # Insert facts
    for fact in all_facts:
        cursor.execute('''
            INSERT INTO extracted_facts 
            (material, temperature_c, property, value, unit, confidence, source_chunk, source_file, source_snippet)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            fact["material"], fact["temperature_c"], fact["property"],
            fact["value"], fact["unit"], fact["confidence"],
            fact["source_chunk"], fact["source_file"], fact["source_snippet"]
        ))
    
    conn.commit()
    conn.close()
    
    print("✅ extract_facts.py completed successfully")
    print(f"📊 Summary: {len(all_facts)} facts extracted and stored")

if __name__ == "__main__":
    main()