import sqlite3

def compare_materials(db_path, materials, properties):
    """Compares multiple materials across specified properties."""
    conn = sqlite3.connect(db_path)
    data = {mat: {prop: None for prop in properties} for mat in materials}

    for mat in materials:
        for prop in properties:
            row = conn.execute(
                "SELECT property_value, unit FROM material_data WHERE material = ? AND property_name = ? LIMIT 1",
                (mat, prop)
            ).fetchone()
            if row:
                data[mat][prop] = {"value": row[0], "unit": row[1]}
    conn.close()

    # Calculate % change vs baseline
    baseline = materials[0]
    insights = []
    for mat in materials[1:]:
        for prop in properties:
            base = data[baseline].get(prop)
            comp = data[mat].get(prop)
            if base and comp and base["value"] != 0:
                pct = ((comp["value"] - base["value"]) / base["value"]) * 100
                insights.append(f"{mat}: {pct:+.1f}% {prop} vs {baseline}")
            else:
                insights.append(f"Insufficient data to compare {prop} for {mat}")

    return {"baseline": baseline, "data": data, "insights": insights}