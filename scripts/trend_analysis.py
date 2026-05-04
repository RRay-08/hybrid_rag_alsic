import sqlite3
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

def analyze_trend(db_path, base_material, target_property, param_keyword):
    """
    Analyzes how a property changes with a varying parameter.
    Example: analyze_trend(DB, 'Al_A356', 'tensile_strength', 'wt_SiC')
    """
    conn = sqlite3.connect(db_path)
    query = """
        SELECT material, property_value
        FROM material_data
        WHERE material LIKE ? AND property_name = ?
        ORDER BY material
    """
    rows = conn.execute(query, [f"%{base_material}%", target_property]).fetchall()
    conn.close()

    if len(rows) < 3:
        return {"status": "insufficient_data", "message": f"Need ≥3 data points, found {len(rows)}"}

    # Extract numeric parameter values from material names
    x_vals, y_vals, labels = [], [], []
    for mat, val in rows:
        # Look for number before the param_keyword
        pattern = rf'(\d+(?:\.\d+)?)\s*{param_keyword}'
        match = re.search(pattern, mat, re.IGNORECASE)
        if match:
            x_vals.append(float(match.group(1)))
            y_vals.append(float(val))
            labels.append(mat)

    if len(x_vals) < 3:
        return {"status": "insufficient_data", "message": f"Could not extract numeric values for '{param_keyword}'"}

    x = np.array(x_vals)
    y = np.array(y_vals)

    # Fit models
    try:
        # Linear
        p1 = np.polyfit(x, y, 1)
        y_lin = np.polyval(p1, x)
        r2_lin = 1 - np.sum((y - y_lin)**2) / np.sum((y - np.mean(y))**2)

        # Quadratic (only if ≥4 points)
        if len(x) >= 4:
            p2 = np.polyfit(x, y, 2)
            y_quad = np.polyval(p2, x)
            r2_quad = 1 - np.sum((y - y_quad)**2) / np.sum((y - np.mean(y))**2)
            use_quad = r2_quad > r2_lin + 0.1
        else:
            use_quad = False
    except Exception as e:
        return {"status": "error", "message": str(e)}

    # Select best fit
    if use_quad:
        trend_type, r2, coeffs, label = "quadratic", r2_quad, p2, "2nd-order polynomial"
    else:
        trend_type, r2, coeffs, label = "linear", r2_lin, p1, "linear"

    # Direction
    slope = coeffs[0] if trend_type == "linear" else coeffs[1]
    direction = "increases" if slope > 0 else "decreases"

    # Uncertainty (std of residuals)
    residuals = y - np.polyval(coeffs, x)
    uncertainty = np.std(residuals)

    return {
        "status": "success",
        "trend_type": trend_type,
        "r2": r2,
        "direction": direction,
        "label": label,
        "uncertainty": f"±{uncertainty:.1f}",
        "data_points": [{"material": l, "x": xv, "y": yv} for l, xv, yv in zip(labels, x_vals, y_vals)],
        "description": f"{target_property} {direction} with {param_keyword} ({label}, R² = {r2:.2f}, n={len(x)})"
    }