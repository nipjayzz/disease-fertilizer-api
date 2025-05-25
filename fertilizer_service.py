from firebase_config import db_firestore, bucket  # Import shared Firestore & Storage
from firebase_admin import firestore  # ✅ Import firestore
import math

def replace_nan(value):
    """Replace NaN values with None."""
    return None if isinstance(value, float) and math.isnan(value) else value

# Get latest NPK values from Firestore
def get_latest_npk_values():
    try:
        print("Fetching latest NPK values from Firestore...")
        
        # Fetch latest document
        docs = db_firestore.collection("npk_sensor_data").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1).stream()
        
        latest_record = None
        for doc in docs:
            latest_record = doc.to_dict()
            print(f"Fetched document: {latest_record}")
            break  # Only need the latest one

        if latest_record:
            # Handle missing or NaN values
            n = latest_record.get('nitrogen', None)
            p = latest_record.get('phosphorus', None)
            k = latest_record.get('potassium', None)

            if any(v is None for v in [n, p, k]):
                print("Warning: Some NPK values are missing in Firestore.")
                return None, None, None  # Ensure missing values are handled

            return n, p, k
        else:
            print("No NPK records found in Firestore.")
            return None, None, None

    except Exception as e:
        print(f"Error fetching latest NPK values: {e}")
        return None, None, None
    

# Estimate bulk density based on soil type
def estimate_bulk_density(soil_type):
    bulk_density_values = {"sandy": 1.3, "loamy": 1.3, "silty": 1.2, "clay": 1.1}
    return bulk_density_values.get(soil_type.lower(), 1.3)

# Convert NPK to kg/ha based on bulk density and soil depth
def convert_npk_to_kg_per_ha(n, p, k, bulk_density=1.3, soil_depth=10):
    factor = (soil_depth * bulk_density) / 10
    return replace_nan(n) * factor, replace_nan(p) * factor, replace_nan(k) * factor


# Convert P and K to their fertilizer equivalents (P₂O₅ and K₂O)
def convert_elemental_to_fertilizer_form(p, k):
    p2o5, k2o = replace_nan(p) * 2.29, replace_nan(k) * 1.2
    return p2o5, k2o


# Calculate nutrient shortfall (difference between recommended and available nutrients)
def calculate_nutrient_shortfall(available_n, available_p, available_k, recommended_n, recommended_p2o5, recommended_k2o):
    available_p2o5, available_k2o = convert_elemental_to_fertilizer_form(available_p, available_k)

    # Ensure all values are valid before subtraction
    n_shortfall = max(replace_nan(recommended_n) - replace_nan(available_n), 0)
    p2o5_shortfall = max(replace_nan(recommended_p2o5) - replace_nan(available_p2o5), 0)
    k2o_shortfall = max(replace_nan(recommended_k2o) - replace_nan(available_k2o), 0)

    return n_shortfall, p2o5_shortfall, k2o_shortfall



# Calculate the amount of each fertilizer required
def calculate_fertilizer_application(n_needed, p2o5_needed, k2o_needed):
    urea_needed = round(replace_nan(n_needed) / 0.46, 2) if n_needed > 0 else 0
    tsp_needed = round(replace_nan(p2o5_needed) / 0.46, 2) if p2o5_needed > 0 else 0
    mop_needed = round(replace_nan(k2o_needed) / 0.60, 2) if k2o_needed > 0 else 0
    return urea_needed, tsp_needed, mop_needed


# Get fertilizer values from Firestore based on age_group and time_to_apply
def get_fertilizer_values(age_group, time_to_apply):
    try:
        print(f"Fetching fertilizer values from Firestore: age_group={age_group}, time_to_apply={time_to_apply}")

        # Use `.where()` instead of `.filter()`
        query = db_firestore.collection("fertilizer_requirements") \
            .where("age_group", "==", age_group) \
            .where("time_to_apply", "==", time_to_apply)
            
        docs = query.stream()

        for doc in docs:
            data = doc.to_dict()
            print(f"Matching fertilizer document found: {data}")

            # Ensure values are valid before returning
            return replace_nan(data.get('urea_kg_per_ha')), replace_nan(data.get('tsp_kg_per_ha')), replace_nan(data.get('mop_kg_per_ha'))

        print("No matching fertilizer document found in Firestore.")
        return 0, 0, 0  # Return zero values instead of None
    except Exception as e:
        print(f"Error fetching fertilizer values: {e}")
        return 0, 0, 0


