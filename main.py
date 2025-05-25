from fastapi import FastAPI, File, Query, UploadFile, HTTPException, logger
from fastapi.responses import JSONResponse
import firebase_admin
from firebase_admin import credentials, storage, firestore
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from PIL import Image
from datetime import datetime
import os
import io

import urllib

from fertilizer_service import (
    get_latest_npk_values,
    estimate_bulk_density,
    convert_npk_to_kg_per_ha,
    get_fertilizer_values,
    calculate_nutrient_shortfall,
    calculate_fertilizer_application,
    replace_nan
)

from firebase_config import db_firestore, bucket  # Import shared Firestore & Storage

# Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (480, 640)

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Model
model = models.mobilenet_v2(pretrained=False)
num_classes = 11
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load("paddy_disease_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

class_names = [
    "bacterial_leaf_blight", "bacterial_leaf_streak", "bacterial_panicle_blight",
    "blast", "brown_spot", "dead_heart", "downy_mildew",
    "hispa", "nitrogen_deficiency", "normal", "tungro"
]

# Initialize FastAPI
app = FastAPI()

# Function to upload image to Firebase Storage
def upload_to_firebase(file_bytes, filename):
    try:
        blob = bucket.blob(filename)
        blob.upload_from_string(file_bytes, content_type='image/jpeg')
        blob.make_public()
        return blob.public_url
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Firebase upload error: {e}")

# Function to predict image class
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)[0][predicted].item()

    predicted_class = class_names[predicted.item()]
    return predicted_class, confidence

# Function to save results in Firestore (Auto-Generated ID + Timestamp Field)
def save_to_firestore(predicted_class, confidence, firebase_url):
    try:
        timestamp = datetime.datetime.now().isoformat()  # ISO timestamp
        doc_ref = db_firestore.collection("predictions").add({
            "prediction": predicted_class,
            "confidence": round(confidence, 3),
            "firebase_url": firebase_url,
            "timestamp": timestamp  # Save timestamp as a field inside the document
        })
        return doc_ref[1].id  # Return the Firestore document ID
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving to Firestore: {e}")

# API endpoint for image upload, prediction, and Firebase storage
@app.post("/predict")
async def predict_and_upload_image(file: UploadFile = File(...)):
    file_bytes = await file.read()
    firebase_path = f'uploaded_images/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{file.filename}'

    # Save image to Firebase Storage
    firebase_url = upload_to_firebase(file_bytes, firebase_path)

    # Predict disease
    predicted_class, confidence = predict_image(file_bytes)

    # Save prediction in Firestore (Auto ID + Timestamp field)
    doc_id = save_to_firestore(predicted_class, confidence, firebase_url)

    # Return response with document ID
    result = {
        "id": doc_id,  # Firestore Auto-Generated Document ID
        "prediction": predicted_class,
        "confidence": round(confidence, 3),
        "firebase_url": firebase_url,
        "timestamp": datetime.datetime.now().isoformat()  # Return timestamp
    }

    return JSONResponse(content=result)


@app.get("/observations")
async def get_observations():
    try:
        observations = []
        docs = db_firestore.collection("predictions").order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        
        for doc in docs:
            doc_data = doc.to_dict()
            observations.append({
                "id": doc.id,
                "timestamp": doc_data.get("timestamp"),
                "firebase_url": doc_data.get("firebase_url")  # Assuming 'firebase_url' is stored in Firestore
            })

        return JSONResponse(content={"observations": observations})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching observations: {e}")


@app.get("/observation/{doc_id}")
async def get_observation(doc_id: str):
    try:
        doc_ref = db_firestore.collection("predictions").document(doc_id)
        doc = doc_ref.get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail="Document not found")

        return JSONResponse(content={"id": doc.id, "data": doc.to_dict()})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching document: {e}")
    

@app.get("/npk_data")
async def get_npk_data():
    try:
        collection_name = "npk_sensor_data"
        docs = db_firestore.collection(collection_name).order_by("timestamp", direction=firestore.Query.DESCENDING).stream()

        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            if "timestamp" in doc_data:
                data.append({
                    "doc_id": doc.id,
                    "timestamp": doc_data["timestamp"]
                })

        if not data:
            raise HTTPException(status_code=404, detail="No data found in Firestore.")

        return JSONResponse(content={"npk_data": data})

    except Exception as e:
        print(f"Error fetching data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/npk_latest_data")
async def get_npk_data():
    try:
        collection_name = "npk_sensor_data"
        docs = db_firestore.collection(collection_name).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(7).stream()

        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            data.append({
                "doc_id": doc.id,
                "timestamp": doc_data.get("timestamp"),
                "data": doc_data  # Includes all document fields
            })

        if not data:
            raise HTTPException(status_code=404, detail="No data found in Firestore.")

        return JSONResponse(content={"npk_data": data})

    except Exception as e:
        print(f"Error fetching data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
    

@app.get("/npk_data/{doc_id}")
async def get_npk_data_by_id(doc_id: str):
    try:
        collection_name = "npk_sensor_data"
        doc_ref = db_firestore.collection(collection_name).document(doc_id)
        doc = doc_ref.get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail=f"Document with ID '{doc_id}' not found.")

        doc_data = doc.to_dict()
        return {
            "doc_id": doc.id,
            "timestamp": doc_data.get("timestamp"),
            "nitrogen": doc_data.get("nitrogen"),
            "phosphorus": doc_data.get("phosphorus"),
            "potassium": doc_data.get("potassium")
        }

    except Exception as e:
        print(f"Error fetching document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
    

@app.get("/age_groups")
async def get_unique_age_groups():
    """
    Retrieve unique age groups from Firestore, returning document IDs and age groups.
    """
    docs = db_firestore.collection("fertilizer_requirements").stream()
    
    unique_age_groups = {}
    
    for doc in docs:
        data = doc.to_dict()
        age_group = data.get("age_group")
        if age_group and age_group not in unique_age_groups:
            unique_age_groups[age_group] = doc.id  # Store first occurrence only

    return [{"doc_id": doc_id, "age_group": age_group} for age_group, doc_id in unique_age_groups.items()]


@app.get("/time_to_apply")
async def get_time_to_apply(age_group: str = Query(..., description="The age group to filter by")):
    """
    Retrieve time_to_apply values based on the provided age_group.
    Returns document IDs along with time_to_apply values.
    """
    docs = db_firestore.collection("fertilizer_requirements").where("age_group", "==", age_group).stream()
    
    time_to_apply_list = []
    
    for doc in docs:
        data = doc.to_dict()
        time_to_apply = data.get("time_to_apply")
        if time_to_apply:
            time_to_apply_list.append({"doc_id": doc.id, "time_to_apply": time_to_apply})

    return time_to_apply_list


@app.get("/get_fertilizer_recommendation")
async def get_fertilizer_recommendation(
    age_group: str = Query(..., description="Age group of the paddy plant"),
    time_to_apply: str = Query(..., description="Time to apply fertilizer"),
    soil_type: str = Query("loamy", description="Type of soil", alias="soil_type")  # Default to loamy
):
    try:
        print(f"Received request: age_group={age_group}, time_to_apply={time_to_apply}, soil_type={soil_type}")

        # Fetch recommended fertilizer values
        recommended_values = get_fertilizer_values(age_group, time_to_apply)
        print(f"Fetched recommended_values from Firestore: {recommended_values}")

        if None in recommended_values:
            print("No matching fertilizer data found in Firestore.")
            return {"error": "No matching fertilizer data found"}

        recommended_n, recommended_p2o5, recommended_k2o = recommended_values

        # Get the latest NPK values from the sensor data
        n, p, k = get_latest_npk_values()
        print(f"Fetched latest NPK values from Firestore: N={n}, P={p}, K={k}")

        if None in [n, p, k]:
            print("No sensor data available.")
            return {"error": "No sensor data available"}

        # Estimate bulk density based on the provided soil type
        bulk_density = estimate_bulk_density(soil_type)
        print(f"Estimated bulk density: {bulk_density}")

        # Convert the latest NPK values to kg/ha
        available_n, available_p, available_k = convert_npk_to_kg_per_ha(n, p, k, bulk_density)
        print(f"Converted NPK to kg/ha: N={available_n}, P={available_p}, K={available_k}")

        # Calculate the nutrient shortfall
        n_needed, p2o5_needed, k2o_needed = calculate_nutrient_shortfall(
            available_n, available_p, available_k, recommended_n, recommended_p2o5, recommended_k2o
        )

        # Convert NaN values to None before returning JSON
        n_needed, p2o5_needed, k2o_needed = replace_nan(n_needed), replace_nan(p2o5_needed), replace_nan(k2o_needed)
        print(f"Nutrient shortfall: N={n_needed}, P2O5={p2o5_needed}, K2O={k2o_needed}")

        # Calculate the fertilizer application needed
        urea_needed, tsp_needed, mop_needed = calculate_fertilizer_application(n_needed, p2o5_needed, k2o_needed)
        print(f"Fertilizer required: Urea={urea_needed}, TSP={tsp_needed}, MOP={mop_needed}")

        # Prepare the output data (convert NaN to None)
        result = {
            "age_group": age_group,
            "time_to_apply": time_to_apply,
            "soil_type": soil_type,
            "timestamp": datetime.utcnow().isoformat(),
            "fertilizer_required": {
                "urea_kg_per_ha": replace_nan(urea_needed),
                "tsp_kg_per_ha": replace_nan(tsp_needed),
                "mop_kg_per_ha": replace_nan(mop_needed)
            },
            "nutrient_shortfall": {
                "n_needed": replace_nan(n_needed),
                "p2o5_needed": replace_nan(p2o5_needed),
                "k2o_needed": replace_nan(k2o_needed)
            },
            "available_npk": {
                "n": replace_nan(available_n),
                "p": replace_nan(available_p),
                "k": replace_nan(available_k)
            },
            "recommended_fertilizer": {
                "urea_kg_per_ha": replace_nan(recommended_n),
                "tsp_kg_per_ha": replace_nan(recommended_p2o5),
                "mop_kg_per_ha": replace_nan(recommended_k2o)
            }
        }

        # Save result in Firestore collection (e.g., "fertilizer_recommendations")
        doc_ref = db_firestore.collection("fertilizer_recommendations").add(result)
        print(f"Saved recommendation in Firestore with ID: {doc_ref}")

        return result

    except Exception as e:
        print(f"Error in /get_fertilizer_recommendation: {e}")
        return {"error": f"Internal server error: {e}"}

@app.get("/get_fertilizer_recommendation_history")
async def get_fertilizer_recommendation_history():
    try:
        docs = db_firestore.collection("fertilizer_recommendations").order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        history = []
        for doc in docs:
            entry = doc.to_dict()
            entry["id"] = doc.id
            history.append(entry)
        return history
    except Exception as e:
        print(f"Error fetching history: {e}")
        return {"error": str(e)}
