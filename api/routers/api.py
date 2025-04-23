from fastapi import APIRouter, HTTPException, File, UploadFile, Form, WebSocket, WebSocketDisconnect 
from fastapi.responses import JSONResponse
import io
import numpy as np
from PIL import Image
from pydantic import BaseModel
from typing import Optional, List, Dict
import cv2
import torch

router = APIRouter()

# model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# # Load your custom YOLOv5 model (replace with your path)
# model = torch.load('path/to/your/custom_yolo_model.pt')
# model.eval()  # Set the model to evaluation mode

# detect objects in the frame. 
def detect_objects(frame: np.ndarray) -> List[Dict]:
    # Example dummy detection: bounding boxes with random values
    # Replace with YOLO or other detection models
    detections = [
        {"class": "person", "confidence": 0.95, "box": [50, 50, 200, 200]},
        {"class": "car", "confidence": 0.85, "box": [300, 100, 500, 400]},
    ]
    return detections

# Detect objects with 1 single image or frame
@router.post("/api/hazard/detect/image")
async def detect_image(file: UploadFile):
    # Read image from the uploaded file
    image_data = await file.read()

    # Convert to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform object detection (replace with your actual detection model)
    detections = detect_objects(img)

    # Return detection details as JSON
    return JSONResponse(content={"detections": detections})

@router.get("/api/scene/describe")
async def describe(  
            image: bytes = File(...),  # Accepting image as a file parameter
            focus_objects: List[str] = Form(...)  # Accepting a list of strings from form data
        ):
    # Convert the image bytes into a Pillow Image object
    image = Image.open(io.BytesIO(image))

    # Process the image (for demonstration purposes, we'll just print the image size)
    image_size = image.size

    # Return the image size and the list of strings as part of the response
    return JSONResponse(content={
        "received_strings": focus_objects,
        "image_size": image_size,
        "description": "Description is produced by VLM. This is a dummy description"
    })
