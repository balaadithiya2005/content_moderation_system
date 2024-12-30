# content_moderation_system
import torch
from transformers import CLIPProcessor, CLIPModel, pipeline
from fastapi import FastAPI, UploadFile, Form
from typing import List
import cv2
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load CLIP model for multimodal analysis
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Sentiment analysis pipeline for text
sentiment_analyzer = pipeline("sentiment-analysis")

# Risk thresholds
RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 1.0
}

# Function to calculate risk score based on model outputs
def calculate_risk_score(text_score, image_score):
    return (text_score + image_score) / 2

# Explainable AI
EXPLANATIONS = {
    "text": "Text content flagged for inappropriate language or sentiment.",
    "image": "Image flagged for containing sensitive or explicit content."
}

# Function to analyze text content
def analyze_text(text):
    result = sentiment_analyzer(text)[0]
    score = result['score'] if result['label'] == "NEGATIVE" else 0
    return score, EXPLANATIONS["text"] if score > RISK_THRESHOLDS["low"] else None

# Function to analyze image content
def analyze_image(image):
    # Preprocess image
    inputs = clip_processor(images=image, return_tensors="pt")
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    score = logits_per_image.max().item()  # Simplified score
    return score, EXPLANATIONS["image"] if score > RISK_THRESHOLDS["low"] else None

# API Endpoint for content moderation
@app.post("/moderate")
async def moderate_content(
    text: str = Form(...),
    files: List[UploadFile] = []
):
    explanations = []
    text_score, text_explanation = analyze_text(text)
    if text_explanation:
        explanations.append(text_explanation)

    max_image_score = 0
    for file in files:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_score, image_explanation = analyze_image(image)
        max_image_score = max(max_image_score, image_score)
        if image_explanation:
            explanations.append(image_explanation)

    # Calculate overall risk score
    overall_score = calculate_risk_score(text_score, max_image_score)
    risk_level = "low"
    for level, threshold in RISK_THRESHOLDS.items():
        if overall_score >= threshold:
            risk_level = level

    return {
        "risk_level": risk_level,
        "risk_score": overall_score,
        "explanations": explanations
    }

# Run with `uvicorn filename:app --reload`
