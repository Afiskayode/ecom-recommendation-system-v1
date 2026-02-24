from fastapi import FastAPI, HTTPException
import requests
from pydantic import BaseModel
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.inference import get_recommendations


app = FastAPI(title="E-commerce Recommender API")

class RecommendationRequest(BaseModel):
    product_id: str

@app.get("/")
def home():
    return {"message":"Recommender Sytem is Running!"}


@app.post('/recommend')
def recommend(request: RecommendationRequest):
    results = get_recommendations(request.product_id)

    if results is None:
        raise HTTPException(status_code=404, detail="Product not found in history")
    
    return {"Input product": request.product_id, "Recommendations": results}

@app.get('/trending')
def get_live_trending():
    try:
        response = requests.get('https://api.escuelajs.co/api/v1/products?offset=0&limit=5')

        return {"source":"Platzi external API", "trending_products":response.json()}
    
    except:
        return {"error" : "Could not fetch live data"}



