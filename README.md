# Real-Time E-Commerce Recommendation Engine

## Project Overview

An end-to-end Machine Learning API that serves product recommendations by combining **historical user behavior** with **live external catalog data**. 

Unlike standard notebook-only data science projects, this system is built with production-ready architecture. It features an offline-trained **Item-Based Collaborative Filtering** model (using Olist e-commerce data) served via a high-performance **FastAPI** backend, alongside a live microservice integration for real-time trending products.

## Architecture & Tech Stack

* **Core ML Algorithm:** K-Nearest Neighbors (KNN) using Cosine Similarity on an Item-User sparse matrix.
* **Data Processing:** `pandas`, `scipy`, `numpy`
* **Model Serving & API:** `FastAPI`, `uvicorn`, `pydantic`
* **External Integration:** `requests` (Platzi Fake Store API)
* **Environment:** Python 3.x, Jupyter Notebooks for ETL/Training.

## Key Engineering Features

* **Data Pipeline & Merging:** Cleaned and joined multiple relational tables (Orders, Customers, Items, Reviews) to handle implicit and explicit feedback.
* **Cold Start Mitigation:** Designed a `/trending` fallback endpoint that fetches live market data when a user has no historical footprint.
* **Memory Efficiency:** Utilized `scipy.sparse` matrices to handle high-dimensional user-item interactions without crashing memory.
* **RESTful API Design:** Implemented proper HTTP methods (POST for inference, GET for fetching) and strict data validation using Pydantic.

## Project Structure
```text
ecom-recommendation-system-v1/
│
├── api/
│   └── main.py                 # FastAPI server and endpoint routing
├── data/
│   ├── clean_interactions.csv  # Processed data (generated)
│   ├── model_knn.pkl           # Pickled KNN model
│   └── item_user_matrix.pkl    # Pickled sparse matrix
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_prep.ipynb      # ETL and feature engineering
│   └── 03_model_training.ipynb # Matrix generation and KNN training
├── src/
│   └── inference.py            # Model loading and prediction logic
├── requirements.txt
└── README.md

```


## How to Run Locally

**1. Clone the repository and navigate to the project folder:**

```bash
git clone https://github.com/Afiskayode/ecom-recommendation-system-v1.git
cd ecom-recommendation-system-v1

```

**2. Set up a virtual environment and install dependencies:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```

**3. Run the API Server:**

```bash
uvicorn api.main:app --reload

```

**4. View Documentation:**
Open your browser and navigate to `http://127.0.0.1:8000/docs` to interact with the Swagger UI.

## 🔌 API Endpoints

### `POST /recommend`

Returns the nearest neighbor products based on the trained collaborative filtering model.

* **Request Body:** `{"product_id": "string"}`
* **Response:** JSON list of similar product IDs and their cosine distance.

### `GET /trending`

Simulates live microservice integration by fetching current trending items from an external catalog (Platzi API).

* **Response:** JSON list of live product metadata (titles, prices, images).
