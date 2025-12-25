# Heart Disease Prediction API

A machine learning powered web application that predicts the likelihood of heart disease based on patient health metrics.
Built using FastAPI, scikit-learn, and Jinja2, with an optional Dockersetup for deployment.

---

## Features

* **Machine Learning Model**

  * Support Vector Machine (SVM)
  * Trained on the UCI Heart Disease dataset
  * Standardized features with `StandardScaler`

* **FastAPI Backend**

  * JSON-based prediction API
  * Automatic request validation using Pydantic
  * Health check endpoint

* **Frontend UI**

  * Bootstrap-based form
  * Sends JSON requests via JavaScript
  * Displays prediction results instantly

---

## Model Inputs

The model expects the following features:

| Feature  | Description             |
| -------- | ----------------------- |
| age      | Age of patient          |
| sex      | 0 = Female, 1 = Male    |
| cp       | Chest pain type         |
| trtbps   | Resting blood pressure  |
| chol     | Cholesterol             |
| fbs      | Fasting blood sugar     |
| restecg  | Resting ECG             |
| thalachh | Max heart rate          |
| exng     | Exercise-induced angina |
| oldpeak  | ST depression           |
| slp      | Slope of ST segment     |
| caa      | Number of major vessels |
| thall    | Thalassemia             |

---

## Running Locally (Without Docker)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the server

```bash
uvicorn app:app --reload
```

### 3. Open in browser

```http://127.0.0.1:8000
```

---

## API Usage

### Open in browser

```http://127.0.0.1:8000/docs
```

### **POST /predict**

**Request (JSON):**

(Paste your json input)

```json
{
  "age": 45,
  "sex": 1,
  "cp": 2,
  "trtbps": 130,
  "chol": 250,
  "fbs": 0,
  "restecg": 1,
  "thalachh": 170,
  "exng": 0,
  "oldpeak": 1.2,
  "slp": 1,
  "caa": 0,
  "thall": 2
}
```

**Response:**

```json
{
  "prediction": 1
}
```

* `1` → High risk of heart disease
* `0` → Low risk

---

## Running with Docker

### Build the image

```bash
docker build -t heartdisease-api .
```

### Run the container

```bash
docker run -p 8000:8000 heartdisease-api
```

Then open:

```http://localhost:8000
```

---

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "OK"
}
```

---

## Tech Stack

* **Backend:** FastAPI, Python
* **ML:** scikit-learn (SVM)
* **Frontend:** HTML, Bootstrap, JavaScript
* **Deployment:** Docker
* **Data:** UCI Heart Disease Dataset
