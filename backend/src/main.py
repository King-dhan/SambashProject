from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import io

app = FastAPI()

origins = [
    "http://localhost:3000",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

clf = DecisionTreeClassifier(random_state=42)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    data = pd.read_csv(io.BytesIO(contents))

    if 'Species' not in data.columns:
        return JSONResponse({"error": "The uploaded file must contain a 'Species' column."})

    X = data.drop(columns=['Species'])
    y = data['Species']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if not hasattr(clf, "tree_"): 
        clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions) * 100  
    X_test_dict = X_test.to_dict(orient="records")

    return JSONResponse({"predictions": predictions.tolist(), "accuracy": accuracy, "X_test": X_test_dict})



