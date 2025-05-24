from fastapi import FastAPI
import base64
from PIL import Image
import io
from infer import infer


app = FastAPI()

@app.post("/predict/")
async def predict(img_data:str):
    img = io.BytesIObase64.b64decode(img_data)
    img = Image.open(img)
    prediction = infer(img)
    return {"prediction": prediction}

