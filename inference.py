from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from keras import models

# Initialize the FastAPI app
app = FastAPI()

# Load your Keras model (provide the path to your model file)
model = models.load_model('saved_model.keras')

# Define the input schema
class ImageRequest(BaseModel):
    image_base64: str

# Function to preprocess the image
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((256, 256))  # Resize the image to the required input size (256x256)
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.get("/")
def hello():
    return "done"

@app.post("/predict/")
async def predict(request: ImageRequest):
    try:
        # Decode the base64 string to an image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_data))

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Run the image through the model to get predictions
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions[0])
        predicted_label = "Your label based on the index"  # Replace with actual label lookup

        return {"predicted_index": int(predicted_index), "label": predicted_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
