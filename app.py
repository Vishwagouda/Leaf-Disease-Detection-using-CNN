import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Model path
MODEL_PATH = "tomato_model.keras"

# Class names
CLASS_NAMES = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold',
    'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot', 'Yellow_Leaf_Curl', 'healthy'
]

# Load model
print("Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Preprocessing
def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction
def predict_disease(pil_img: Image.Image):
    arr = preprocess_image(pil_img)
    preds = model.predict(arr)
    index = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds)) * 100
    label = CLASS_NAMES[index]
    return f"Disease: {label}\nConfidence: {confidence:.2f}%"

# Gradio UI
iface = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil", label="Upload a Tomato Leaf Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Tomato Leaf Disease Classifier",
    description="Upload a tomato leaf image and get the predicted disease."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
