import numpy as np
import tensorflow as tf
from PIL import Image
from datasets import load_dataset

IMG_SIZE = 64
MODEL_PATH = "facial_expressions_cnn.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# get label names
ds = load_dataset("seaurkin/facial_exrpressions")
train = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
label_names = train.features["label"].names

def prep_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

img_path = r"C:\Users\joaqu\Desktop\ok\test2.jpg"


x = prep_image(img_path)

pred = model.predict(x, verbose=0)[0]
pred_class = int(np.argmax(pred))

print("Predicted:", label_names[pred_class])
print("Confidence:", float(np.max(pred)))
