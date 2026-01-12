# Facial Expression Detection (Beginner CNN)

A beginner-friendly facial expression classifier using a simple Keras **Sequential CNN** and a Hugging Face dataset.

This README is fully self-contained and includes:
- Setup
- Training script
- Prediction script (your own image)
- How to run
- Common mistakes
- Suggested `.gitignore`

---

## What this project does

- Loads the dataset `seaurkin/facial_exrpressions` from Hugging Face  
- Resizes images to **64×64**  
- Normalizes pixel values to **0..1**  
- Trains a small CNN with:
  - `Conv2D`
  - `MaxPooling2D`
  - `Flatten`
  - `Dense`
- Saves the trained model as `facial_expressions_cnn.keras`
- Predicts facial expressions for your own images

---

## Install

```bash
pip install datasets tensorflow pillow
```

Notes:
- `pip install PIL` will fail.
- The correct package is **pillow**.
- You still import using:
  ```python
  from PIL import Image
  ```

---

## Recommended folder layout

```text
Facial-Expression-Detection/
├─ facialexpression.py
├─ run.py
└─ facial_expressions_cnn.keras   (created after training)
```

---

## 1) Training script — `facialexpression.py`

Create a file named `facialexpression.py` and paste:

```python
import numpy as np
import tensorflow as tf
from datasets import load_dataset

# 1) Load dataset
ds = load_dataset("seaurkin/facial_exrpressions")

# Use train if it exists, otherwise first available split
split_name = "train" if "train" in ds else list(ds.keys())[0]
data = ds[split_name]

print("Splits:", list(ds.keys()))
print("Columns:", data.column_names)
print("Features:", data.features)

# 2) Set column names (change if your printed columns differ)
image_col = "image"
label_col = "label"

# 3) Create train/val
if "train" in ds and "validation" in ds:
    train_hf = ds["train"]
    val_hf = ds["validation"]
else:
    temp = data.train_test_split(test_size=0.2, seed=42)
    train_hf = temp["train"]
    val_hf = temp["test"]

# 4) Determine number of classes
label_feature = train_hf.features.get(label_col, None)
if label_feature is not None and hasattr(label_feature, "num_classes"):
    num_classes = label_feature.num_classes
else:
    labels_np = np.array(train_hf[label_col])
    num_classes = int(labels_np.max()) + 1

print("num_classes =", num_classes)

# 5) Convert HF images to numpy arrays
IMG_SIZE = 64

def to_arrays(hf_split):
    X, y = [], []
    for ex in hf_split:
        img = ex[image_col].convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        X.append(arr)
        y.append(ex[label_col])
    return np.array(X), np.array(y)

X_train, y_train = to_arrays(train_hf)
X_val, y_val = to_arrays(val_hf)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

# 6) Beginner-friendly CNN (Sequential)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ✅ Only trains when you run this file directly
if __name__ == "__main__":
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32
    )

    model.save("facial_expressions_cnn.keras")
    print("Saved model as facial_expressions_cnn.keras")
```

### Train

```bash
python facialexpression.py
```

After training, you should see:

```text
facial_expressions_cnn.keras
```

---

## 2) Prediction script — `run.py`

Create a file named `run.py` and paste:

```python
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from datasets import load_dataset

IMG_SIZE = 64

# If your model is in the same folder as run.py:
MODEL_PATH = "facial_expressions_cnn.keras"

# Otherwise, use an absolute path, e.g.:
# MODEL_PATH = r"C:\Users\joaqu\Desktop\ok\facial_expressions_cnn.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}\n"
        "Run facialexpression.py first to create it."
    )

model = tf.keras.models.load_model(MODEL_PATH)

# Get label names for readable output
ds = load_dataset("seaurkin/facial_exrpressions")
train = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
label_names = train.features["label"].names

def prep_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ✅ Set this to YOUR actual image file
# DO NOT point to a folder.
img_path = r"C:\Users\joaqu\Desktop\ok\test.jpg"

x = prep_image(img_path)

pred = model.predict(x, verbose=0)[0]
pred_class = int(np.argmax(pred))
confidence = float(np.max(pred))

print("Predicted:", label_names[pred_class])
print("Confidence:", confidence)

# Optional top-3
top3 = np.argsort(pred)[-3:][::-1]
print("Top-3:")
for i in top3:
    print(f"{label_names[i]}: {pred[i]:.4f}")
```

### Predict

```bash
python run.py
```

---

## Common mistakes

### 1) Folder path instead of file path

Wrong:
```python
img_path = r"C:\Users\joaqu\Desktop\ok"
```

Right:
```python
img_path = r"C:\Users\joaqu\Desktop\ok\test.jpg"
```

### 2) Model file not found

If you run `run.py` from a folder that doesn't contain the model, either:
- move `facial_expressions_cnn.keras` into that folder, or
- set an absolute `MODEL_PATH`.

---

## Suggested `.gitignore`

```txt
__pycache__/
*.pyc
.venv/
venv/
.env
*.keras
*.h5
data/
datasets/
```

Remove `*.keras` if you want to push your trained model file.

---

## Next improvements

- Add early stopping
- Add data augmentation
- Increase image size to 96 or 128 if your PC can handle it

---

simple end-to-end pipeline for:
**train → save → predict your own images**.
