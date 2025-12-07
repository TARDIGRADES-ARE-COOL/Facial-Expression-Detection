import numpy as np
import tensorflow as tf
from datasets import load_dataset

image_col = "image"
label_col = "label"
IMG_SIZE = 64
MODEL_PATH = "facial_expressions_cnn.keras"

def build_model(num_classes):
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
    return model

def load_data():
    ds = load_dataset("seaurkin/facial_exrpressions")

    split_name = "train" if "train" in ds else list(ds.keys())[0]
    data = ds[split_name]

    if "train" in ds and "validation" in ds:
        train_hf = ds["train"]
        val_hf = ds["validation"]
    else:
        temp = data.train_test_split(test_size=0.2, seed=42)
        train_hf = temp["train"]
        val_hf = temp["test"]

    label_feature = train_hf.features.get(label_col, None)
    if label_feature is not None and hasattr(label_feature, "num_classes"):
        num_classes = label_feature.num_classes
    else:
        labels_np = np.array(train_hf[label_col])
        num_classes = int(labels_np.max()) + 1

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

    return train_hf, val_hf, X_train, y_train, X_val, y_val, num_classes

if __name__ == "__main__":
    train_hf, val_hf, X_train, y_train, X_val, y_val, num_classes = load_data()
    model = build_model(num_classes)

    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32
    )

    model.save(MODEL_PATH)
    print("Saved model.")
