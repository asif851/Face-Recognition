import os
import numpy as np
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix


# Image and training parameters
img_size = (224, 224)
batch_size = 16
epochs = 50

train_dir = "dataset/train"
val_dir = "dataset/val"

# -----------------------------
# Check number of images per class
def count_images_in_directory(directory):
    print(f"\nImage count in '{directory}':")
    for cls in os.listdir(directory):
        cls_path = os.path.join(directory, cls)
        if os.path.isdir(cls_path):
            print(f"{cls}: {len(os.listdir(cls_path))} images")

count_images_in_directory(train_dir)
count_images_in_directory(val_dir)

# -----------------------------
#  Data augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_gen = ImageDataGenerator(rescale=1./255)

# -----------------------------
#  Load datasets
train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# -----------------------------
#  Model definition using MobileNetV2
base_model = MobileNetV2(input_shape=(*img_size, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(train_data.class_indices), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
#  Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# -----------------------------
# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[early_stop]
)

# -----------------------------
# ✅ Evaluate and report
val_loss, val_acc = model.evaluate(val_data)
print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")

# Classification report
val_data.reset()
predictions = model.predict(val_data, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes
class_labels = list(val_data.class_indices.keys())



print("\nClassification Report:")
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

print("\nConfusion Matrix:")
matrix = confusion_matrix(y_true, y_pred)
print(matrix)

# -----------------------------
#  Save model and reports
os.makedirs("model", exist_ok=True)
model.save("model/face_model.h5")
with open("model/class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

with open("model/metrics_report.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(matrix))
