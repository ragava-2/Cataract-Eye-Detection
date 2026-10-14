import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, GlobalAveragePooling2D

# Define the model path
model_path = 'squeezenet.h5'

# Create a dummy model
# Expected input: (224, 224, 3)
# Expected output: 3 classes
model = Sequential([
    tf.keras.Input(shape=(224, 224, 3)),
    Conv2D(16, (3, 3), activation='relu'),
    GlobalAveragePooling2D(),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the model
model.save(model_path)
print(f"Dummy model saved to {os.path.abspath(model_path)}")
