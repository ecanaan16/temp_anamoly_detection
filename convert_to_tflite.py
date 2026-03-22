import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model("temperature_model.h5", compile=False)

# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional but recommended: optimization for smaller model
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert model
tflite_model = converter.convert()

# Save TFLite model
with open("temperature_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as temperature_model.tflite")