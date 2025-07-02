import tensorflow as tf

# Load the TensorFlow SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model("mobilenetv1_tf")

# Optional: enable optimizations for size/speed (quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert to TFLite
tflite_model = converter.convert()

# Save to file
with open("mobilenetv1.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversion to TFLite completed!")
