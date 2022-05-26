import tensorflow as tf
import glob
import cv2
import numpy as np
from config import *


print(lp_images)
def representative_data_gen():
  dataset_list = tf.data.Dataset.list_files(lp_images)
  for i in range(100):
    image = next(iter(dataset_list))
    print(image)
    image = tf.io.read_file(image)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, [24, 94])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]

converter = tf.lite.TFLiteConverter.from_saved_model(recognition_edge_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_types = [tf.int8]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

with open('tflite/recognition.tflite', 'wb') as f:
  f.write(tflite_quant_model)