import tensorflow as tf

from absl import logging
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

# Setup
assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')
logging.set_verbosity(logging.ERROR)

# Model
print("Training Model")
spec = model_spec.get('efficientdet_lite1')
train_data, validation_data, test_data = object_detector.DataLoader.from_csv('./detections.csv')
model = object_detector.create(train_data, model_spec=spec, batch_size=32, train_whole_model=True, validation_data=validation_data)

print("Evaluating Tensorflow Model")
model.evaluate(test_data)

# Export
print("Exporting Model")
model.export(export_dir='.')

print("Evaluating TensorflowLite Model")
model.evaluate_tflite('model.tflite', test_data)