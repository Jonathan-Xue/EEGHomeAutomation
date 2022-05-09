# Tensorflow Lite Models
## Overview
dataset.py
- Reformats the data and CSVs.
- Arguments/Flags:
    - -d: Download the relevant training, test, and validation data (as specified in classes) from Open Images V6.

test.py
- Runs object detection on an input image and returns and output image.

train.py
- Trains and evaluates the corresponding Tensorflow Lite object detection models.

## Relevant Links
[Object Detection with TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/tutorials/model_maker_object_detection)

[Open Images Dataset V6](https://storage.googleapis.com/openimages/web/download.html)

[FiftyOne Documentation](https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/datasets.html#dataset-zoo-open-images-v6)