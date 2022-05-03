# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""

from object_detector import ObjectDetector
from object_detector import ObjectDetectorOptions

import argparse
import cv2
import sys
import time
import queue
import utils


class ObjectDetectionModule:
    def __init__(self):
        pass


    def objectDetection(self, model: str, camera_id: int, width: int, height: int, num_threads: int, enable_edgetpu: bool, duration: float) -> None:
        """Continuously run inference on images acquired from the camera.
        Args:
        model: Name of the TFLite object detection model.
        camera_id: The camera id to be passed to OpenCV.
        width: The width of the frame captured from the camera.
        height: The height of the frame captured from the camera.
        num_threads: The number of CPU threads to run the model.
        enable_edgetpu: True/False whether the model is a EdgeTPU model.
        """

        # Start capturing video input from the camera
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Initialize the object detection model
        options = ObjectDetectorOptions(num_threads=num_threads, score_threshold=0.5, max_results=3, enable_edgetpu=enable_edgetpu)
        detector = ObjectDetector(model_path=model, options=options)

        # Continuously capture images from the camera and run inference
        start_time = time.time()
        while cap.isOpened() and time.time() - start_time < duration:
            success, image = cap.read()
            if not success:
                sys.exit(
                    'ERROR: Unable to read from webcam. Please verify your webcam settings.'
                )

            image = cv2.rotate(image, cv2.ROTATE_180)

            # Run object detection estimation using the model.
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = detector.detect(rgb_image)

            # Dictionary of detected objects
            objects = {'left': [], 'center': [], 'right': []}
            for d in detections:
                center = (d[0][2] + d[0][0]) / 2 
                new_obj = {'left': d[0][0], 'top': d[0][1], 'right': d[0][2],
                        'bottom': d[0][3], 'label': d[1][0][0], 'score': d[1][0][1]}
                if center < image.shape[1]/3:
                    objects['left'].append(new_obj)
                elif center < 2*image.shape[1]/3:
                    objects['center'].append(new_obj)
                else:
                    objects['right'].append(new_obj)
            if objects != {'left': [], 'center': [], 'right': []}:
                cap.release()
                cv2.destroyAllWindows()
                return objects

        cap.release()
        cv2.destroyAllWindows()
        return {'left': [], 'center': [], 'right': []}
