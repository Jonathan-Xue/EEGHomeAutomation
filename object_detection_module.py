from eye_tracking_module import EyeTrackingModule, PositionEnum
from object_detector import ObjectDetector
from object_detector import ObjectDetectorOptions

import cv2
import statistics
import sys
import time

class ObjectDetectionModule:
    def __init__(self, eyeTrackingModule, modelPath, cameraId, width, height, numThreads, enableEdgeTPU):
        self._eyeTrackingModule = eyeTrackingModule
        self._modelPath = modelPath,
        self._cameraId = cameraId
        self._width = width
        self._height = height
        self._numThreads = numThreads
        self._enableEdgeTPU = enableEdgeTPU

        # Load Model
        options = ObjectDetectorOptions(num_threads=numThreads, score_threshold=0.5, max_results=3, enable_edgetpu=enableEdgeTPU)
        self._model = ObjectDetector(model_path=modelPath, options=options)

    # Public Methods
    def objectDetection(self, duration):
        # Capture Video Input
        cap = cv2.VideoCapture(self._cameraId)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        
        # Capture Images & Run Inference
        samples = []
        startTime = time.time()
        while cap.isOpened() and time.time() - startTime < duration:
            # Eye Tracking
            position = self._eyeTrackingModule.eyePosition()

            # Object Recognition
            success, image = cap.read()
            if not success:
                sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')
            image = cv2.rotate(image, cv2.ROTATE_180)

            # Object Detection Estimation
            rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = self._model.detect(rgbImage)

            # Inference Samples
            potentialSamples = []
            for d in detections:
                newObj = {
                    'left': d[0][0],
                    'top': d[0][1],
                    'right': d[0][2],
                    'bottom': d[0][3], 
                    'label': d[1][0][0],
                    'score': d[1][0][1]
                }
                objCenter = (newObj['left'] + newObj['right']) / 2.0

                if objCenter < image.shape[1] / 3:
                    if position == PositionEnum.LEFT: potentialSamples.append(newObj)
                elif objCenter < 2 * image.shape[1] / 3:
                    if position == PositionEnum.CENTER: potentialSamples.append(newObj)
                else:
                    if position == PositionEnum.RIGHT: potentialSamples.append(newObj)
            
            # Highest Confidence Sample
            if len(potentialSamples):
                highConfidenceSample = max(potentialSamples, key=lambda x:x['score'])
                samples.append(highConfidenceSample['label'])

        cap.release()
        return statistics.mode(samples) if len(samples) else None
