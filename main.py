import concurrent.futures
import os
import requests
import threading
import time

from eye_tracking_module import EyeEnum, EyeTrackingModule
from object_detection_module import ObjectDetectionModule
from eeg_module import CommandEnum, EEGModule

# Constants
DURATION = 10

# Main
def main():
    eyeModuleTriggerThreshold = 3.0

    # Modules
    eyeModule = EyeTrackingModule(pinL = 5, pinR = 6, sensorThreshold = 300)
    objectModule = ObjectDetectionModule(eyeTrackingModule=eyeModule, modelPath='./models/efficientdet_lite0.tflite', cameraId=0, width=640, height=480, numThreads=4, enableEdgeTPU=False)
    eegModule = EEGModule(modelPath='./models/model_RF.pkl')

    # Loop
    startTime = float('inf')
    while True:
        # Eyes Closed
        if eyeModule.eyePosition() == EyeEnum.CLOSED:
            if startTime == float('inf'):
                startTime = time.time()
            elif time.time() - startTime > eyeModuleTriggerThreshold:
                # Thread Pool
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    objectModuleThread = executor.submit(objectModule.objectDetection, DURATION)
                    eegModuleThread = executor.submit(eegModule.modelPrediction, DURATION)

                    position = eyeModuleThread.result()
                    objects = objectModuleThread.result()
                    command = eegModuleThread.result()

                    # Device
                    device = None
                    if position == EyeEnum.LEFT:
                        device = objects['left'][0]['label'] if len(objects['left']) > 0 else None
                    elif position == EyeEnum.CENTER:
                        device = objects['center'][0]['label'] if len(objects['center']) > 0 else None
                    elif position == EyeEnum.RIGHT:
                        device = objects['right'][0]['label'] if len(objects['right']) > 0 else None

                    # Send Request
                    # requests.post('http://127.0.0.1:5000/rpi', json={'device': device, 'command': command.name})
                    startTime = float('inf')

if __name__ == '__main__':
    main()