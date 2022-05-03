import enum
import queue
import RPi.GPIO as GPIO
import statistics
import time

class EyeEnum(enum.Enum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2
    CLOSED = 3

class EyeTrackingModule:
    def __init__(self, pinL, pinR, sensorThreshold):
        GPIO.setmode(GPIO.BCM)

        self._pinL = pinL
        self._pinR = pinR
        self._sensorThreshold = sensorThreshold

    def _readInfra(self, pin):
        GPIO.setup(pin, GPIO.OUT)
        
        # Charge Capacitor
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.01)

        pulse_start = time.time()
        
        # Discharge Capacitor
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        while GPIO.input(pin) > 0:
            pass
        
        # Duration
        return time.time() - pulse_start
    
    # Public Methods
    def eyePositionCurr(self):
        leftVal = self._readInfra(self._pinL)
        rightVal = self._readInfra(self._pinR)

        # Core Logic
        if leftVal < self._sensorThreshold and rightVal > self._sensorThreshold:
            return EyeEnum.LEFT
        elif leftVal > self._sensorThreshold and rightVal < self._sensorThreshold:
            return EyeEnum.RIGHT
        elif leftVal > self._sensorThreshold and rightVal > self._sensorThreshold:
            return EyeEnum.CENTER
        elif leftVal < self._sensorThreshold and rightVal < self._sensorThreshold:
            return EyeEnum.CLOSED

    def eyePositionMode(self, duration):
        startTime = time.time()
        samples = []
        while time.time() - startTime < duration:
            leftVal = self._readInfra(self._pinL)
            rightVal = self._readInfra(self._pinR)
            
            # Core Logic
            # print(f'LOG: Left {leftVal}')
            # print(f'LOG: Right {rightVal}')
            if leftVal < self._sensorThreshold and rightVal > self._sensorThreshold:
                samples.append(EyeEnum.LEFT)
            elif leftVal > self._sensorThreshold and rightVal < self._sensorThreshold:
                samples.append(EyeEnum.RIGHT)
            elif leftVal > self._sensorThreshold and rightVal > self._sensorThreshold:
                samples.append(EyeEnum.CENTER)
            elif leftVal < self._sensorThreshold and rightVal < self._sensorThreshold:
                samples.append(EyeEnum.CLOSED)
        
        return statistics.mode(samples)