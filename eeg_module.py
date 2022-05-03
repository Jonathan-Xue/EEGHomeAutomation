from pylsl import StreamInlet, resolve_byprop

import enum
import time
import numpy as np
import joblib
import os

class CommandEnum(enum.Enum):
    TURN_OFF = 0
    TURN_ON = 1

class EEGModule:
    def __init__(self, channels=4, bufferLength=25, shiftLength=0.4, windowLength=1903):
        self.channels = channels
        self.bufferLength = bufferLength
        self.windowLength = windowLength
        self.shiftLength = shiftLength

    def setupStream(self) -> list:
        try:
            print('Looking for an EEG stream')
            streams = resolve_byprop('type', 'EEG', timeout=10)

            if len(streams) == 0:
                raise RuntimeError('Can\'t find EEG stream.')
            return streams
        except RuntimeError as e:
            print(e)

    def loadModel(self, modelPath):
        model = joblib.load(modelPath)
        return model

    def modelPrediction(self, model, streams, n=10) -> bool:
        try:
            print("Start acquiring data from EEG stream")
            inlet = StreamInlet(streams[0], max_chunklen=50)
            info = inlet.info()
            fs = int(info.nominal_srate())
            eegBuffer = np.zeros(
                (int(fs * self.bufferLength), self.channels), dtype=np.float32)

            startTime = time.time()
            timeDelta = 0.
            i = lastChunkLen = 0
            lastIdx = 0

            while timeDelta <= n:
                eegData, _ = inlet.pull_chunk(
                    timeout=5, max_samples=int(fs * self.shiftLength))
                channelsData = np.array(eegData, dtype=np.float32)[
                    :, 0:self.channels]
                eegBuffer[i:i+channelsData.shape[0], :] = channelsData
                lastIdx = i
                lastChunkLen = channelsData.shape[0]
                i += channelsData.shape[0]
                timeDelta = time.time() - startTime

            ts = eegBuffer[lastIdx+lastChunkLen - self.windowLength:lastIdx+lastChunkLen, :]
            xTest = np.expand_dims(ts.ravel(), axis=0)
            yPred = model.predict(xTest)

            # Output
            if yPred[0] == "turn_on":
                return CommandEnum.TURN_ON
            else:
                return CommandEnum.TURN_OFF

        except Exception as e:
            print(e)
        except KeyboardInterrupt:
            print('Closing')

# if __name__ == "__main__":
#     command_recognition = CommandRecognition()
#     streams = command_recognition.setupStream()
#     model = command_recognition.loadModel(os.path.abspath("./models/model_RF.pkl"))

#     while True:
#         res = command_recognition.modelPrediction(model, streams)
#         print(res)