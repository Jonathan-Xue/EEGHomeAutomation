from pylsl import StreamInlet, resolve_byprop

import enum
import time
import numpy as np
import joblib
import os
import queue

class CommandEnum(enum.Enum):
    TURN_OFF = 0
    TURN_ON = 1

class EEGModule:
    def __init__(self, modelPath, channels=4, bufferLength=25, shiftLength=0.4, windowLength=1903):
        self._channels = channels
        self._bufferLength = bufferLength
        self._windowLength = windowLength
        self._shiftLength = shiftLength

        # Setup Stream
        try:
            print('Looking for an EEG stream')
            self._streams = resolve_byprop('type', 'EEG', timeout=10)

            if len(self._streams) == 0:
                raise RuntimeError('Can\'t find EEG stream.')
        except RuntimeError as e:
            print(e)

        # Load Model
        self.model = joblib.load(modelPath)

    # Public Methods
    def modelPrediction(self, duration):
        try:
            print("Start acquiring data from EEG stream")
            inlet = StreamInlet(self._streams[0], max_chunklen=50)
            info = inlet.info()
            fs = int(info.nominal_srate())
            eegBuffer = np.zeros((int(fs * self._bufferLength), self._channels), dtype=np.float32)

            startTime = time.time()
            timeDelta = 0.
            i = lastChunkLen = 0
            lastIdx = 0

            while timeDelta <= duration:
                eegData, _ = inlet.pull_chunk(timeout=5, max_samples=int(fs * self._shiftLength))
                channelsData = np.array(eegData, dtype=np.float32)[:, 0:self._channels]
                eegBuffer[i:i+channelsData.shape[0], :] = channelsData
                lastIdx = i
                lastChunkLen = channelsData.shape[0]
                i += channelsData.shape[0]
                timeDelta = time.time() - startTime

            ts = eegBuffer[lastIdx+lastChunkLen - self._windowLength:lastIdx+lastChunkLen, :]
            xTest = np.expand_dims(ts.ravel(), axis=0)
            yPred = self.model.predict(xTest)

            # Output
            if yPred[0] == "turn_on":
                return CommandEnum.TURN_ON
            else:
                return CommandEnum.TURN_OFF
        except Exception as e:
            print(e)
        except KeyboardInterrupt:
            print('Closing')