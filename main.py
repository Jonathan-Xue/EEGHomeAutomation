import subprocess
import threading
import time

#from eye_tracking_module import eye_tracking
#from object_detection_module import object_detection
from eeg_module import eeg

def main():
    p = subprocess.Popen(['muselsl', 'stream'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, encoding='utf-8', errors='replace')
    while True:
        output = p.stdout.readline()

        if output == '' and p.poll() is not None:
            break
        if output:
            print(output)
    print('done')

if __name__ == '__main__':
    main()