# JustThinkI(o)T
foobar

## Hardware
- Muse2
- Pololu QTR-1RC Reflectance Sensor (x2)
- Raspberry Pi 4B
- Raspberry Pi Camera Module

## Relevant Documentation
[MuseLSL](https://github.com/alexandrebarachant/muse-lsl)

[Eye-Tracking Reference](https://create.arduino.cc/projecthub/H0meMadeGarbage/eye-motion-tracking-using-infrared-sensor-227467)

## Setup
### Software
To setup *muselsl* and *pylsl*, execute the following commands on the Raspberry Pi
```python
# Additional Dependencies
sudo apt-get install qtbase5-dev
sudo apt-get install libqt5core5a

# Pylsl & Muselsl
pip install pylsl
pip install muselsl

# Compile liblsl32.so for ARM architecture
git clone https://github.com/sccn/liblsl.git
cd liblsl
./standalone_compilation_linux.sh
mv liblsl.so /path/to/liblsl32.so

# Additional Dependencies
sudo apt-get install libatlast-base-dev
```

### Hardware
Attach the 5V, data, and ground pins of the QTR-1RC sensors to the corresponding pins on the Raspberry Pi. The data pins are specified in [main.py](main.py).

## Training The Models
- [EEG][train/eeg/README.md)
- [Object Detection](train/object_detection/README.md)

## Execution
- Turn on the Muse. By default, it will enter pairing mode, as indicated by the strobing white LED. 
- Setup the server on a separate device in the same network as the Raspberry Pi. Update the IP and port in [main.py](main.py).
    - ```python server.py```
- Setup the LSL stream on the Raspberry Pi. If it states No Devices Found, verify the Muse is turned on and in pairing mode and try again.
    - ```muselsl stream```
- Start the script on the Raspberry Pi. You can see the output results at server's base address.
    - ```python main.py```