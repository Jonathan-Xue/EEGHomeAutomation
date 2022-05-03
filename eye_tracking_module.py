import RPi.GPIO as GPIO
import statistics
import time

def read_infra(pin):
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

def eye_tracking(pin_l, pin_r, sensor_threshold, trigger_threshold, num_samples, sample_rate = 1.0):
    GPIO.setmode(GPIO.BCM)

    # Loop
    trigger_count = 0
    samples = []
    while True:
        left_val = read_infra(pin_l)
        right_val = read_infra(pin_r)
        
        # Core Logic
        print(f'LOG: {left_val}\t{right_val}')
        if trigger_count >= trigger_threshold:
            if left_val < sensor_threshold and right_val < sensor_threshold:
                print("LOG: Neutral")
                samples.append(0)
            elif left_val > sensor_threshold and right_val < sensor_threshold:
                print("LOG: Left")
                samples.append(-1)
            elif left_val < sensor_threshold and right_val > sensor_threshold:
                print("LOG: Right")
                samples.append(1)
        else:
            if left_val > sensor_threshold and right_val > sensor_threshold:
                print("LOG: Closed")
                trigger_count += 1
                samples.clear()

        # Exit Condition
        if len(samples) >= num_samples:
            return statistics.mode(samples)

        # Sample Rate 
        time.sleep(1.0 / sample_rate)

if __name__ == '__main__':
    main()