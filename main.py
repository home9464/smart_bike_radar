"""
env: Coral Dev Board Mini

mkdir -p models
curl -OL https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
curl -OL  https://dl.google.com/coral/canned_models/coco_labels.txt
mv mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite coco_labels.txt models/

timedatectl set-timezone "America/Los_Angeles"


"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import io
import time
import json
import logging
import datetime
import subprocess
import platform
import signal
import _thread as T


import serial
import pytz
import pynmea2
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from playsound import playsound


SIMULATE = None
#SIMULATE = '/Users/ysun5/Desktop/tracker/2021_07_27_16_01_38.mp4'
#SIMULATE = '/Users/ysun5/Desktop/tracker/bike/video/2021_08_03_16_58_11.mp4'

DEVICE = '/dev/ttyS1'  # ToF distance sensor or GPS
#DEVICE = '/dev/cu.usbserial-14210'  # on adapter

USE_GPS = True

CAMERA_VIEW_WIDTH = 640
CAMERA_VIEW_HEIGHT = 480

PLAY_ALARM_SOUND = True

TIMEZONE_US_PACIFIC = pytz.timezone('US/Pacific')
TIMEZONE_UTC = pytz.timezone('UTC')

ROOT = os.path.dirname(os.path.realpath(__file__))

if SIMULATE is None:
    SAVE_PATH_ROOT = '/home/mendel/sdcard/bike'
else:
    SAVE_PATH_ROOT = '/Users/ysun5/Desktop/tracker/bike'

LOG_FILE_NAME = os.path.join(SAVE_PATH_ROOT, 'log', 'main.log')
SNAPSHOT_SAVE_PATH = os.path.join(SAVE_PATH_ROOT, 'image')
VIDEO_SAVE_PATH = os.path.join(SAVE_PATH_ROOT, 'video')

TIME_SYNCHRONIZED = False
GPS_CONNECTED = False

if SIMULATE is None:
    logging.basicConfig(filename=LOG_FILE_NAME,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

def sound_time_synchronized():
    playsound(os.path.join(ROOT, 'sound', 'time.synchronized.mp3'))

def sound_gps_connected():
    playsound(os.path.join(ROOT, 'sound', 'gps.connected.mp3'))

def sound_initialized():
    playsound(os.path.join(ROOT, 'sound', 'initialized.mp3'))

def sound_alarm():
    playsound(os.path.join(ROOT, 'sound', 'alarm.mp3'))


GPS = {
        'latitude': None,
        'longitude': None,
        'altitude': None,
        'speed': None
      }


def gps():
    """run this function in a thread to continuously acquire GPS data

    """
    ser = serial.Serial(DEVICE, 9600, timeout=5.0)
    NMEA = io.TextIOWrapper(io.BufferedRWPair(ser, ser))
    global GPS
    global GPS_CONNECTED
    global TIME_SYNCHRONIZED
    while True:
        try:
            for sentence in NMEA:
                sentence = sentence.strip()
                if "GNRMC" in sentence:
                    data = pynmea2.parse(sentence)
                    #if isinstance(data, pynmea2.types.talker.RMC):
                    sys_datetime = datetime.datetime.now(tz=TIMEZONE_UTC)
                    if data.datestamp is not None and data.timestamp is not None:
                        gps_datetime = datetime.datetime.combine(data.datestamp, data.timestamp, tzinfo=TIMEZONE_UTC).astimezone(TIMEZONE_US_PACIFIC)
                        if gps_datetime > sys_datetime:
                            sz = gps_datetime.strftime('%Y-%m-%d %H:%M:%S')  # '2015-11-20 16:14:50'
                            set_time_cmd = f'sudo timedatectl set-ntp false && sudo timedatectl set-time "{sz}"'
                            os.system(set_time_cmd)

                        if not TIME_SYNCHRONIZED:
                            sound_time_synchronized()
                            TIME_SYNCHRONIZED = True

                    GPS['latitude'] = data.latitude
                    GPS['longitude'] = data.longitude
                    GPS['speed'] = data.spd_over_grnd
                    if GPS['latitude'] is not None and GPS['longitude'] is not None:
                        if not GPS_CONNECTED:
                            sound_gps_connected()
                        GPS_CONNECTED = True
                if "$GNGGA" in sentence:
                    data = pynmea2.parse(sentence)
                    GPS['altitude'] = data.altitude
                time.sleep(0.1)
        except Exception as e:
            print('Error:', e)
            time.sleep(0.1)

##############################################
# car tracking
##############################################

class VehicleDetector:
    def __init__(self):
        if SIMULATE is None:  # on Edge TPU
            saved_model_tflite = os.path.join(ROOT, 'models', 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
            self.interpreter = tflite.Interpreter(model_path=saved_model_tflite,
                                              experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
        else:  # on CPU
            saved_model_tflite = os.path.join(ROOT, 'models', 'ssd_mobilenet_v2_coco_quant_postprocess.tflite')
            self.interpreter = tflite.Interpreter(model_path=saved_model_tflite)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.danger_region = []
        self.camera_view_width = 640
        self.camera_view_height = 480

        self.prev_area_ratio = 0
        self.possible_danger_index = 0
        self.interested_object_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        self.save_image_idx = 0


    def _get_output_tensor(self, index:int):
        """the 4 indices are 0, 1, 2, 3 represents TFLite_Detection_PostProcess
        Args:
            index: int; the index of postprocessing ops
                0: bounding boxes
                1: classes
                2: scores
                4: count
        """
        assert index >=0 and index <= 3
        output_detail = self.interpreter.get_output_details()[index]
        tensor = self.interpreter.get_tensor(output_detail['index'])
        return np.squeeze(tensor)  # from [1, ...] to [N], remove the batch dimension

    def detect(self, raw_frame, show_bounding_box=True):
        """detect if cars are present
        """
        screen_height, screen_width, channel = raw_frame.shape
        _, expected_height, expected_width, _ = self.input_details[0]['shape']
        input_shape = (expected_height, expected_width)
        input_data = cv2.resize(raw_frame, input_shape, interpolation = cv2.INTER_AREA)
        quantitized_input_data = input_data.astype(np.uint8)
        quantitized_input_data = np.expand_dims(quantitized_input_data, 0)   #[W, H, C] -> [b, W, H, C]
        self.interpreter.set_tensor(self.input_details[0]['index'], quantitized_input_data)
        self.interpreter.invoke()
        boxes = self._get_output_tensor(0)
        classes = self._get_output_tensor(1)
        scores = self._get_output_tensor(2)
        count = int(self._get_output_tensor(3))

        if show_bounding_box:
            for i in range(count):
                if scores[i] > 0.5 and int(classes[i]) in self.interested_object_classes:
                    ymin, xmin, ymax, xmax = boxes[i]
                    xmin = int(xmin*screen_width)
                    ymin = int(ymin*screen_height)
                    xmax = int(xmax*screen_width)
                    ymax = int(ymax*screen_height)
                    #cv2.rectangle(raw_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    # crop the box that contains a vehicle and save as '00001.jpg' ...
                    crop_vechicle_image = raw_frame[ymin:ymax, xmin:xmax]
                    if crop_vechicle_image.size > 0:
                        if ymax > (ymin + 50) and xmax > (xmin + 50):
                            #filename = os.path.join(SNAPSHOT_SAVE_PATH, f'{self.interested_object_classes[classes[i]]}-{scores[i]}-{self.save_image_idx:05d}.jpg')
                            filename = os.path.join(SNAPSHOT_SAVE_PATH, f'{self.save_image_idx:05d}.jpg')
                            cv2.imwrite(filename, crop_vechicle_image)
                            self.save_image_idx += 1
                            #cv2.rectangle(raw_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255),2)
                            #cv2.putText(raw_frame, f'{self.interested_object_classes[classes[i]]}, {scores[i]}', (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    """
                    area_occupied = (xmax-xmin) * (ymax-ymin)
                    current_area_ratio = area_occupied / (self.camera_view_width * self.camera_view_height)
                    if xmin >= (self.camera_view_width // 3):  # on my left
                        if current_area_ratio > self.prev_area_ratio:  # continuously approaching
                            self.prev_area_ratio = current_area_ratio
                            self.possible_danger_index += 1
                            cv2.rectangle(raw_frame, (xmin, ymin), (xmax, ymax), (255,0,0),2)
                        else:  # reset
                            self.prev_area_ratio = 0
                            self.possible_danger_index = 0

                        if self.possible_danger_index >= 5 and current_area_ratio >= 0.05:
                            print(scores[i], current_area_ratio, xmin, ymin, xmax, ymax)
                            cv2.rectangle(raw_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255),2)
                            cv2.putText(raw_frame, 'Danger', (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            return True
                    """
        return False

class DistanceMeasure:
    """measure distance with TF Luna Lidar
    """
    def __init__(self):
        if SIMULATE is None:
            self.ser = serial.Serial(DEVICE, 115200,timeout=0) # mini UART serial device

    @property
    def centermeters(self):
        if SIMULATE is not None:
            return 0
        centermeters = -1
        max_wait = 10
        while max_wait > 0:
            counter = self.ser.in_waiting # count the number of bytes of the serial port
            if counter > 8:
                bytes_serial = self.ser.read(9) # read 9 bytes
                self.ser.reset_input_buffer() # reset buffer
                if bytes_serial[0] == 0x59 and bytes_serial[1] == 0x59: # check first two bytes
                    centermeters = bytes_serial[2] + bytes_serial[3]*256 # distance in next two bytes
                    #strength = bytes_serial[4] + bytes_serial[5]*256 # signal strength in next two bytes
                    #temperature = bytes_serial[6] + bytes_serial[7]*256 # temp in next two bytes
                    #temperature = (temperature/8.0) - 256.0 # temp scaling and offset
                    #print(distance/100.0,strength,temperature)
                    break
            max_wait -= 1
            time.sleep(0.1)
        return centermeters

class Main:
    def __init__(self):
        # start GPS
        global TIME_SYNCHRONIZED
        global GPS_CONNECTED
        if USE_GPS and SIMULATE is None:
            T.start_new_thread(gps, ())
        else:
            TIME_SYNCHRONIZED = True

        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        self.alarm_distance_threshold_max = 500 # centimeter
        self.alarm_distance_threshold_min = 100 # centimeter

        CAMERA_VIEW_WIDTH = 640
        CAMERA_VIEW_HEIGHT = 480
        CAMERA_FPS = 10  # int(self.camera.get(cv2.CAP_PROP_FPS))

        while not TIME_SYNCHRONIZED:
            time.sleep(1)

        timestamp = datetime.datetime.now(pytz.timezone('US/Pacific')).strftime('%Y_%m_%d_%H_%M_%S')
        if SIMULATE is not None:
            self.camera = cv2.VideoCapture(SIMULATE)
        else:
            output_video_file = os.path.join(VIDEO_SAVE_PATH, f'{timestamp}.mp4')
            self.camera = cv2.VideoCapture(1)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_VIEW_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_VIEW_HEIGHT)
            # the camera is rotated clockwise 90 degrees. so when saving to file as (480, 640)
            self.video_writer = cv2.VideoWriter(output_video_file,
                                                cv2.VideoWriter_fourcc(*'mp4v'),  # XVID-> .avi, mp4v-> .mp4, FMP4-> .mp4
                                                CAMERA_FPS,
                                                (CAMERA_VIEW_HEIGHT, CAMERA_VIEW_WIDTH))

        #self.vehicle_detector = VehicleDetector()
        # self.distance_measure = DistanceMeasure()

        if SIMULATE is None:
            sound_initialized()

    def run(self):
        index = 0
        try:
            while not self.kill_now:
                start = time.time()
                ret, frame = self.camera.read()
                if not ret:
                    print("failed to grab frame")
                    break

                if SIMULATE is None:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                #is_danger = self.vehicle_detector.detect(frame)
                #distance_cm = self.distance_measure.centermeters
                #if is_danger and distance_cm < self.alarm_distance_threshold_max and distance_cm > self.alarm_distance_threshold_min:
                #if is_danger and PLAY_ALARM_SOUND:
                #    T.start_new_thread(sound_alarm(),())
                #    print('alarm')

                fps = 1 / (time.time() - start)

                """
                cv2.putText(frame, f'FPS: {fps:.2f}',
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                tm = datetime.datetime.now(pytz.timezone('US/Pacific')).strftime('%Y/%m/%d %H:%M:%S')
                cv2.putText(frame, f'Time: {tm}',
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if GPS["altitude"] is not None and GPS["speed"] is not None:
                    cv2.putText(frame, f'Alt/Spd: {GPS["altitude"]}/{int(GPS["speed"])}',
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if GPS["latitude"] is not None and GPS["longitude"] is not None:
                    lat = f'{GPS["latitude"]:.5f}'
                    lon = f'{GPS["longitude"]:.5f}'
                    cv2.putText(frame, f'Lat/Lon: {lat}/{lon}',
                            (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                """

                if SIMULATE is None:
                    self.video_writer.write(frame)
                else:
                    cv2.imshow("test", frame)
                    if cv2.waitKey(1) & 0xFF == 27: # ESC pressed
                        break
        finally:
            print('Release all resource')
            if SIMULATE is None:
                self.video_writer.release()
            self.camera.release()
            if SIMULATE is not None:
                cv2.destroyAllWindows()

    def exit_gracefully(self, *args):
        self.kill_now = True


def test_detector():
    #camera = cv2.VideoCapture('/Users/ysun5/Desktop/short.mov')
    camera = cv2.VideoCapture('/Users/ysun5/Desktop/2021_07_27_15_11_57.mp4')
    det = VehicleDetector()
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_AREA)
            count = det.detect(frame)
            cv2.imshow("test", frame)
            if cv2.waitKey(1) & 0xFF == 27: # ESC pressed
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

class test_GPS():
    def __init__(self):
        pass

    def run(self):
        T.start_new_thread(gps, ())
        while True:
            time.sleep(0.1)
            print(GPS)
            timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            print(timestamp)


if __name__ == '__main__':
    Main().run()
