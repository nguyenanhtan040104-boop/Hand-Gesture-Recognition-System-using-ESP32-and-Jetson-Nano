#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import cvzone
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import pyfirmata  # Thêm thư viện để điều khiển Arduino

import cv2
import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import cvzone
from pynput.keyboard import Controller
import numpy as np
import mediapipe as mp
import keyboard
import time
import pyautogui

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

# Khởi tạo kết nối Arduino
comport = 'COM11'
board = pyfirmata.Arduino(comport)
led_pins = [8, 9, 10, 11, 12]
leds = [board.get_pin(f'd:{pin}:o') for pin in led_pins]

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1280)
    parser.add_argument("--height", help='cap height', type=int, default=720)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

class Button:
    def __init__(self, pos, text, size=[100, 100]):
        self.pos = pos
        self.size = size
        self.text = text

def drawAll(img, buttonList, device_mode):
    # Always draw the DEVICE/KEYBOARD button regardless of mode
    device_button = buttonList[-3]  # DEVICE/KEYBOARD button is third from the end
    x, y = device_button.pos
    w, h = device_button.size
    cvzone.cornerRect(img, (x, y, w, h), 20, rt=0)
    cv2.rectangle(img, device_button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
    
    # Display "KEYBOARD" when in device_mode, otherwise display "DEVICE"
    button_text = "KEYBOARD" if device_mode else "DEVICE"
    cv2.putText(img, button_text, (x + 20, y + 65),
                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    
    if device_mode:
        # Only draw MODE buttons when in device mode
        for button in buttonList[-2:]:  # Last two buttons (MODE1, MODE2)
            x, y = button.pos
            w, h = button.size
            cvzone.cornerRect(img, (x, y, w, h), 20, rt=0)
            cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
            cv2.putText(img, button.text, (x + 20, y + 65),
                      cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    
    if not device_mode:
        # Draw keyboard buttons only when not in device mode
        for button in buttonList[:-3]:  # All buttons except the last three (which are function buttons)
            x, y = button.pos
            w, h = button.size
            cvzone.cornerRect(img, (x, y, w, h), 20, rt=0)
            cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
            cv2.putText(img, button.text, (x + 20, y + 65),
                      cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    
    return img

def main():
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    mode = 0

    # Bàn phím ảo
    keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
            ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

    buttonList = []
    for i in range(len(keys)):
        for j, key in enumerate(keys[i]):
            buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

    # Thêm nút DEVICE và các nút chế độ
    device_button = Button([10, cap_height - 150], "DEVICE", [200, 100])
    mode1_button = Button([240, cap_height - 150], "MODE1", [200, 100])  # Increased X position
    mode2_button = Button([470, cap_height - 150], "MODE2", [200, 100])  # Further increased X position
    buttonList.extend([device_button, mode1_button, mode2_button])

    finalText = ""
    last_press_time = 0
    stable_frames = 0
    dragging_active = False
    device_mode = False  # Trạng thái chế độ điều khiển thiết bị
    device_control_mode = 1  # 1: D0-D4, 2: Số ngón tay
    allowed_keys = ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P",
                    "A", "S", "D", "F", "G", "H", "J", "K", "L", ";",
                    "Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # Vẽ bàn phím ảo và các nút chức năng
        debug_image = drawAll(debug_image, buttonList, device_mode)

        # Kiểm tra các nút chức năng có được nhấn không
        hands_data, _ = detector.findHands(debug_image, draw=False)
        if hands_data:
            for hand in hands_data:
                lmList = hand["lmList"]
                x, y = lmList[8][0], lmList[8][1]  # Đầu ngón trỏ
                
                # Kiểm tra nút DEVICE
                # Kiểm tra nút DEVICE/KEYBOARD
                dev_x, dev_y = device_button.pos
                dev_w, dev_h = device_button.size
                if dev_x < x < dev_x + dev_w and dev_y < y < dev_y + dev_h:
                    button_text = "KEYBOARD" if device_mode else "DEVICE"
                    cv2.rectangle(debug_image, (dev_x-5, dev_y-5), (dev_x + dev_w + 5, dev_y + dev_h + 5), 
                         (0, 255, 0), cv2.FILLED)
                    cv2.putText(debug_image, button_text, (dev_x + 20, dev_y + 65),
                              cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    length, _, _ = detector.findDistance(lmList[8][:2], lmList[12][:2])
                    if length < 40:
                        device_mode = not device_mode  # Chuyển đổi chế độ
                        sleep(0.3) # Tránh nhấn nhiều lần
                
                # Kiểm tra nút MODE1 (chỉ khi ở chế độ device)
                if device_mode:
                    mode1_x, mode1_y = mode1_button.pos
                    mode1_w, mode1_h = mode1_button.size
                    if mode1_x < x < mode1_x + mode1_w and mode1_y < y < mode1_y + mode1_h:
                        cv2.rectangle(debug_image, (mode1_x-5, mode1_y-5), (mode1_x + mode1_w + 5, mode1_y + mode1_h + 5), 
                             (0, 255, 0), cv2.FILLED)
                        cv2.putText(debug_image, mode1_button.text, (mode1_x + 20, mode1_y + 65),
                                  cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                        length, _, _ = detector.findDistance(lmList[8][:2], lmList[12][:2])
                        if length < 40:
                            device_control_mode = 1
                            sleep(0.3)
                    
                    # Kiểm tra nút MODE2 (chỉ khi ở chế độ device)
                    mode2_x, mode2_y = mode2_button.pos
                    mode2_w, mode2_h = mode2_button.size
                    if mode2_x < x < mode2_x + mode2_w and mode2_y < y < mode2_y + mode2_h:
                        cv2.rectangle(debug_image, (mode2_x-5, mode2_y-5), (mode2_x + mode2_w + 5, mode2_y + mode2_h + 5), 
                             (0, 255, 0), cv2.FILLED)
                        cv2.putText(debug_image, mode2_button.text, (mode2_x + 20, mode2_y + 65),
                                  cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                        length, _, _ = detector.findDistance(lmList[8][:2], lmList[12][:2])
                        if length < 40:
                            device_control_mode = 2
                            sleep(0.3)
        
        # Hiển thị trạng thái chế độ
        mode_text = "DEVICE MODE" if device_mode else "KEYBOARD MODE"
        control_mode_text = f"CTRL MODE: {device_control_mode} (1:D0-D4, 2:Fingers)" if device_mode else ""
        cv2.putText(debug_image, f"Mode: {mode_text}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_image, control_mode_text, (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(debug_image, f"LEDs: {[led.read() for led in leds]}", 
                    (10, cap_height - 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 255, 255), 2)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                hand_label = handedness.classification[0].label

                if device_mode:
                    # Chế độ điều khiển thiết bị
                    gesture = keypoint_classifier_labels[hand_sign_id]
                    
                    if device_control_mode == 1:
                        # Chế độ 1: Điều khiển LED bằng cử chỉ D0-D4
                        if gesture == "D0":
                            leds[0].write(1)  # Bật LED 1
                            leds[1].write(0)
                            leds[2].write(0)
                            leds[3].write(0)
                            leds[4].write(0)
                        elif gesture == "D1":
                            leds[0].write(0)
                            leds[1].write(1)  # Bật LED 2
                            leds[2].write(0)
                            leds[3].write(0)
                            leds[4].write(0)
                        elif gesture == "D2":
                            leds[0].write(0)
                            leds[1].write(0)
                            leds[2].write(1)  # Bật LED 3
                            leds[3].write(0)
                            leds[4].write(0)
                        elif gesture == "D3":
                            leds[0].write(0)
                            leds[1].write(0)
                            leds[2].write(0)
                            leds[3].write(1)  # Bật LED 4
                            leds[4].write(0)
                        elif gesture == "D4":
                            leds[0].write(0)
                            leds[1].write(0)
                            leds[2].write(0)
                            leds[3].write(0)
                            leds[4].write(1)  # Bật LED 5
                        else:
                            # Nếu không phải cử chỉ D0-D4, tắt tất cả LED
                            for led in leds:
                                led.write(0)
                    
                    elif device_control_mode == 2:
                        # Chế độ 2: Điều khiển LED bằng số ngón tay giơ lên
                        fingers = detector.fingersUp(hand)
                        num_fingers = sum(fingers)
                        
                        # Bật LED tương ứng với số ngón tay
                        for i in range(5):
                            leds[i].write(1 if i < num_fingers else 0)
                    
                    # Hiển thị trạng thái LED
                    cv2.putText(debug_image, f"LEDs: {[led.read() for led in leds]}", 
                                (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                (0, 255, 255), 2)
                
                elif hand_label == "Right":
                    # Xử lý cử chỉ tay phải (điều khiển chuột)
                    gesture = keypoint_classifier_labels[hand_sign_id]
                    if gesture == "Move_mouse":
                        move_mouse_from_hand_landmark(landmark_list, pyautogui.size().width, pyautogui.size().height, cap_width, cap_height)
                    elif gesture == "Left Click":
                        pyautogui.click(button='left')
                        print("Left Click")
                    elif gesture == "Right Click":
                        pyautogui.click(button='right')
                        print("Right Click")
                    elif gesture == "Hold":
                        move_mouse_from_hand_landmark(landmark_list, pyautogui.size().width, pyautogui.size().height, cap_width, cap_height)
                        pyautogui.mouseDown(button='left')
                        print("Dragging")
                        dragging_active = True
                    else:
                        if dragging_active and gesture != "Hold":
                            pyautogui.mouseUp(button='left')
                            dragging_active = False
                            print("Drop")
                
                elif hand_label == "Left" and not device_mode:
                    # Xử lý cử chỉ tay trái (bàn phím ảo) chỉ khi không ở chế độ device
                    for button in buttonList[:-3]:  # Bỏ qua các nút chức năng
                        x, y = button.pos
                        w, h = button.size

                        if x < landmark_list[8][0] < x + w and y < landmark_list[8][1] < y + h:
                            cv2.rectangle(debug_image, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                            cv2.putText(debug_image, button.text, (x + 20, y + 65),
                                        cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                            length, _, _ = detector.findDistance(landmark_list[8][:2], landmark_list[12][:2])

                            if length < 60:
                                stable_frames += 1
                            else:
                                stable_frames = 0

                            current_time = time.time()
                            if stable_frames > 5 and (current_time - last_press_time) > 0.5:
                                if button.text in allowed_keys:
                                    pyautogui.write(button.text)
                                    finalText += button.text
                                    last_press_time = current_time
                                    stable_frames = 0
                                    sleep(0.3)

                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Hiển thị chuỗi ký tự đã nhập (chỉ khi không ở chế độ device)
        if not device_mode:
            cv2.rectangle(debug_image, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)
            cv2.putText(debug_image, finalText, (60, 430),
                        cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

        cv.imshow('Hand Gesture Recognition', debug_image)

    # Đóng kết nối Arduino khi thoát
    for led in leds:
        led.write(0)
    board.exit()
    cap.release()
    cv.destroyAllWindows()

# ... (giữ nguyên các hàm phụ trợ khác không thay đổi)

def move_mouse_from_hand_landmark(landmark_list, screen_width, screen_height, cap_width, cap_height):
    index_finger_tip = landmark_list[8]
    mouse_x = np.interp(index_finger_tip[0], [0, cap_width], [0, screen_width])
    mouse_y = np.interp(index_finger_tip[1], [0, cap_height], [0, screen_height])
    pyautogui.moveTo(mouse_x, mouse_y)

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)
    return image

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)
    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

if __name__ == '__main__':
    main() 