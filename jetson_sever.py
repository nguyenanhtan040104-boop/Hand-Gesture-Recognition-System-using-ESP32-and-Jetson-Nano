# code tren jetson nano dung de ket noi giua jetson va may tinh 
import socket
import Jetson.GPIO as GPIO
import time

HOST = '0.0.0.0' 
PORT = 65432      
LED_PINS = [12, 16, 18, 22, 26]  

try:
    GPIO.setmode(GPIO.BOARD) 
    for pin in LED_PINS:
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
    print(">>> [Server] GPIO da duoc thiet lap.")
except Exception as e:
    print(f">>> [Server] L?i thiet lap GPIO: {e}")
    print(">>> Vui lang chay voi sudo: sudo python3 jetson_gpio_server.py")
    exit()

def control_leds_by_gesture(gesture):
    """dieu khien LED dua tren cu chi D0-D4"""
  
    for pin in LED_PINS:
        GPIO.output(pin, GPIO.LOW)
    
    try:
        led_index = int(gesture[1:])
        if 0 <= led_index < len(LED_PINS):
            GPIO.output(LED_PINS[led_index], GPIO.HIGH)
            print(f">>> [Server] so bit LED tai chan {LED_PINS[led_index]} theo cu chi {gesture}")
    except (ValueError, IndexError):
        pass

def control_leds_by_finger_count(num_fingers):
    """dieu khien LED dua tren so ngon tay"""
    print(f">>> [Server] Bit {num_fingers} LED.")
    for i, pin in enumerate(LED_PINS):
        if i < num_fingers:
            GPIO.output(pin, GPIO.HIGH)
        else:
            GPIO.output(pin, GPIO.LOW)


print(">>> [Server] dang khoi tao...")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    print(f">>> [Server] dang lang nghe toi dia chi {HOST} tren cong {PORT}")
    
    while True: 
        s.listen()
        conn, addr = s.accept()
        with conn:
            print(f">>> [Server] da co ket noi  {addr}")
            try:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        print(f">>> [Server] Client {addr} da ngat ket noi.")
                        break
                    
                    command_str = data.decode('utf-8').strip()
                    print(f">>> [Server] Nhan duoc lenh: '{command_str}'")
                    
         
                    parts = command_str.split(':')
                    cmd_type = parts[0]
                    
                    if cmd_type == "GESTURE":
                        control_leds_by_gesture(parts[1])
                    elif cmd_type == "FINGERS":
                        num_fingers = int(parts[1])
                        control_leds_by_finger_count(num_fingers)
                    elif cmd_type == "STATE" and parts[1] == "OFF":
                        for pin in LED_PINS:
                            GPIO.output(pin, GPIO.LOW)

            except ConnectionResetError:
                print(f">>> [Server] Client {addr} da ngat ket noi .")
            finally:
                print(">>> [Server]  ket noi moi...")
                for pin in LED_PINS:
                    GPIO.output(pin, GPIO.LOW)


print(">>> [Server] D?n d?p GPIO.")
GPIO.cleanup()