import RPi.GPIO as GPIO
import cv2
import ipywidgets.widgets as widgets
import time
import numpy as np
import copy
import demjson
from aip import AipBodyAnalysis


class CarDetect(object):
    """网球检测类，负责通过图像处理检测白色网球位置"""

    def __init__(self):
        self.lower_range = np.array([0, 0, 220])
        self.higher_range = np.array([180, 30, 255])
        self.cut_edge = 0

    def tennis_detect(self, img, video_return=True):
        x_pos, y_pos, radius = 0, 0, 0
        copy.copy(img)
        img = img[self.cut_edge:img.shape[0], :, :]
        img = cv2.resize(img, (320, 240))
        img = cv2.blur(img, (3, 3))
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 30,
                                   param1=80, param2=25, minRadius=8, maxRadius=40)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            x = circles[:, 0]
            y = circles[:, 1]
            r = circles[:, 2]
            s_r = (r / 1.5).astype(int)
            num = circles.shape[0]
            rate = np.zeros(num)
            for i in range(num):
                detect_area = hsv_img[y[i] - s_r[i]:y[i] + s_r[i], x[i] - s_r[i]:x[i] + s_r[i]]
                if detect_area.size != 0:
                    tennis_color_mask = cv2.inRange(detect_area, self.lower_range, self.higher_range)
                    num_point = np.sum(tennis_color_mask / 255, dtype = np.int64)
                    rate[i] = num_point / (detect_area.shape[0] * detect_area.shape[1])
            i = np.argmax(rate)
            if rate[i] > 0.3:
                x_pos, y_pos, radius = x[i], y[i], r[i]
                x_pos = int(x_pos * 2)
                y_pos = int(y_pos * 2)
                radius = int(radius * 2)
        if video_return:
            return x_pos, y_pos, radius
        return x_pos, y_pos, radius


class CarMove(object):
    """小车运动控制类，负责控制电机运动"""

    def __init__(self):
        self.IN1 = 20
        self.IN2 = 21
        self.IN3 = 19
        self.IN4 = 26
        self.ENA = 16
        self.ENB = 13
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.ENA, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.IN1, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN2, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.ENB, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.IN3, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN4, GPIO.OUT, initial=GPIO.LOW)
        self.pwm_ENA = GPIO.PWM(self.ENA, 2000)
        self.pwm_ENB = GPIO.PWM(self.ENB, 2000)
        self.pwm_ENA.start(0)
        self.pwm_ENB.start(0)

    def forward(self, delaytime):
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm_ENA.ChangeDutyCycle(15)
        self.pwm_ENB.ChangeDutyCycle(15)
        time.sleep(delaytime)

    def back(self, delaytime):
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.HIGH)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.HIGH)
        self.pwm_ENA.ChangeDutyCycle(15)
        self.pwm_ENB.ChangeDutyCycle(15)
        time.sleep(delaytime)

    def left(self, delaytime):
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.HIGH)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm_ENA.ChangeDutyCycle(12)
        self.pwm_ENB.ChangeDutyCycle(12)
        time.sleep(delaytime)

    def right(self, delaytime):
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.HIGH)
        self.pwm_ENA.ChangeDutyCycle(12)
        self.pwm_ENB.ChangeDutyCycle(12)
        time.sleep(delaytime)

    def brake(self, delaytime):
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm_ENA.ChangeDutyCycle(0)
        self.pwm_ENB.ChangeDutyCycle(0)
        time.sleep(delaytime)


class Car(CarMove, CarDetect):
    """综合控制类，继承运动和检测功能"""

    def __init__(self):
        CarMove.__init__(self)
        CarDetect.__init__(self)
        self.TrigPin = 1
        self.EchoPin = 0
        GPIO.setup(self.TrigPin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.EchoPin, GPIO.IN)

    def cleanup(self):
        self.brake(0.1)
        self.pwm_ENA.stop()
        self.pwm_ENB.stop()
        GPIO.cleanup()

    def distance(self):
        """测量超声波距离，优化超时处理"""
        GPIO.output(self.TrigPin, GPIO.LOW)
        time.sleep(0.000002)
        GPIO.output(self.TrigPin, GPIO.HIGH)
        time.sleep(0.000015)
        GPIO.output(self.TrigPin, GPIO.LOW)

        start = time.time()
        while GPIO.input(self.EchoPin) == 0:
            if time.time() - start > 0.03:
                return -1

        start = time.time()
        while GPIO.input(self.EchoPin) == 1:
            if time.time() - start > 0.03:
                return -1

        stop = time.time()
        distance = (stop - start) * 34000 / 2
        return distance

    def distance_test(self):
        """优化超声波测量，3次取中值，减少时延"""
        ultrasonic = []
        for _ in range(3):  # 减少到 3 次测量
            for _ in range(2):  # 最多重试 2 次
                distance = self.distance()
                if 0 < distance < 500:
                    ultrasonic.append(distance)
                    break
            else:
                ultrasonic.append(-1)  # 如果重试失败，记录无效值
            time.sleep(0.005)  # 缩短延时
        ultrasonic = sorted(ultrasonic)  # 使用内置排序
        distance = ultrasonic[1] if ultrasonic[1] != -1 else 1000  # 取中值，无效时返回安全值
        print(f"Distance: {distance:.2f} cm")
        return distance


def bgr8_to_jpeg(value):
    return bytes(cv2.imencode('.jpg', value)[1])


hand = {'One': '数字1', 'Five': '数字5', 'Fist': '拳头', 'Ok': 'OK',
        'Prayer': '祈祷', 'Congratulation': '作揖', 'Honour': '作别',
        'Heart_single': '比心心', 'Thumb_up': '点赞', 'Thumb_down': 'Diss',
        'ILY': '我爱你', 'Palm_up': '掌心向上', 'Heart_1': '双手比心1',
        'Heart_2': '双手比心2', 'Heart_3': '双手比心3', 'Two': '数字2',
        'Three': '数字3', 'Four': '数字4', 'Six': '数字6', 'Seven': '数字7',
        'Eight': '数字8', 'Nine': '数字9', 'Rock': 'Rock', 'Insult': '竖中指', 'Face': '脸'}

APP_ID = ''
API_KEY = ''
SECRET_KEY = ''

# camera = PiCamera()
client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)

g_camera = cv2.VideoCapture('http://192.168.50.1:8080/?action=stream')
g_camera.set(3, 640)
g_camera.set(4, 480)
g_camera.set(5, 120)  # 设置帧率
g_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
g_camera.set(cv2.CAP_PROP_BRIGHTNESS, 40)  # 设置亮度 -64 - 64  0.0
g_camera.set(cv2.CAP_PROP_CONTRAST, 50)  # 设置对比度 -64 - 64  2.0
g_camera.set(cv2.CAP_PROP_EXPOSURE, 156)  # 设置曝光值 1.0 - 5000  156.0

ret, frame = g_camera.read()
image_widget = widgets.Image(format='jpeg', width=600, height=500)  # 设置摄像头显示组件
image_widget.value = bgr8_to_jpeg(frame)


def test_motors(car):
    print("测试前进...")
    car.forward(1)
    car.brake(0.5)
    print("测试左转...")
    car.left(1)
    car.brake(0.5)
    print("测试右转...")
    car.right(1)
    car.brake(0.5)
    print("电机测试完成")

zydj = 11
sxdj = 9
GPIO.setup(zydj, GPIO.OUT)
GPIO.setup(sxdj, GPIO.OUT)
def servo_pulse(sp,myangle):

    pulsewidth = (myangle * 11) + 500

    GPIO.output(sp, GPIO.HIGH)

    time.sleep(pulsewidth/1000000.0)

    GPIO.output(sp, GPIO.LOW)

    time.sleep(20.0/1000-pulsewidth/1000000.0)

def main():
    print("程序启动")
    try:
        show_display = False
        distance = 30
        image = cv2.VideoCapture('http://192.168.50.1:8080/?action=stream')
        if not image.isOpened():
            print("无法打开摄像头")
            return
        print("摄像头已打开")
        image.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        image.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        image.set(cv2.CAP_PROP_FPS, 30)
        image.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        car = Car()

        radius_mov_ave = 35
        x_mov_ave = 320
        last_action_time = time.time()
        action_interval = 0.05
        last_print_time = time.time()
        F = True
        while F:
            """1.拍照 """
            g_camera.read()

            g_camera.read()

            # image = get_file_content('./image.jpg')

            """ 2.调用手势识别 """
            raw = str(client.gesture(image_widget.value))
            text = demjson.decode(raw)
            try:
                res = text['result'][0]['classname']
            except IndexError:
                print('识别结果：什么也没识别到哦~')
                # 1 dst 2 文字内容 3 坐标 4 5 字体大小 6 color 7 粗细 8 line type
                #       cv2.putText(frame, '未识别', (250,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,200), 2, cv2.LINE_AA) #只能显示英文
            else:
                print('识别结果：' + hand[res])
                F = False
                # cv2.putText(frame, hand[res], (250,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

            ret, frame = g_camera.read()
            image_widget.value = bgr8_to_jpeg(frame)
        while True:
            ret, frame = image.read()
            distance = 0.3 * car.distance_test() + 0.7 * distance  # 提高新数据权重
            if not ret:
                print("无法捕获图像帧")
                time.sleep(0.01)
                continue

            x, y, r = car.tennis_detect(frame, video_return=False)

            if show_display:
                cv2.imshow('Camera', frame)

            if r != 0:
                radius_mov_ave = 0.1 * r + 0.9 * radius_mov_ave
                x_mov_ave = 0.4 * x + 0.6 * x_mov_ave

            current_time = time.time()
            if current_time - last_action_time >= action_interval:

                ForB = 'Brake' if r == 0 or radius_mov_ave < 15 or radius_mov_ave > 50 else 'Forward'
                LorR = 'Brake' if x == 0 else 'Right' if x_mov_ave > 440 else 'Left' if x_mov_ave < 200 else 'Brake'
                if distance < 20:
                    ForB = 'Brake'
                    LorR = 'Brake'

                if current_time - last_print_time >= 1.0:
                    print(
                        f"检测: x={x}, r={r}, x_mov_ave={x_mov_ave:.1f}, radius_mov_ave={radius_mov_ave:.1f}, ForB={ForB}, LorR={LorR}, Distance={distance:.2f} cm")
                    last_print_time = current_time

                if ForB == 'Brake':
                    if LorR == 'Left':
                        car.left(0.05)
                    elif LorR == 'Right':
                        car.right(0.05)
                    else:
                        car.brake(0.02)
                        car.right(0.05)

                elif ForB == 'Forward':
                    if LorR == 'Left':
                        car.left(0.05)
                    elif LorR == 'Right':
                        car.right(0.05)
                    else:
                        car.forward(0.05)

                last_action_time = current_time

            time.sleep(0.005)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        car.cleanup()
        image.release()

    except KeyboardInterrupt:
        print("程序被用户终止")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()