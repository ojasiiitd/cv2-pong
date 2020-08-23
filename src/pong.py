import time
import numpy as np
import cv2
import pyautogui as pygui
from PIL import ImageGrab

def get_screen():
    PONG_BOX = (115,285,785,685)

    while True:
        start = time.time()

        screen = ImageGrab.grab(bbox=PONG_BOX)
        frame = np.array(screen)
        frame = cv2.cvtColor(frame , cv2.COLOR_RGB2BGR)

        cv2.imshow("Screen" , frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print("fps: " , time.time() - start)

    cv2.destroyAllWindows()

get_screen()