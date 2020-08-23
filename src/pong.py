import time
import numpy as np
import cv2
import pyautogui as pygui
from PIL import ImageGrab

def get_screen(PONG_BOX):
    screen = ImageGrab.grab(bbox=PONG_BOX)
    frame = np.array(screen)
    frame = cv2.cvtColor(frame , cv2.COLOR_RGB2GRAY)
    return frame[15: , 20:]

if __name__ == "__main__":
    fgbg = cv2.createBackgroundSubtractorKNN(history=20)

    while True:
        start = time.time()
        
        PONG_BOX = (115,285,775,675)
        frame = get_screen(PONG_BOX)

        ballMask = fgbg.apply(frame)
        movingFrame = cv2.bitwise_and(frame , frame , mask=ballMask)
        edges = cv2.Canny(movingFrame , 10 , 10)

        indices = np.where(edges != [0])
        coordinates = zip(indices[0], indices[1])
        for pts in coordinates:
            print(pts)
            cv2.circle(frame , pts , 2 , (0,255,0) , -1)

        cv2.imshow("Orgiginal" , frame)
        # cv2.imshow("Eges" , edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print("fps: " , time.time() - start)

    cv2.destroyAllWindows()