import time
import numpy as np
import cv2
import pyautogui
from PIL import ImageGrab

def movingFunc():
    test_img = np.zeros(shape=(600,600,3)).astype('uint8')
    cv2.imshow('Original',test_img)
    cv2.moveWindow('Original',1250,100)
    cv2.waitKey(1)

def get_screen(PONG_BOX = (115,285,775,675)):
    screen = ImageGrab.grab(bbox=PONG_BOX)
    frame = np.array(screen)
    frame = cv2.cvtColor(frame , cv2.COLOR_RGB2GRAY)
    blur = cv2.blur(frame , (10,10)) # blurred for better edge detection
    return frame[15: , 20:]

if __name__ == "__main__":

    # movingFunc()

    fgbg = cv2.createBackgroundSubtractorKNN(history=10) # history is low as we have to track only the ball and not other dynamic things like the score, paddles, etc

    while True:
        start = time.time()
        
        frame = get_screen()

        ballMask = fgbg.apply(frame) # getting moving portions i.e. the ball
        movingFrame = cv2.bitwise_and(frame , frame , mask=ballMask)
        edges = cv2.Canny(movingFrame , 10 , 10)

        indices = np.where(edges != [0])
        coordinates = zip(indices[1] , indices[0]) # getting all edge coordinates\
        for pt in coordinates:
            # cv2.circle(frame , pt , 3 , (255,255,255) , -1) # to track the ball
            ball_Y = pt[1] # only the y-coordinate of the ball
            break

        pyautogui.dragTo(800, (285+20)+ball_Y , button="left")

        # cv2.imshow("Original" , frame)
        # cv2.imshow("Eges" , edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print("fps: " , time.time() - start)

    cv2.destroyAllWindows()