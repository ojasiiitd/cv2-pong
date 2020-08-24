import time
import pyKey
import numpy as np
import cv2
from PIL import ImageGrab

MAX_FRAMES = 550
PADDLE_Y = [245 , 255]

def windowMovingFunc(winName="Original"):
    test_img = np.zeros(shape=(600,600,3)).astype('uint8')
    cv2.imshow(winName,test_img)
    cv2.moveWindow(winName,1250,100)
    cv2.waitKey(1)

def get_screen(PONG_BOX = (100,220,845,720)):
    screen = ImageGrab.grab(bbox=PONG_BOX)
    frame = np.array(screen)
    frame = cv2.cvtColor(frame , cv2.COLOR_RGB2GRAY)
    # frame = cv2.cvtColor(frame , cv2.COLOR_RGB2BGR)
    return frame

def movingEdges(frame):
    ballMask = fgbg.apply(frame) # getting moving portions i.e. the ball
    movingFrame = cv2.bitwise_and(frame , frame , mask=ballMask) # mask = ballMask for applying ballMask
    edges = cv2.Canny(movingFrame , 100 , 100)
    return edges

def ballYcoor(edges):
    indices = np.where(edges[40: , :725] != [0])
    coordinates = zip(indices[1] , indices[0]) # getting all edge coordinates
    y = []
    for i,pt in enumerate(coordinates):
        y.append(pt[1]+40)
        break
    if len(y) == 0:
        return 250
    return y[0]

def paddleYcoor(edges):
    indices = np.where(edges[: , 720:] != [0])
    coordinates = zip(indices[1] , indices[0]) # getting all edge coordinates
    y = []
    for i,pt in enumerate(coordinates):
        y.append(pt[1])
    if len(y) == 0:
        return -1
    return [min(y)+5 , max(y)-5]

def control(ball , paddle) :
    if ball in range(paddle[0] , paddle[1]):
        pyKey.releaseKey("UP")
        pyKey.releaseKey("DOWN")
    elif ball > paddle[1]:
        pyKey.releaseKey("UP")
        pyKey.pressKey("DOWN")
    elif ball < paddle[0]:
        pyKey.releaseKey("DOWN")
        pyKey.pressKey("UP")

if __name__ == "__main__":

    # windowMovingFunc()

    fgbg = cv2.createBackgroundSubtractorKNN(history=40) # history is low as we have to track only the ball and not other dynamic things like the score, paddles, etc

    frame_count = 0

    while True:
        start = time.time()
        
        frame = get_screen()
        
        edges = movingEdges(frame)

        ball_Y = ballYcoor(edges)
        if len(PADDLE_Y) > 1000:
            PADDLE_Y = [245,255]
        if paddleYcoor(edges) != -1:
            PADDLE_Y.append(paddleYcoor(edges))
        paddle_Y = PADDLE_Y[-1]

        # show Y coordinates
        cv2.circle(frame , (350 , ball_Y) , 4 , (0,255,0) , -1)
        # cv2.circle(frame , (720 , paddle_Y[0]) , 4 , (0,0,255) , -1)
        # cv2.circle(frame , (720 , paddle_Y[1]) , 4 , (0,0,255) , -1)

        control(ball_Y , paddle_Y)

        # cv2.imshow("Original" , frame)
        # cv2.imshow("Edges" , edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        # if frame_count >= MAX_FRAMES:
        #     pyKey.releaseKey("UP")
        #     pyKey.releaseKey("DOWN")
        #     break
        print(frame_count , "fps: " , time.time() - start)

    cv2.destroyAllWindows()