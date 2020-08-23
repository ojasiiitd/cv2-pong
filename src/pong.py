import time
import numpy as np
import cv2
from PIL import ImageGrab

MAX_FRAMES = 700

def windowMovingFunc(winName="Original"):
    test_img = np.zeros(shape=(600,600,3)).astype('uint8')
    cv2.imshow(winName,test_img)
    cv2.moveWindow(winName,1250,100)
    cv2.waitKey(1)

def get_screen(PONG_BOX = (100,220,835,720)):
    screen = ImageGrab.grab(bbox=PONG_BOX)
    frame = np.array(screen)
    frame = cv2.cvtColor(frame , cv2.COLOR_RGB2BGR)
    blur = cv2.blur(frame , (20,20)) # blurred for better edge detection
    return frame

def movingEdges(frame):
    ballMask = fgbg.apply(frame) # getting moving portions i.e. the ball
    movingFrame = cv2.bitwise_and(frame , frame , mask=ballMask) # mask = ballMask for applying ballMask
    edges = cv2.Canny(movingFrame , 10 , 10)
    return edges

def testEdges(frame , edges):
    indices = np.where(edges != [0])
    coordinates = zip(indices[1] , indices[0]) # getting all edge coordinates
    for pt in coordinates:
        cv2.circle(frame , pt , 3 , (0,255,0) , -1) # to track the ball

def getYcoor(edges):
    indices = np.where(edges[: , :725] != [0])
    coordinates = zip(indices[1] , indices[0]) # getting all edge coordinates
    y = []
    for i,pt in enumerate(coordinates):
        y.append(pt[1])
    avgCoor = y[0]
    if len(y) > 0:
        avgCoor = sum(y)//len(y)
    return avgCoor

if __name__ == "__main__":

    windowMovingFunc()

    fgbg = cv2.createBackgroundSubtractorKNN(history=30) # history is low as we have to track only the ball and not other dynamic things like the score, paddles, etc

    frame_count = 0

    while True:
        start = time.time()
        
        frame = get_screen()
        
        edges = movingEdges(frame)

        # testEdges(frame , edges)

        ball_Y = getYcoor(edges)

        # cv2.imshow("Original" , frame)
        cv2.imshow("Edges" , edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        if frame_count >= MAX_FRAMES:
            break
        print(frame_count , "fps: " , time.time() - start)

    cv2.destroyAllWindows()