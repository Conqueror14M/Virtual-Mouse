import cv2
import numpy as np
import HandTrackerModule as htm
import time
import pyautogui

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.95)
wScr, hScr = pyautogui.size()

while True:
	success, img = cap.read()

	img = detector.findHands(img)
	lmList, bbox = detector.findPosi(img)
	if len(lmList) != 0:
		x1, y1 = lmList[8][0:]
		x2, y2 = lmList[12][0:]

		fingers = detector.fingersUp()
		
		if fingers[1]==1 and fingers[2]==0:

			x3 = np.interp(x1, (0, wCam), (0, wScr))
			y3 = np.interp(y1, (0, hCam), (0, hScr))

			pyautogui.moveTo(wScr-x3, y3)
			cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)


	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime
	cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
	cv2.imshow("Image", img)
	cv2.waitKey(1)