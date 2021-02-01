import cv2
import numpy as np
cap = cv2.VideoCapture("test.mp4")


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'xvid')
out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
out1 = cv2.VideoWriter('output1.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
out2 = cv2.VideoWriter('difference.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
frame1_print = frame1

while(1):
    ret, frame2 = cap.read()
    frame2_print = frame2
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',prvs)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    else:
        out1.write(frame1_print)
        out.write(rgb)
        out2.write(frame2_print-frame1_print)
    prvs = next
    frame1_print = frame2_print
cap.release()
cv2.destroyAllWindows()