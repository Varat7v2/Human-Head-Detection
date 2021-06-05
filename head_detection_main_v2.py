import sys
import time
import numpy as np
import tensorflow as tf
import cv2

from myFROZEN_GRAPH_HEAD import FROZEN_GRAPH_HEAD

PATH_TO_CKPT_HEAD = 'models/HEAD_DETECTION_300x300_ssd_mobilenetv2.pb'
head_detector = FROZEN_GRAPH_HEAD(PATH_TO_CKPT_HEAD)
TEST_VIDEO_NAME = 'london_street'

writeVideo_flag = True
output_filename = './{}.avi'.format(TEST_VIDEO_NAME)

if __name__ == "__main__":
    source = 'videos/{}.mp4'.format(TEST_VIDEO_NAME)
    # source = 0
    cap = cv2.VideoCapture(source)

    if writeVideo_flag:
        # w = int(cap.get(3))
        # h = int(cap.get(4))
        w = 1920
        h = 1080
        f = 24
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_filename, fourcc, f, (w, h))

    while True:
        t_start = time.time()
        ret, image = cap.read()

        if ret == 0:
            break

        im_height, im_width, im_channel = image.shape
        image = cv2.flip(image, 1)

        # Head-detection run model
        image, heads = head_detector.run(image, im_width, im_height)

        fps = 1 / (time.time() - t_start)
        cv2.putText(image, "FPS: {:.2f}".format(fps), (10, 30), 0, 5e-3 * 130, (0,0,255), 2)
        cv2.putText(image, "HEAD DETECTION - Project by Varat7v2 (@https://github.com/Varat7v2)", (int(im_width/2)+50, im_height-10), 0, 0.5, (255,255,255), 1)
        cv2.imshow("HEAD DETECTION USING FROZEN GRAPH", image)

        if writeVideo_flag:
            out.write(cv2.resize(image, (1920, 1080), interpolation = cv2.INTER_CUBIC))

        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    if writeVideo_flag:
        out.release()

    cap.release()

    cv2.destroyAllWindows()
