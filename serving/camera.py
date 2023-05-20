import cv2
import os

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        filename =  './static/frame.jpg'
        cv2.imwrite(filename, image)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def get_heartbeat(self):
        image = cv2.imread('noise-green.jpg')
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()