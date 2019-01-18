import cv2
import numpy
import io
from PIL import Image

class VideoFeed:

    def __init__(self,mode=1,name="w1",capture=1):
        print(name)
        self.camera_index = 0
        self.name = name
        if capture == 1:
            self.cam = cv2.VideoCapture(self.camera_index)

# improve

    def red_pulse(self, frame, shape, frameCnt, interFrame, repeatLen):
        if (frameCnt % interFrame == 0) and (repeatLen > 0):
            frameEnc = 255 * numpy.ones(shape, dtype=numpy.uint8)
            frameEnc[:, :] = (255, 0, 0)  # red looks better than white
            newframe = frameEnc
        else:
            newframe = frame
        return newframe

    def get_frame(self, frameCnt, startflag):
        ret_val, img = self.cam.read()
        c = cv2.waitKey(1)
        if (c == "n"): #in "n" key is pressed while the popup window is in focus
            self.camera_index += 1 #try the next camera index
            self.cam = cv2.VideoCapture(self.camera_index)
            if not self.cam: #if the next camera index didn't work, reset to 0.
                self.camera_index = 0
                self.cam = cv2.VideoCapture(self.camera_index)

        #cv2.imshow('my webcam', img)
        cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        shape = cv2_im.shape
        if startflag:
            cv2_im = self.red_pulse(cv2_im, shape, frameCnt, 50, 1)
        pil_im = Image.fromarray(cv2_im)
        b = io.BytesIO()
        pil_im.save(b, 'jpeg')
        im_bytes = b.getvalue()
        return im_bytes

    def set_frame(self, frame_bytes, startflag, out):
        pil_bytes = io.BytesIO(frame_bytes)
        pil_image = Image.open(pil_bytes)
        cv_image = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
        if startflag:
            out.write(cv_image)
        cv2.imshow(self.name, cv_image)
        cv2.waitKey(1)
'''
if __name__=="__main__":
    vf = VideoFeed(1,"test",1)
    while 1:
        m = vf.get_frame()
        vf.set_frame(m)
'''
