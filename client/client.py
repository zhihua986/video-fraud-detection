import socket, videosocket
import io
from videofeed import VideoFeed
import sys
import time
import threading
import multiprocessing as mp
import cv2
import queue
#import numpy as np

myqueue = queue.Queue(1)

class Client:
    def __init__(self, ip_addr = "169.254.162.107"):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((ip_addr, 6000))
        self.vsock = videosocket.videosocket (self.client_socket)
        self.videofeed = VideoFeed(1,"client",1)
        self.data = io.StringIO()


    def mysend(self):
        reFrameCnt = 0
        fps = 0
        tt = 0
        while True:
            reFrameCnt = reFrameCnt + 1
            if reFrameCnt % 10 == 0:

                t1 = time.time()
                fps = int(10/(t1 - tt))
                tt = t1
                print("send fps = ", fps)
            #print(1111)
            frame1 = self.videofeed.get_frame()
            self.vsock.vsend(frame1)


    def myreceive(self):
        reFrameCnt = 0
        fps = 0
        tt = 0
        while True:
            reFrameCnt = reFrameCnt + 1
            if reFrameCnt % 10 == 0:

                t1 = time.time()
                fps = int(10/(t1 - tt))
                tt = t1
                print("receive fps = ", fps)
            frame = self.vsock.vreceive()
            my_img = self.videofeed.set_frame(frame)
            my_img = cv2.resize(my_img, (1280, 960), interpolation=cv2.INTER_AREA)
            #cv2.putText(my_img, str(fps), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.imshow("aaa", my_img)
            cv2.waitKey(1)
            #myqueue.put(my_img)
            #A = show_img
            #print(2222)




    '''
    def connect(self):

        #thread_list = []
        
        while True:

            
            tget1 = time.process_time()
            frame=self.videofeed.get_frame()
            tget2 = time.process_time()
            print('get frame time ' + str(tget2 - tget1))

            tsend1 = time.process_time()
            self.vsock.vsend(frame)
            tsend2 = time.process_time()
            print('send time ' + str(tsend2 - tsend1))

            tre1 = time.process_time()
            frame = self.vsock.vreceive()
            tre2 = time.process_time()
            print('receive time ' + str(tre2 - tre1))

            tset1 = time.process_time()
            self.videofeed.set_frame(frame)
            tset2 = time.process_time()
            print('set frame time ' + str(tset2 - tset1))
    '''

if __name__ == "__main__":
    #runtime.LockOSThread()
    ip_addr = "169.254.162.107"
    if len(sys.argv) == 2:
        ip_addr = sys.argv[1]

    print ("Connecting to " + ip_addr + "....")

    #cv2.namedWindow("aaa")
    #cv2.waitKey(1)
    #cv2.resizeWindow("aaa", 480, 640)
    client = Client(ip_addr)
    #client.connect()
    #show_img = np.zeros((480, 640, 3), np.uint8)
    #cv2.imshow("aaa", show_img)

    #cv2.waitKey(1)

    th1 = threading.Thread(target=client.mysend)

    th2 = threading.Thread(target=client.myreceive)

    th1.start()
    th2.start()
    '''
    while True:
        #while not myqueue.empty():
            item = myqueue.get()
            cv2.imshow("aaa", item)
            cv2.waitKey(1)
    '''

    th1.join()
    th2.join()




