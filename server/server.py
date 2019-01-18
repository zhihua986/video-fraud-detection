import socket, videosocket
from videofeed import VideoFeed
import time
import threading
from detect_faces_video_process import MyClass
import multiprocessing as mp
import cv2
import keyboard
import queue

hahaha = queue.Queue(1)
startflag = False
wflag = True
qflag = True
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('testVideo\\' + '0' + '.mp4', fourcc, 20.0, (640, 480))

class Server:
    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("", 6000))
        self.server_socket.listen(5)
        self.videofeed = VideoFeed(1,"server",1)
        print("TCPServer Waiting for client on port 6000")

    def keyBoard(self):
        global out
        global qflag
        global wflag
        global startflag
        while True:
            if keyboard.is_pressed('q') and qflag:
                print("q is pressed")
                timestr = time.strftime('%Y%m%d-%H%M%S')
                out = cv2.VideoWriter('testVideo\\' + timestr + '.mp4', fourcc, 20.0, (640, 480))
                startflag = True
                #print("frame count start write: ", +frameCnt)
                qflag = False
            if keyboard.is_pressed('w') and wflag:
                out.release()
                print("w is pressed")
                classface = threading.Thread(target=MyClass.my_main, args=(timestr, hahaha,))
                classface.start()
                startflag = False
                wflag = False
            if not hahaha.empty():
                hahaha.get()
                qflag = True
                wflag = True
                print("finish")

    def mysend(self, vsock):
        global out
        global qflag
        global wflag
        global startflag
        sendCnt = 0
        while True:
            sendCnt = sendCnt + 1
            frame1 = self.videofeed.get_frame(sendCnt, startflag)
            vsock.vsend(frame1)

            if keyboard.is_pressed('q') and qflag:
                print("q is pressed")
                timestr = time.strftime('%Y%m%d-%H%M%S')
                out = cv2.VideoWriter('testVideo\\' + timestr + '.mp4', fourcc, 30.0, (640, 480))
                startflag = True
                #print("frame count start write: ", +frameCnt)
                qflag = False
            if keyboard.is_pressed('w') and wflag:
                out.release()
                print("w is pressed")
                classface = threading.Thread(target=MyClass.my_main, args=(timestr, hahaha,))
                classface.start()
                startflag = False
                wflag = False
            if not hahaha.empty():
                hahaha.get()
                qflag = True
                wflag = True
                print("finish")

            #print(1111)

    def myreceive(self, vsock):
        while True:
            #print(2222)
            frame = vsock.vreceive()
            self.videofeed.set_frame(frame, startflag, out)
if __name__ == "__main__":
    server = Server()
    client_socket, address = server.server_socket.accept()
    print("I got a connection from ", address)
    vsock = videosocket.videosocket(client_socket)
    t1 = threading.Thread(target=server.mysend, args=[vsock])
    t2 = threading.Thread(target=server.myreceive, args=[vsock])
    t3 = threading.Thread(target=server.keyBoard)
    t2.start()
    t1.start()
    #t3.start()

    #t3.join()
    t2.join()
    t1.join()
