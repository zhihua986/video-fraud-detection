import socket
import time

class videosocket:
    '''A special type of socket to handle the sending and receiveing of fixed
       size frame strings over ususal sockets
       Size of a packet or whatever is assumed to be less than 100MB
    '''

    def __init__(self , sock=None):
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock= sock

    def connect(self,host,port):
        self.sock.connect((host,port))

    def vsend(self, framestring):

        t1 = time.clock()
        totalsent = 0
        metasent = 0
        length =len(framestring)
        lengthstr=str(length).zfill(8)

        while metasent < 8 :
            sent = self.sock.send(lengthstr[metasent:].encode('utf-8'))   # remove ".encode('utf-8')" when running client at another computer.
            if sent == 0:
                raise RuntimeError("Socket connection broken")
            metasent += sent
        
        
        while totalsent < length :
            sent = self.sock.send(framestring[totalsent:])
            if sent == 0:
                raise RuntimeError("Socket connection broken")
            totalsent += sent
        t2 = time.clock()
        #print('inside send time******* ' + str(t2 - t1))

    def vreceive(self):

        totrec=0
        metarec=0
        msgArray = []
        metaArray = []
        t11 = time.clock()
        while metarec < 8:
            chunk = self.sock.recv(8 - metarec)
            if chunk == '':
                raise RuntimeError("Socket connection broken")
            metaArray.append(chunk)
            metarec += len(chunk)
        # lengthstr= ''.join(metaArray)
        t22 = time.clock()
        lengthstr= b''.join(metaArray) 
        length=int(lengthstr)

        while totrec<length :
            chunk = self.sock.recv(length - totrec)
            if chunk == '':
                raise RuntimeError("Socket connection broken")
            msgArray.append(chunk)
            totrec += len(chunk)
        # return ''.join(msgArray)

        dt = (t22 - t11)
        #print('inside receive time--------- ' + str(dt))
        return b''.join(msgArray)

