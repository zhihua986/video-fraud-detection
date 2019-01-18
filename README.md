# video-fraud-detection


Increase the frame rate by socket multi-threading half-finished, need do something to the frame count, because frame count is different between two threads, delete sending frame count via socket.

Two programs in this zip file, one is for server. One is for client.


1. Replace the IP address of server if needed

2. Run server.py

3. Run client.py

4. Wait for the console printout and see if it shows the connection is success.

5. In server, press 'q' to start video fraud detection process, server will send challenges to client(one red pause every 50 frames in this code), client will display the read pause and send the frame back, server write frames to a mp4 file into the testVideo folder.

6. In server, press 'w' to stop the challenge sending and sampling period, then server will automatically start a thread to process the saved mp4 files frame by frame to detect the challenge

