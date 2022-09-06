import cv2
import imagezmq
import time
import os
import zmq
import threading

image_hub = imagezmq.ImageHub(open_port='tcp://127.0.0.1:5555', REQ_REP=False)
cv2.namedWindow('VIDEO', cv2.WINDOW_NORMAL)

start_time = time.time()*1000
fps = 15 
vid_path, vid_writer = [None] * 1, [None] * 1
frame_count = 0
snippet_count = int(time.time())
lock = threading.Lock()
concat_lock = threading.Lock()
# Video request 
context = zmq.Context()
#socket = context.socket(zmq.PULL)
socket = context.socket(zmq.REP)

socket.bind("tcp://*:5556")

def recorder_thread(img):
    global fps, frame_count, start_time, snippet_count
    lock.acquire()
    curr_time = time.time()*1000
    total_frame_count = len([file for file in os.listdir("./frame_dir/")])
    if curr_time - start_time >= 2000 and total_frame_count >= 15 :
        snippet_count = int(time.time())
        os.system(f"ffmpeg -framerate 7.5  -i ./frame_dir/frame%04d.jpg -vcodec mpeg4 -y ./clips/{snippet_count}.mp4")
        os.system("rm -rf ./frame_dir/*")
        start_time = curr_time
        frame_count = 0
        snippet_count += 1
        #lock.release()
    else:
        img_path = f"./frame_dir/frame{('0'*(4-len(str(frame_count))))+str(frame_count)}.jpg"
        #print(f"Dumping {img_path}")
        cv2.imwrite(img_path, img)
        #lock.release()
    # Clear 2min old clips
    clip_timestamp = sorted([int(os.path.splitext(File)[0]) for File in os.listdir("./clips/")])
    #print(f"clip_timestamp: {clip_timestamp}")
    for timestamp in clip_timestamp:
        #print(f"timestamp diff : {(timestamp - (curr_time//1000))}")
        if  ((curr_time//1000)-timestamp) >= 120:
            #print(f"reonving clip ./clips/{timestamp}.mp4")

            os.system(f"rm -rf ./clips/{timestamp}.mp4")
    lock.release()
    frame_count += 1
    return

def concat_clips(request_timestamp, emp_name):
    evidence_dir = "./attendance_evidence/"
    if not os.path.isdir(evidence_dir):
        os.mkdir(evidence_dir)
    # Concat 20 sec. evidence clips
    File_concat_list = open("mylist.txt", "w")
    clip_timestamps = sorted([int(os.path.splitext(File)[0]) for File in os.listdir("./clips/")], reverse=True)
    count = 0
    String_to_be_concat = ""
    for clip_timestamp in clip_timestamps:
        #if count > 0:
        #    File_concat_list.write("\n")
        if clip_timestamp <= request_timestamp and (request_timestamp - clip_timestamp)<=20:
            if count > 0:
                String_to_be_concat = "\n" + String_to_be_concat
                #File_concat_list.write("\n")
            String_to_be_concat = f"file ./clips/{clip_timestamp}.mp4" + String_to_be_concat
            #File_concat_list.write(f"file ./clips/{clip_timestamp}.mp4")
            count += 1
    File_concat_list.write(String_to_be_concat)
    File_concat_list.close()
    from datetime import datetime
    attendance_time = datetime.fromtimestamp(request_timestamp).strftime('%H-%M-%S')
    evidence_dir = evidence_dir + datetime.fromtimestamp(request_timestamp).strftime('%d-%m-%y')
    if not os.path.isdir(evidence_dir):
        os.mkdir(evidence_dir)
    os.system(f"ffmpeg -f concat -safe 0 -i mylist.txt -c copy {evidence_dir}/{emp_name}_{attendance_time}.mp4")
    return f"{evidence_dir}/{emp_name}_{attendance_time}.mp4" 


def request_thread():
    while True:
        import json
        request = socket.recv()
        message = json.loads(request)
        request_timestamp = int(message['timestamp'])
        empoloyee_name = message['EmpoloyeeName']
        concat_lock.acquire()
        start_time = int(time.time()) + 5
        eviedence_clip = concat_clips(request_timestamp,empoloyee_name)
        end_time = int(time.time())
        print(f"Time taken to concat clips: {end_time-start_time}sec.")
        concat_lock.release()
        print(f"Timestamp in Received request : {message['timestamp']} and timestamp type {type(message['timestamp'])}")
        send_json = message
        send_json["Evidence_snippets"] = eviedence_clip
        send_json_str = json.dumps(send_json)
        socket.send_string(send_json_str)
    



t1 = threading.Thread(target=request_thread, args=())
t1.start()    
while True:  # show streamed images until Ctrl-C
    host_name, image = image_hub.recv_image()
    #print("Creating Recording Thread!!")
    t = threading.Thread(target=recorder_thread, args=(image,))
    t.start()
    #recoreder_thread(image)
    cv2.imshow('VIDEO', image) # 1 window for each RPi
    cv2.waitKey(1)
