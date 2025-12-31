import cv2
import tkinter as tk
from tkinter import Frame, Label, Button, Text, Scrollbar
from PIL import Image, ImageTk
from google.cloud import vision
import pyttsx3
import queue
import numpy as np
import socket

# ================= INTERNET CHECK =================
def internet_available(timeout=2):
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname("google.com")
        return True
    except:
        return False

# ================= OFFLINE MODELS =================
from ultralytics import YOLO
import easyocr

yolo_model = YOLO("yolov8n.pt")
ocr_reader = easyocr.Reader(['en'], gpu=False)

# ================= GOOGLE VISION =================
vision_client = vision.ImageAnnotatorClient()

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ================= APP STATE =================
paused = False
last_frame = None
MODE = "AUTO"

# ================= TKINTER =================
root = tk.Tk()
root.title("Vision Capture")

# IMPORTANT: RESIZABLE WINDOW (NO FULLSCREEN)
root.resizable(True, True)
root.minsize(800, 480)

# ================= GRID LAYOUT =================
root.grid_rowconfigure(0, weight=1)   # video
root.grid_rowconfigure(1, weight=0)   # controls
root.grid_columnconfigure(0, weight=1)

CONTROL_HEIGHT = 140
current_video_width = 800
current_video_height = 480 - CONTROL_HEIGHT

# ================= VIDEO FRAME =================
video_frame = Frame(root, bg="black")
video_frame.grid(row=0, column=0, sticky="nsew")

video_label = Label(video_frame, bg="black")
video_label.pack(fill="both", expand=True)

# ================= CONTROL BAR =================
control_frame = Frame(root, bg="black", height=CONTROL_HEIGHT)
control_frame.grid(row=1, column=0, sticky="ew")
control_frame.grid_propagate(False)

control_frame.grid_columnconfigure(0, weight=1)
control_frame.grid_columnconfigure(1, weight=0)
control_frame.grid_columnconfigure(2, weight=0)

# ---- STATUS TEXT ----
status_text = Text(
    control_frame,
    wrap="word",
    font=("Arial", 16),
    fg="white",
    bg="black",
    bd=0,
    height=3
)
status_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

scrollbar = Scrollbar(control_frame, command=status_text.yview)
scrollbar.grid(row=0, column=1, sticky="ns")
status_text.config(yscrollcommand=scrollbar.set)

# ---- BUTTONS ----
button_frame = Frame(control_frame, bg="black")
button_frame.grid(row=0, column=2, padx=10, pady=10)

def on_capture():
    global paused
    if not paused:
        paused = True
        run_detection()
    else:
        paused = False
        update_status("Ready")

capture_button = Button(
    button_frame,
    text="CAPTURE",
    font=("Arial", 22, "bold"),
    bg="green",
    fg="white",
    width=9,
    command=on_capture
)
capture_button.grid(row=0, column=0, padx=5)

def toggle_mode():
    global MODE
    MODE = {"AUTO": "ONLINE", "ONLINE": "OFFLINE", "OFFLINE": "AUTO"}[MODE]
    mode_label.config(text=f"MODE: {MODE}")

mode_button = Button(
    button_frame,
    text="MODE",
    font=("Arial", 14, "bold"),
    bg="gray",
    fg="white",
    width=8,
    command=toggle_mode
)
mode_button.grid(row=0, column=1, padx=5)

mode_label = Label(
    button_frame,
    text="MODE: AUTO",
    font=("Arial", 12, "bold"),
    fg="yellow",
    bg="black"
)
mode_label.grid(row=1, column=0, columnspan=2, pady=4)

# ================= SPEECH =================
speech_queue = queue.Queue()
speaking = False

def process_speech_queue():
    global speaking
    if not speaking and not speech_queue.empty():
        text = speech_queue.get()
        speaking = True
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        speaking = False
    root.after(100, process_speech_queue)

root.after(100, process_speech_queue)

# ================= RESIZE HANDLER =================
def on_resize(event):
    global current_video_width, current_video_height
    w = root.winfo_width()
    h = root.winfo_height()
    current_video_width = max(1, w)
    current_video_height = max(1, h - CONTROL_HEIGHT)

root.bind("<Configure>", on_resize)

# ================= UI HELPERS =================
def show_frame(frame):
    if current_video_width <= 1 or current_video_height <= 1:
        return
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (current_video_width, current_video_height))
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(img)
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

def update_status(msg):
    status_text.config(state="normal")
    status_text.delete("1.0", "end")
    status_text.insert("end", msg)
    status_text.config(state="disabled")

# ================= LIVE VIDEO =================
def update_video():
    global last_frame
    if not paused:
        ret, frame = cap.read()
        if ret:
            last_frame = frame.copy()
            show_frame(frame)
    root.after(30, update_video)

# ================= OFFLINE DETECTION =================
def offline_detect(frame):
    detected_objects = []
    detected_texts = []

    original_frame = frame.copy()
    yolo_frame = frame.copy()

    results = yolo_model(yolo_frame, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = r.names[cls]
            detected_objects.append(label)
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cv2.rectangle(yolo_frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(yolo_frame,label,(x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    for bbox,text,conf in ocr_reader.readtext(original_frame):
        if conf < 0.4: continue
        detected_texts.append(text)
        pts = np.array(bbox, np.int32)
        cv2.polylines(yolo_frame,[pts],True,(255,0,0),2)
        x,y = pts[0]
        cv2.putText(yolo_frame,text[:15],(x,y-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

    return yolo_frame, detected_objects, detected_texts

# ================= ONLINE DETECTION =================
def online_detect(frame):
    detected_objects = []
    detected_texts = []

    _, buf = cv2.imencode(".jpg", frame)
    image = vision.Image(content=buf.tobytes())

    objects = vision_client.object_localization(image=image).localized_object_annotations
    ocr = vision_client.text_detection(image=image).text_annotations

    h,w,_ = frame.shape

    for o in objects:
        v = o.bounding_poly.normalized_vertices
        x1,y1,x2,y2 = int(v[0].x*w),int(v[0].y*h),int(v[2].x*w),int(v[2].y*h)
        detected_objects.append(o.name)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,o.name,(x1,y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    for t in ocr[1:]:
        pts = [[v.x,v.y] for v in t.bounding_poly.vertices if v.x is not None]
        if len(pts) < 4: continue
        label = t.description.strip()
        detected_texts.append(label)
        pts = np.array(pts,np.int32)
        cv2.polylines(frame,[pts],True,(255,0,0),2)
        x,y = pts[0]
        cv2.putText(frame,label[:15],(x,y-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

    return frame, detected_objects, detected_texts

# ================= RUN DETECTION =================
def run_detection():
    global last_frame
    if last_frame is None:
        return

    frame = last_frame.copy()
    update_status("Detecting...")

    if MODE == "ONLINE" or (MODE == "AUTO" and internet_available()):
        frame, objs, texts = online_detect(frame)
        mode_label.config(text="MODE: ONLINE")
    else:
        frame, objs, texts = offline_detect(frame)
        mode_label.config(text="MODE: OFFLINE")

    show_frame(frame)

    message = "Objects: "
    message += ", ".join(objs) if objs else "None"
    message += "\nText: "
    message += ", ".join(texts) if texts else "None"

    update_status(message)
    speech_queue.put(message)

# ================= EXIT =================
def on_close():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

# ================= START =================
update_video()
root.mainloop()
