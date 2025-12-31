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
MODE = "AUTO"  # AUTO | ONLINE | OFFLINE

# ================= TKINTER =================
root = tk.Tk()
root.title("Vision Capture")

# ---- AUTO SCREEN SIZE ----
root.update_idletasks()
SCREEN_W = root.winfo_screenwidth()
SCREEN_H = root.winfo_screenheight()

root.geometry(f"{SCREEN_W}x{SCREEN_H}")
root.minsize(SCREEN_W, SCREEN_H)
root.maxsize(SCREEN_W, SCREEN_H)

# ---- SAFE AREA FIX FOR RASPBERRY PI DESKTOP ----
WINDOW_TOP_MARGIN = 40   # title bar + desktop panel
SAFE_HEIGHT = SCREEN_H - WINDOW_TOP_MARGIN

# ---- LAYOUT CONSTANTS ----
CONTROL_HEIGHT = 140

# ================= VIDEO FRAME =================
video_frame = Frame(
    root,
    bg="black",
    height=SAFE_HEIGHT - CONTROL_HEIGHT
)
video_frame.pack(side="top", fill="x")
video_frame.pack_propagate(False)

video_label = Label(video_frame, bg="black")
video_label.pack(fill="both", expand=True)

# ================= CONTROL BAR =================
control_frame = Frame(root, bg="black", height=CONTROL_HEIGHT)
control_frame.pack(side="bottom", fill="x")
control_frame.pack_propagate(False)

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
status_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)

scrollbar = Scrollbar(control_frame, command=status_text.yview)
scrollbar.pack(side="left", fill="y")
status_text.config(yscrollcommand=scrollbar.set)

# ---- BUTTON AREA (HORIZONTAL) ----
button_frame = Frame(control_frame, bg="black")
button_frame.pack(side="right", padx=10, pady=10)

# ================= BUTTONS =================
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
    height=1,
    command=on_capture
)
capture_button.pack(side="left", padx=8)

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
mode_button.pack(side="left", padx=5)

mode_label = Label(
    control_frame,
    text="MODE: AUTO",
    font=("Arial", 12, "bold"),
    fg="yellow",
    bg="black"
)
mode_label.pack(side="right", padx=10)

# ================= SPEECH =================
speech_queue = queue.Queue()
speaking = False

def process_speech_queue():
    global speaking
    if not speaking and not speech_queue.empty():
        text = speech_queue.get()
        speaking = True
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print("Speech error:", e)
        speaking = False
    root.after(100, process_speech_queue)

root.after(100, process_speech_queue)

# ================= UI HELPERS =================
def show_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(
        frame,
        (SCREEN_W, SAFE_HEIGHT - CONTROL_HEIGHT)
    )
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(img)
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

def update_status(msg):
    status_text.config(state="normal")
    status_text.delete("1.0", "end")
    status_text.insert("end", msg)
    status_text.see("end")
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

    # YOLO OBJECTS
    results = yolo_model(yolo_frame, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = r.names[cls]
            detected_objects.append(label)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(yolo_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                yolo_frame,
                label,
                (x1, max(y1 - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # EASYOCR ON CLEAN FRAME
    for bbox, text, conf in ocr_reader.readtext(original_frame):
        if conf < 0.4:
            continue
        detected_texts.append(text)
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(yolo_frame, [pts], True, (255, 0, 0), 2)
        x, y = pts[0]
        cv2.putText(
            yolo_frame,
            text[:15],
            (x, max(y - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    return yolo_frame, detected_objects, detected_texts

# ================= ONLINE DETECTION =================
def online_detect(frame):
    detected_objects = []
    detected_texts = []

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    image = vision.Image(content=buf.tobytes())

    objects = vision_client.object_localization(image=image)\
                           .localized_object_annotations
    ocr = vision_client.text_detection(image=image).text_annotations

    h, w, _ = frame.shape

    # OBJECTS
    for o in objects:
        v = o.bounding_poly.normalized_vertices
        x1, y1 = int(v[0].x * w), int(v[0].y * h)
        x2, y2 = int(v[2].x * w), int(v[2].y * h)
        detected_objects.append(o.name)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            o.name,
            (x1, max(y1 - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # OCR (WITH TEXT)
    for t in ocr[1:]:
        pts = []
        for v in t.bounding_poly.vertices:
            if v.x is not None and v.y is not None:
                pts.append([v.x, v.y])
        if len(pts) < 4:
            continue

        label = t.description.strip()
        if not label:
            continue

        detected_texts.append(label)
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (255, 0, 0), 2)

        x, y = pts[0]
        cv2.putText(
            frame,
            label[:15],
            (x, max(y - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    return frame, detected_objects, detected_texts

# ================= RUN DETECTION =================
def run_detection():
    global last_frame

    if last_frame is None:
        return

    frame = last_frame.copy()
    update_status("Detecting...")

    use_online = (
        MODE == "ONLINE" or
        (MODE == "AUTO" and internet_available())
    )

    if use_online:
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

    while not speech_queue.empty():
        speech_queue.get_nowait()
    speech_queue.put(message)

# ================= EXIT =================
def on_close():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

# ================= START =================
update_video()
root.mainloop()
