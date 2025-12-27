import cv2
import tkinter as tk
from tkinter import Label, Button, Frame, Text, Scrollbar
from PIL import Image, ImageTk
from google.cloud import vision
import pyttsx3
import queue
import numpy as np

# ================= GOOGLE VISION =================
vision_client = vision.ImageAnnotatorClient()

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

# ================= STATE =================
paused = False
last_frame = None

# ================= TKINTER =================
root = tk.Tk()
root.title("Capture Detection (Vision API)")
root.configure(bg="black")
root.attributes("-fullscreen", True)
root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

# ================= VIDEO FRAME =================
video_frame = Frame(root, bg="black")
video_frame.pack(fill="both", expand=True)

video_label = Label(video_frame, bg="black")
video_label.pack(fill="both", expand=True)

# ================= CONTROL FRAME =================
control_frame = Frame(root, bg="black")
control_frame.pack(fill="x", side="bottom")

status_text = Text(
    control_frame,
    wrap="word",
    font=("Arial", 18),
    fg="white",
    bg="black",
    bd=0,
    height=6
)
status_text.pack(fill="both", side="left", expand=True)

scrollbar = Scrollbar(control_frame, command=status_text.yview)
scrollbar.pack(side="left", fill="y")
status_text.config(yscrollcommand=scrollbar.set)

capture_button = Button(
    control_frame,
    text="CAPTURE",
    font=("Arial", 24, "bold"),
    bg="green",
    fg="white",
    command=lambda: on_capture()
)
capture_button.pack(side="right", padx=10, pady=5, fill="y")

# ================= SPEECH (MAIN THREAD SAFE) =================
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

# ================= DISPLAY =================
def show_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 480))
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(img)
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

# ================= STATUS =================
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

# ================= CAPTURE BUTTON HANDLER =================
def on_capture():
    global paused

    if not paused:
        paused = True
        run_detection()
    else:
        paused = False
        update_status("Ready")

# ================= DETECTION =================
def run_detection():
    global last_frame

    if last_frame is None:
        return

    frame = last_frame.copy()
    update_status("Detecting...")

    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    image = vision.Image(content=buffer.tobytes())

    # Object detection
    objects = vision_client.object_localization(image=image)\
                           .localized_object_annotations

    # OCR
    texts = vision_client.text_detection(image=image)\
                         .text_annotations
    ocr_items = texts[1:] if len(texts) > 1 else []

    detected_objects = []
    detected_texts = []

    h, w, _ = frame.shape

    # Draw object boxes (GREEN)
    for obj in objects:
        verts = obj.bounding_poly.normalized_vertices
        if len(verts) < 4:
            continue

        x1 = int(verts[0].x * w)
        y1 = int(verts[0].y * h)
        x2 = int(verts[2].x * w)
        y2 = int(verts[2].y * h)

        detected_objects.append(obj.name)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, obj.name, (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw text boxes (BLUE)
    for text in ocr_items:
        pts = []
        for v in text.bounding_poly.vertices:
            if v.x is not None and v.y is not None:
                pts.append([v.x, v.y])

        if len(pts) < 4:
            continue

        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (255, 0, 0), 2)

        label = text.description.strip()
        if label:
            detected_texts.append(label)
            x, y = pts[0]
            cv2.putText(frame, label[:15], (x, max(y - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    show_frame(frame)

    message = "Objects: "
    message += ", ".join(detected_objects) if detected_objects else "None"
    message += "\nText: "
    message += ", ".join(detected_texts) if detected_texts else "None"

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