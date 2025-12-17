import cv2
import tkinter as tk
from tkinter import Label, Button, Frame, Text, Scrollbar
from PIL import Image, ImageTk
from ultralytics import YOLO
import easyocr
import pyttsx3
import threading
import queue

# ================= YOLO =================
model = YOLO("yolov8n.pt")

# ================= EASYOCR =================
reader = easyocr.Reader(['en'], gpu=False)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

# ================= TKINTER =================
root = tk.Tk()
root.title("Capture Detection")
root.configure(bg="black")
root.attributes("-fullscreen", True)
root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

# ================= VIDEO =================
video_frame = Frame(root, bg="black")
video_frame.pack(fill="both", expand=True)

video_label = Label(video_frame, bg="black")
video_label.pack(expand=True)

# ================= STATUS AREA =================
status_frame = Frame(root, bg="black", height=160)
status_frame.pack(fill="x", side="bottom")
status_frame.pack_propagate(False)

scrollbar = Scrollbar(status_frame)
scrollbar.pack(side="right", fill="y")

status_text = Text(
    status_frame,
    wrap="word",
    yscrollcommand=scrollbar.set,
    font=("Arial", 18),
    fg="white",
    bg="black",
    bd=0
)
status_text.pack(fill="both", expand=True)
scrollbar.config(command=status_text.yview)
status_text.config(state="disabled")

# ================= SPEECH =================
speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break

        # Each message gets its own thread for non-blocking speech
        def speak(text_to_speak):
            engine = pyttsx3.init()
            engine.say(text_to_speak)
            engine.runAndWait()
            engine.stop()

        threading.Thread(target=speak, args=(text,), daemon=True).start()

# Start speech worker thread
threading.Thread(target=speech_worker, daemon=True).start()

# ================= DRAW YOLO =================
def draw_boxes(frame, results):
    names = []
    r = results[0]
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        label = f"{r.names[cls]} {conf:.2f}"
        names.append(r.names[cls])

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame, names

# ================= EASYOCR (FULL FRAME) =================
def detect_text(frame):
    texts = []
    results = reader.readtext(frame)
    for _, text, conf in results:
        if conf > 0.4:
            texts.append(text)
    return texts

# ================= DISPLAY =================
def show_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (1150, 750))
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(img)
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

# ================= LIVE VIDEO =================
def update_video():
    ret, frame = cap.read()
    if ret:
        show_frame(frame)
    root.after(10, update_video)

# ================= STATUS =================
def update_status(msg):
    status_text.config(state="normal")
    status_text.delete("1.0", "end")
    status_text.insert("end", msg)
    status_text.see("end")
    status_text.config(state="disabled")

# ================= CAPTURE =================
def capture_predict():
    ret, frame = cap.read()
    if not ret:
        update_status("Camera error")
        return

    update_status("Detecting...")

    results = model(frame)
    frame, objects = draw_boxes(frame, results)
    texts = detect_text(frame)

    show_frame(frame)

    message = ""
    if objects:
        message += "Objects: " + ", ".join(objects)
    if texts:
        if message:
            message += "\n"
        message += "Text: " + ", ".join(texts)

    if not message:
        message = "Nothing detected"

    update_status(message)
    speech_queue.put(message)

# ================= BUTTON =================
Button(
    root,
    text="CAPTURE",
    command=capture_predict,
    font=("Arial", 26, "bold"),
    bg="green",
    fg="white",
    width=18,
    height=2
).pack(pady=10)

# ================= START =================
update_video()

# ================= EXIT =================
def on_close():
    speech_queue.put(None)
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
