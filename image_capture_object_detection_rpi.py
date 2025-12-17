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
model = YOLO("yolov8l.pt")  # Nano model for Raspberry Pi CPU

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

# ================= VIDEO FRAME =================
video_frame = Frame(root, bg="black")
video_frame.pack(fill="both", expand=True)

video_label = Label(video_frame, bg="black")
video_label.pack(fill="both", expand=True)

# ================= STATUS + BUTTON FRAME =================
control_frame = Frame(root, bg="black")
control_frame.pack(fill="x", side="bottom")

# Status Text
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

# Adaptive CAPTURE Button
capture_button = Button(
    control_frame,
    text="CAPTURE",
    command=lambda: threading.Thread(target=capture_predict, daemon=True).start(),
    font=("Arial", 24, "bold"),
    bg="green",
    fg="white"
)
capture_button.pack(side="right", padx=10, pady=5, fill="y")

# ================= SPEECH =================
speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break

        def speak(text_to_speak):
            engine = pyttsx3.init()
            engine.say(text_to_speak)
            engine.runAndWait()
            engine.stop()

        threading.Thread(target=speak, args=(text,), daemon=True).start()

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

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame, names

# ================= EASYOCR =================
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
    frame = cv2.resize(frame, (640, 480))
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(img)
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

# ================= LIVE VIDEO =================
def update_video():
    ret, frame = cap.read()
    if ret:
        show_frame(frame)
    root.after(30, update_video)  # slower refresh to reduce CPU load

# ================= STATUS =================
def update_status(msg):
    status_text.config(state="normal")
    status_text.delete("1.0", "end")
    status_text.insert("end", msg)
    status_text.see("end")
    status_text.config(state="disabled")

# ================= CAPTURE =================
# ================= CAPTURE =================
def capture_predict():
    ret, frame = cap.read()
    if not ret:
        update_status("Camera error")
        return

    update_status("Detecting...")

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (640, 480))

    # YOLO detection (objects only)
    results = model(small_frame)
    frame_with_boxes, objects = draw_boxes(small_frame.copy(), results)  # draw boxes on copy

    # EasyOCR detection (text only)
    texts = detect_text(frame)  # <-- Use original frame for OCR

    # Show frame with YOLO boxes
    show_frame(frame_with_boxes)

    # Build message
    message = ""
    if objects:
        message += "Objects: " + ", ".join(objects)
    else:
        message += "Objects: None"

    if texts:
        message += "\nText: " + ", ".join(texts)
    else:
        message += "\nText: None"

    # Update status and speak
    update_status(message)
    speech_queue.put(message)


# ================= START =================
update_video()

# ================= EXIT =================
def on_close():
    speech_queue.put(None)
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()


