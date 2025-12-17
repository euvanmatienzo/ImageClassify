import cv2
import time
import pyttsx3
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


# ---------------- Allowed Library ---------------- #
LIBRARY = set([
    *[str(i) for i in range(11)],
    *[chr(i) for i in range(65, 91)],
    "APPLE","BANANA","PINEAPPLE","MANGO","GRAPES","ORANGE","WATERMELON","STRAWBERRY",
    "EGGPLANT","CARROT","CABBAGE","PUMPKIN","GARLIC","ONION","RADISH","BELL PEPPER","CUCUMBER","LETTUCE",
    "CAT","DOG","COW","FISH","SHARK","CHICKEN","DUCK","SHEEP","HORSE","PIG",
    "PENCIL","NOTEBOOK","CHALK","ERASER","CHAIR","PHONE","PEN","BAG","BOOK","PERSON"
])

# ---------------- Text-to-Speech ---------------- #
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# ---------------- MediaPipe Classifier ---------------- #
options = vision.ImageClassifierOptions(
    base_options=BaseOptions(model_asset_path="mobilenet_v1_1.0_224.tflite"),
    max_results=5,
    score_threshold=0.05
)

classifier = vision.ImageClassifier.create_from_options(options)


# ---------------- Camera ---------------- #
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("Press SPACE to capture | ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Preview", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

    if key == 32:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = classifier.classify(mp_image)

        y = 40
        detected = []

        if result.classifications:
            for c in result.classifications[0].categories:
                label = c.category_name.upper()
                score = c.score

                print(label, score)

                if label in LIBRARY:
                    detected.append(label)
                    cv2.putText(
                        frame,
                        f"{label} ({score:.2f})",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    y += 35

        cv2.imshow("Result", frame)

        if detected:
            speak(", ".join(detected))

        time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()
