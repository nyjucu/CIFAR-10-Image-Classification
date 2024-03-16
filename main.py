import time

import cv2
import numpy
import numpy as np
import threading
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import tkinter as tk

video_name = "plane.mp4"
video = cv2.VideoCapture(video_name)

model = load_model("classification.h5")
class_names = ["PLANE", "CAR", "BIRD", "CAT", "DEER", "DOG", "FROG", "HORSE", "SHIP", "TRUCK"]

root = tk.Tk()

video_label = tk.Label(root)
prediction_label = tk.Label(root, font=25, padx=50)
canvas = tk.Canvas(root, width=480, height=480)

video_label.pack(side=tk.LEFT)
canvas.pack(side=tk.LEFT)
prediction_label.pack(side=tk.LEFT)


def predict(image) -> None:
    image_frame = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)

    labeled_image = ImageTk.PhotoImage(image=Image.fromarray(numpy.array(cv2.resize(image_frame, (480, 480)))))
    canvas.create_image(0, 0, anchor="nw", image=labeled_image)

    image_frame = image_frame / 255.0
    image_frame = np.expand_dims(image_frame, axis=0)

    predictions = model.predict(image_frame)
    predictions = np.array(predictions).reshape(-1)

    class_index = np.argmax(predictions)
    class_label = class_names[class_index]

    predictions = [int(p * 100) for p in predictions]

    class_and_probability = list(zip(class_names, predictions))
    class_and_probability = sorted(class_and_probability, key=lambda x: x[1], reverse=True)

    text = ""

    for cp in class_and_probability:
        text += cp[0] + ": " + str(cp[1]) + "%\n"

    prediction_label.config(text=text)


def stream(label) -> None:
    while True:
        ret, frame = video.read()
        if not ret:
            break

        image_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predict(image_frame)

        image_frame = Image.fromarray(image_frame)
        frame_image = ImageTk.PhotoImage(image_frame)

        label.config(image=frame_image)
        label.image = frame_image


if __name__ == "__main__":
    thread = threading.Thread(target=stream, args=(video_label,))
    thread.daemon = 1
    thread.start()

    root.mainloop()

