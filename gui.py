import cv2
import torch
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageTk
import threading

# Model Loading
yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = yolo.getUnconnectedOutLayersNames()

# Initialize GUI
top = Tk()
top.geometry('800x600')
top.title("Pedestrian Detection System")
top.configure(background='#CDCDCD')

# Defining Labels to be displayed
image_input = Label(top)
image_input.pack(side="bottom", expand=True)

# Function to detect pedestrians using YOLOv3
def detect_pedestrians_yolo(frame):
    height, width, channels = frame.shape

    # YOLO input pre-processing
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo.setInput(blob)
    outs = yolo.forward(layer_names)

    # Post-process the outputs
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes on the image
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame

# Function to display the detected image
def show_detected_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = detect_pedestrians_yolo(img)

    # Convert the image to PhotoImage
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(img)

    # Display the image in a Label
    image_input.configure(image=img_tk)
    image_input.image = img_tk

# Function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 600))
        frame = detect_pedestrians_yolo(frame)
        cv2.imshow('Real-Time Detection', frame)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to upload an image
def upload_image():
    try:
        filepath = filedialog.askopenfilename()
        show_detected_image(filepath)

    except Exception as e:
        print("Error:", e)

# Detecting Pedestrian
upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground="white", font=('arial', 10, 'bold'))
upload.pack(side="bottom", expand=True)

# Detecting Pedestrian in Video
def start_video_processing():
    video_path = filedialog.askopenfilename()
    threading.Thread(target=process_video, args=(video_path,), daemon=True).start()

video_btn = Button(top, text="Upload Video", command=start_video_processing, padx=10, pady=5)
video_btn.configure(background="#364156", foreground="white", font=('arial', 10, 'bold'))
video_btn.pack(side="bottom", expand=True)

# Set Heading / Title
heading = Label(top, text="Pedestrian Detection System", pady=20, font=('arial', 20, 'bold'))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

# Start program
top.mainloop()

# DEBUG
print("Done")
