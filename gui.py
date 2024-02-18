# Model Imports
import torch
import numpy as np
from model import load_model
from torchvision.transforms import ToPILImage

# GUI Imports
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw

# Model Loading
model = load_model()

# Initialize GUI
top = Tk()
top.geometry('800x600')
top.title("Pedestrian Detection System")
top.configure(background='#CDCDCD')

# Defining Labels to be displayed
image_label = Label(top)
image_label.pack(side="bottom", expand=True)

image_input = Label(top)

def show_detected_image(file_path):
    """
    Display the image with detected pedestrians using Tkinter.

    Args:
    - file_path (str): The path to the input image file.

    Returns:
    None
    """
    image, predictions = detect_pedestrians(file_path)
    image = ToPILImage()(image)
    print(f"Predictions: {predictions}")

    # Draw bounding boxes on the image
    mask = Image.new('L', image.size, color=255)
    draw = ImageDraw.Draw(mask)

    for box in predictions['boxes']:
        box = tuple(map(int, box.cpu().numpy()))
        draw.rectangle(box, fill=250, outline="red", width=2)
    image.putalpha(mask)

    # Convert the image to PhotoImage
    img_tk = ImageTk.PhotoImage(image)

    # Display the image in a Label
    image_label.configure(image=img_tk)
    image_label.image = img_tk

def load_image(file_path):
    """
    Load and preprocess the image from the file path.

    Args:
    - file_path (str): The path to the input image file.

    Returns:
    - list: List containing the preprocessed image as a PyTorch tensor.
    """
    img = Image.open(file_path)
    img = np.array(img)
    img = np.delete(img, 0, 1)
    img = img / 255.
    img = np.transpose(img, (2, 0, 1))  # Change the order to (C, H, W)

    # Convert NumPy array to PyTorch tensor
    img = torch.from_numpy(img).float()
    return [img]

def detect_pedestrians(file_path):
    """
    Detect pedestrians in the given image using the pre-trained model.

    Args:
    - file_path (str): The path to the input image file.

    Returns:
    - tuple: Tuple containing the original image and the detection predictions.
    """
    img = load_image(file_path)
    with torch.no_grad():
        predictions = model(img)[0]
    return img[0], predictions

def show_detect_btn(file_path):
    """
    Show the 'Detect Image' button after an image is uploaded.

    Args:
    - file_path (str): The path to the input image file.

    Returns:
    None
    """
    detect_btn = Button(top, 
        text="Detect Pedestrians", 
        command=lambda: show_detected_image(file_path),
        padx=10, pady=5
    )

    detect_btn.configure(
        background="#364156",
        foreground="white",
        font=('arial', 10, 'bold')
    )

    detect_btn.place(relx=0.79, rely=0.46)

def upload_image():
    """
    Open a file dialog to upload an image and display it.

    Args:
    None

    Returns:
    None
    """
    try:
        filepath = filedialog.askopenfilename()
        uploaded = Image.open(filepath)
        uploaded.thumbnail((
            (top.winfo_width()/2.25), (top.winfo_height()/2.25)
        ))

        image = ImageTk.PhotoImage(uploaded)
        image_input.configure(image=image)
        image_input.image = image
        show_detect_btn(filepath)

    except Exception as e:
        print("Error:", e)

# Detecting Pedestrian
upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground="white", font=('arial', 10, 'bold'))
upload.pack(side="bottom", expand=True)

# Visual Setting
image_input.pack(side="bottom", expand=True)

# Set Heading / Title
heading = Label(top, text="Pedestrian Detection System", pady=20, font=('arial', 20, 'bold'))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

# Start program
top.mainloop()

# DEBUG
print("Done")
