import os

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import pyrqlite.dbapi2 as dbapi2
import struct

from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

from face_alignment import align

# Define preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load face embedding model
model_face_embedding = torch.hub.load('otroshi/edgeface', 'edgeface_s_gamma_05', source='github', pretrained=True)

# Move the model to the GPU if available
model_face_embedding = model_face_embedding.to(device)

# Set model to eval
model_face_embedding.eval()

print(f"Model is loaded on {device}")

# Load YOLO model
model_yolo = YOLO("models/yolov11s-face.pt")

# Connect to the database
connection = dbapi2.connect(
    host='localhost',
    port=4001,
)

# Check SQLite and sqlite-vec version
conn = connection.cursor()
conn.execute("SELECT sqlite_version(), vec_version()")
sqlite_version, vec_version = conn.fetchone()
print(f"SQLite Version: {sqlite_version}, sqlite-vec Version: {vec_version}")

# Create virtual table for vector storage
conn.execute("""
CREATE VIRTUAL TABLE IF NOT EXISTS images USING vec0(
    name TEXT, 
    file_dir TEXT, 
    embedding float[512] distance_metric=cosine
);
""")

# Function to serialize embeddings
def serialize_f32(vector):
    """Serialize a list of floats into raw bytes for storage."""
    return struct.pack("%sf" % len(vector), *vector)

# create temp_data folder
if not os.path.exists("temp_data"):
    os.makedirs("temp_data")

# Open the video source (0 for the default camera or specify a file path)
video = cv2.VideoCapture("sample_video.mp4")

# frame count
frame_count = 0

while True:
    ret, image = video.read()  # Read a frame from the video
    if not ret:
        break  # Exit the loop if no frame is captured

    # Print Frame Count
    print(frame_count)

    # Save image
    image_path = os.path.join("temp_data", str(frame_count) + ".jpg")
    cv2.imwrite(image_path, image)

    print(image_path)
    
    # Update database from a video loop
    person_name = "name_" + str(frame_count)

    # Plot Image
    # plt.imshow(image)
    # plt.axis("off")  # Hide axis
    # plt.show()

    if image is not None:
        # Run YOLO inference
        results = model_yolo(image, device=device, verbose=False)  

        # Extract bounding boxes
        if len(results) > 0:
            boxes = results[0].boxes  # Get detected bounding boxes

            if len(boxes) > 0:
                x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy()

                margin = 30
                h, w, _ = image.shape  # Get image dimensions

                # Clip coordinates to stay within image bounds
                x1 = int(max(0, x1 - margin))
                y1 = int(max(0, y1 - margin))
                x2 = int(min(w, x2 + margin))
                y2 = int(min(h, y2 + margin))

                face_image = image[y1:y2, x1:x2]

                # Plot Image
                # plt.imshow(face_image)
                # plt.axis("off")  # Hide axis
                # plt.show()

                # Get Embedding
                # Convert the OpenCV image (BGR) to PIL image (RGB)
                pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
                aligned = align.get_aligned_face(None, pil_image) # align face

                # Check If alignment result good
                if aligned is not None:
                    
                    # Plot Image
                    # plt.imshow(aligned)
                    # plt.axis("off")  # Hide axis
                    # plt.show()

                    transformed_input = transform(aligned).unsqueeze(0).to(device) # preprocessing

                    # extract embedding
                    face_embedding = model_face_embedding(transformed_input).cpu().detach().numpy().flatten()

                    # save into database
                    conn.execute(
                        "INSERT INTO images (rowid, name, file_dir, embedding) VALUES (?, ?, ?, ?)",
                        (None, person_name, image_path, serialize_f32(face_embedding)),
                    )
    
    # increment frame_count
    frame_count = frame_count + 1

# Print Done
print("Done")

# Release the video object
video.release()
