import argparse
import io
from PIL import Image
import datetime

import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    send_file,
    url_for,
    Response,
    jsonify,
)
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob

from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('best.pt')  # Load YOLO model once globally
progress = 0  # Global variable to store progress

@app.route("/")
def hello_world():
    if "image_path" in request.args:
        image_path = request.args["image_path"]
        return render_template("index.html", image_path=image_path)
    return render_template("index.html")



@app.route("/", methods=["GET", "POST"])
def predict_img():
    global progress
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            # basepath = os.path.dirname(__file__)
            
            # Get the absolute path of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            print("script_dir:", script_dir)

            # Set the basepath to the directory containing the script
            basepath = script_dir
            print("bashpath:", basepath)
            
            filepath = os.path.join(basepath, 'uploads', f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)

            file_extension = f.filename.rsplit('.', 1)[1].lower()


            if file_extension == 'jpg':
                # Handle image upload
                img = cv2.imread(filepath)

                # Perform the detection
                detections = model(img, save=True)

                # Find the latest subdirectory in the 'runs/detect' folder
                folder_path = os.path.join(basepath, 'runs', 'detect')
                print("folder path:", folder_path)
                subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
                print("subfolders:", subfolders)
                latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
                print("latest subfolders:", latest_subfolder)

                # Construct the relative path to the detected image file
                static_folder = os.path.join(basepath, 'static', 'assets')
                print("static folder:", static_folder)
                relative_image_path = os.path.relpath(os.path.join(folder_path, latest_subfolder, f.filename),
                                                    static_folder)
                image_path = os.path.join(folder_path, latest_subfolder, f.filename)
                print("image path:", image_path)
                print("Relative image path:", relative_image_path)  # Print the relative_image_path for debugging

                return render_template('index.html', image_path=relative_image_path, media_type='image')

            elif file_extension == "mp4":

                video_path = filepath
                cap = cv2.VideoCapture(video_path)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (frame_width, frame_height))

                min_width_react = 5
                min_hight_react = 5

                count_line_position = 800
                offset = 6

                counter = {'bus': 0, 'car': 0, 'motorcycle': 0}
                detect = []

                def center_handle(x, y, w, h):
                    cx = x + int(w / 2)
                    cy = y + int(h / 2)
                    return cx, cy

                def put_text_with_background(frame, text, pos, font, font_scale, font_color, bg_color, thickness):
                    # Mendapatkan ukuran teks
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    
                    # Menentukan posisi kotak latar belakang
                    x, y = pos
                    box_coords = ((x, y), (x + text_size[0], y - text_size[1]))
                    
                    # Menggambar kotak latar belakang
                    cv2.rectangle(frame, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
                    
                    # Menambahkan teks di atas kotak latar belakang
                    cv2.putText(frame, text, pos, font, font_scale, font_color, thickness)
                
                frame_counter = 0
                progress = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_counter += 1
                    progress = (frame_counter / total_frames) * 100

                    # Perform object detection
                    results = model(frame)[0]

                    for result in results:
                        x1, y1, x2, y2 = map(int, result.boxes.xyxy[0])
                        confidence = result.boxes.conf[0]
                        class_id = result.boxes.cls[0]
                        class_name = model.names[int(class_id)]

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'{class_name}: {confidence:.2f}'
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if class_name in counter:
                            w = x2 - x1
                            h = y2 - y1
                            validate_counter = (w >= min_width_react) and (h >= min_hight_react)
                            if not validate_counter:
                                print(f"Skipped {class_name} due to size: w={w}, h={h}")
                                continue

                            center = center_handle(x1, y1, w, h)
                            detect.append((center, class_name))
                            cv2.circle(frame, center, 4, (0, 0, 255), -1)

                    cv2.line(frame, (450, count_line_position), (1100, 600), (255, 127, 0), 3)

                    for (center, cls) in detect:
                        cx, cy = center
                        line_y = int(-0.3077 * (cx - 450) + 800)  # Menghitung cy pada garis untuk cx tertentu
                        if cy < (line_y + offset) and cy > (line_y - offset) and cx > 600:
                            counter[cls] += 1
                            cv2.line(frame, (450, 800), (1100, 600), (0, 127, 255), 3)
                            detect.remove((center, cls))
                            print(f"{cls} counter: {counter[cls]}")

                    # Tampilkan jumlah setiap kendaraan
                    for i, (cls, cnt) in enumerate(counter.items()):
                        put_text_with_background(frame, f"{cls.capitalize()}: {cnt}", (50, 70 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (255, 127, 0), (0, 0, 0), 3)
                    
                    # Hitung total kendaraan
                    total_vehicles = sum(counter.values())
                    put_text_with_background(frame, f"Total: {total_vehicles}", (frame_width - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 127, 0), (0, 0, 0), 3)

                    out.write(frame)

                cap.release()
                out.release()
                cv2.destroyAllWindows()

                return render_template('index.html', video_path='output.mp4', media_type='video')

    # If no file uploaded or GET request, return the template with default values
    return render_template("index.html", image_path="", media_type='image')

@app.route('/progress')
def get_progress():
    global progress
    return jsonify(progress=progress)

@app.route('/download/<filename>')
def download_file(filename):
    # Ensure the file exists and is in the expected directory
    file_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.isfile(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found", 404


@app.route("/<path:filename>")
def display(filename):
    folder_path = "runs/detect"
    subfolders = [
        f
        for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f))
    ]
    latest_subfolder = max(
        subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x))
    )
    directory = os.path.join(folder_path, latest_subfolder)
    print("printing directory: ", directory)
    files = os.listdir(directory)
    latest_file = files[0]

    print(latest_file)

    image_path = os.path.join(directory, latest_file)

    file_extension = latest_file.rsplit(".", 1)[1].lower()

    if file_extension == "jpg":
        return send_file(image_path, mimetype="image/jpeg")
    elif file_extension == "mp4":
        return send_file(image_path, mimetype="video/mp4")
    else:
        return "Invalid file format"


def get_frame():
    folder_path = os.getcwd()
    mp4_files = "output.mp4"
    print("files being read...")
    video = cv2.VideoCapture(mp4_files)  # detected video path
    while True:
        success, frame = video.read()
        if not success:
            print("file not being read")
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

        yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
        )
        time.sleep(0.1)  # Control the frame rate to display one frame every 100 milliseconds:


# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    # folder_path = os.getcwd()
    # mp4_file = "output.mp4"
    # video_path = os.path.join(folder_path, mp4_file)
    # return send_file(video_path, mimetype="video")
    return Response(get_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/webcam_feed")
def webcam_feed():
    cap = cv2.VideoCapture(0)  # 0 for camera

    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Perform object detection on the frame
            img = Image.fromarray(frame)
            results = model(img, save=True)

            # Plot the detected objects on the frame
            res_plotted = results[0].plot()
            img_BGR = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)

            # Convert the frame to JPEG format for streaming
            ret, buffer = cv2.imencode(".jpg", img_BGR)
            frame = buffer.tobytes()

            yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)
