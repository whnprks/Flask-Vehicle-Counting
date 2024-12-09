## Overview

The Object Detection and Classification in Video project focuses on implementing object detection and classification capabilities in a video stream. It is designed for beginners interested in learning computer vision concepts, particularly object detection and classification. The project involves selecting object classes, choosing a pre-trained model, implementing the application, and evaluating its performance.

## Features

- Object detection and classification in videos or images
- User-friendly web interface for uploading media and visualizing results
- Utilization of pre-trained models for object detection and classification

## Installation

To set up the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone "https://github.com/whnprks/vehicle-detection.git"
   ```

2. Navigate to the project directory:

   ```bash
   cd Object_Detection
   ```

3. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv env
   ```

4. Activate the virtual environment:

   - On Windows: `.\env\Scripts\activate`
   - On Unix or Linux: `source env/bin/activate`

5. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the application, follow these steps:

1. Launch the Flask web application:

   ```bash
   python webapp.py
   ```

2. Open a web browser and navigate to `http://localhost:5000` to access the application.

3. Upload videos or images using the provided interface on the left side.

4. The results, including detected and classified objects, will be displayed on the right-side panel.

### Web App Interface

The web application features a user-friendly interface for uploading videos or images and visualizing the object detection and classification results.

The following are the results obtained from testing the web application:

- Object detection and classification results are displayed in real-time on the right-side panel of the web application interface.
- For video uploads, the processed video with object detection overlays is shown dynamically.
- Object bounding boxes and labels are drawn around detected objects, providing visual feedback to the user.
