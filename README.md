# Face Recognition Attendance System

![Project Demo](demo.gif) <!-- Add a demo GIF or image here -->

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
  - [Workflow](#workflow)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Description

The Face Recognition Attendance System is a Python-based web application that allows users to take attendance by recognizing faces. It utilizes the K-Nearest Neighbors (KNN) algorithm and augmentation techniques for face recognition. Users can register their faces by clicking the "Add User" button, which captures 100 images of the user and trains the model. When the "Take Attendance" button is clicked, the camera opens and detects registered users, marking their attendance.

## Features

- User registration and face capture.
- Face recognition for attendance tracking.
- Utilizes K-Nearest Neighbors (KNN) algorithm.
- Augmentation techniques to improve recognition accuracy.
- Web-based frontend using Flask.

## Technologies Used

- Python
- Flask
- K-Nearest Neighbors (KNN)
- MTCNN (Multi-task Cascaded Convolutional Networks)
- Face Recognition Library
- OpenCV

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-recognition-attendance-system.git
   cd face-recognition-attendance-system

2.Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
### Workflow
###User Registration:

Click the "Add User" button to initiate the user registration process.The system captures 100 images of the user's face to create a dataset for training.
### Training Model:

After capturing images, the system uses K-Nearest Neighbors (KNN) and augmentation techniques to train the face recognition model.
This step ensures that the system can recognize the registered user during attendance.\
### Taking Attendance:

Click the "Take Attendance" button to start the attendance process.
The camera opens to capture images in real-time.\
### Face Recognition:

The system uses the trained model to recognize registered users in the camera feed.
Detected users' attendance is marked and recorded.


