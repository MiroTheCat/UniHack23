from matplotlib import pyplot as plt
import csv
import numpy as np
import os
import io
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import requests
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image
import sys
import tqdm
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import time
from PushupClasses import *

# Initilize tracker, classifier and counter.
# Do that before every video as all of them have state.
class_name='pushups_down'


# Folder with pose class CSVs. That should be the same folder you using while
# building classifier to output CSVs.
pose_samples_folder = 'fitness_poses_csvs_out'

# Initialize tracker.
pose_tracker = mp_pose.Pose()
#upper_body_only=False

# Initialize embedder.
pose_embedder = FullBodyPoseEmbedder()

# Initialize classifier.
# Ceck that you are using the same parameters as during bootstrapping.
pose_classifier = PoseClassifier(
    pose_samples_folder=pose_samples_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

# Initialize EMA smoothing.
pose_classification_filter = EMADictSmoothing(
    window_size=10,
    alpha=0.2)

# Initialize counter.
repetition_counter = RepetitionCounter(
    class_name=class_name,
    enter_threshold=6,
    exit_threshold=4)

from mediapipe.python.solutions import drawing_utils as mp_drawing

font = cv2.FONT_HERSHEY_SIMPLEX

frame_idx = 0
# output_frame = None
start_time = time.time()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    # Get next frame of the video.
    # frame_js = video_frame()
    # if not frame_js:
    #     break
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.resize(image, (320, 180))
    if frame_idx % 1 == 0:
        # Run pose tracker.
        result = pose_tracker.process(image=image)
        pose_landmarks = result.pose_landmarks

        if pose_landmarks is not None:
          mp_drawing.draw_landmarks(
              image=image,
              landmark_list=pose_landmarks,
              connections=mp_pose.POSE_CONNECTIONS)

        if pose_landmarks is not None:
            # Get landmarks.
            frame_height, frame_width = image.shape[0], image.shape[1]
            pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                       for lmk in pose_landmarks.landmark], dtype=np.float32)
            assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

            # Classify the pose on the current frame.
            pose_classification = pose_classifier(pose_landmarks)

            # Smooth classification using EMA.
            pose_classification_filtered = pose_classification_filter(pose_classification)

            # Count repetitions.
            repetitions_count = repetition_counter(pose_classification_filtered)
        else:
            # No pose => no classification on current frame.
            pose_classification = None

            # Still add empty classification to the filter to maintaing correct
            # smoothing for future frames.
            pose_classification_filtered = pose_classification_filter(dict())
            pose_classification_filtered = None

            # Don't update the counter presuming that person is 'frozen'. Just
            # take the latest repetitions count.
            repetitions_count = repetition_counter.n_repeats

    # Draw classification plot and repetition counter.
    # output_frame = pose_classification_visualizer(
    #     frame=output_frame,
    #     pose_classification=pose_classification,
    #     pose_classification_filtered=pose_classification_filtered,
    #     repetitions_count=repetitions_count)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.putText(image,
                str(repetitions_count),
                (50, 50),
                font, 1,
                (0, 255, 255),
                2,
                cv2.LINE_4)

    cv2.imshow('Webcam', image)

    print(frame_idx)
    frame_idx += 1
    if cv2.waitKey(5) & 0xFF == 27:
      break

# Release MediaPipe resources.
pose_tracker.close()
cap.release()
cv2.destroyAllWindows()
