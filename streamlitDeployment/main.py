import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, VideoProcessorBase
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

from utils import FullBodyPoseEmbedder, PoseClassifier, EMADictSmoothing, RepetitionCounter

@st.cache_resource
def load_model():
    class_name = "pushups_down"

    pose_samples_folder = 'fitness_poses_csvs_out'

    # Initialize tracker.
    pose_tracker = mp_pose.Pose()
    #upper_body_only=False

    # Initialize embedder.c
    pose_embedder = FullBodyPoseEmbedder()

    # Initialize classifier.
    # Ceck that you are using the same parameters as during bootstrapping.
    pose_classifier = PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    # # Uncomment to validate target poses used by classifier and find outliers.
    # outliers = pose_classifier.find_pose_sample_outliers()
    # print('Number of pose sample outliers (consider removing them): ', len(outliers))

    # Initialize EMA smoothing.
    pose_classification_filter = EMADictSmoothing(
        window_size=10,
        alpha=0.2)

    # Initialize counter.
    repetition_counter = RepetitionCounter(
        class_name=class_name,
        enter_threshold=6,
        exit_threshold=4)
    
    return pose_tracker, pose_classifier, pose_classification_filter, repetition_counter


pose_tracker, pose_classifier, pose_classification_filter, repetition_counter = load_model()

goalReps = st.number_input("How many reps do you want to do?", 1, 1000, 10)

def playSound():
    st.text("You did it!")
    st.balloons()
    st.audio("cheer.mp3")
    import winsound
    winsound.Beep(500, 1000)

class ImageProcessor(VideoProcessorBase):

    def __init__(self):
        self.repetitions_count = 0

    def transform(self, image):
        image = image.to_ndarray(format="bgr24")
        image.flags.writeable = False

        result = pose_tracker.process(image=image)
        pose_landmarks = result.pose_landmarks


        if pose_landmarks:

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)
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
            self.repetitions_count = repetition_counter(pose_classification_filtered)
        else:
            # No pose => no classification on current frame.
            pose_classification = None

            pose_classification_filtered = None

            self.repetitions_count = repetition_counter.n_repeats


        print(self.repetitions_count, goalReps)

        if self.repetitions_count >= goalReps:
            playSound()

        cv2.putText(image,
            str(self.repetitions_count),
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 255, 255),
            2,
            cv2.LINE_4)
        
        return image
    

    


resetButton = st.button("Reset")
if resetButton:
    st.cache_resource.clear()
    st.experimental_rerun()


webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=ImageProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

