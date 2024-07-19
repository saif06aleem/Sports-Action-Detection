import streamlit as st
import os
from keras.models import load_model
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
from mediapipe.python.solutions import drawing_utils as mp_drawing
from PIL import Image
import io

# Define paths for each model
model_paths = {
    "Model1": "model_baseball.h5",
    "Model2": "model_basketball.h5",
    "Model3": "model_football.h5",
    "Model4": "model_volleyball.h5",
    "Model5": "model.h5",
}

# Define class names for each model
class_names = {
    "Model1": ['pitching', 'saving', 'striking'],
    "Model2": ['dribble', 'drop', 'shoot'],
    "Model3": ['defending', 'goal_keeping', 'kicking'],
    "Model4": ['block', 'defence', 'smash'],
    "Model5": ['pitching', 'saving', 'striking', 'dribble', 'drop', 'shoot', 'defending', 'goal_keeping', 'kicking', 'block', 'defence', 'smash'],
}

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define landmark names
n_landmarks = 33
landmark_names = [
    'nose',
    'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear',
    'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_pinky_1', 'right_pinky_1',
    'left_index_1', 'right_index_1',
    'left_thumb_2', 'right_thumb_2',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
]

# Define torso size multiplier
torso_size_multiplier = 2.5

# Define column names for DataFrame
col_names = []
for i in range(n_landmarks):
    name = mp_pose.PoseLandmark(i).name
    name_x = name + '_X'
    name_y = name + '_Y'
    name_z = name + '_Z'
    name_v = name + '_V'
    col_names.append(name_x)
    col_names.append(name_y)
    col_names.append(name_z)
    col_names.append(name_v)

def load_models():
    models = {}
    for model_name, model_path in model_paths.items():
        models[model_name] = load_model(model_path, compile=True)
    return models

models = load_models()

def process_image(image_file, threshold, model_name):
    model = models[model_name]
    class_name = class_names[model_name]
    sport_name = sport_names[model_name]  # Get the corresponding sport name

    img = Image.open(image_file)
    img = np.array(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)
    if result.pose_landmarks:
        lm_list = []
        for landmarks in result.pose_landmarks.landmark:
            # Preprocessing
            max_distance = 0
            lm_list.append(landmarks)
        center_x = (lm_list[landmark_names.index('right_hip')].x +
                    lm_list[landmark_names.index('left_hip')].x) * 0.5
        center_y = (lm_list[landmark_names.index('right_hip')].y +
                    lm_list[landmark_names.index('left_hip')].y) * 0.5

        shoulders_x = (lm_list[landmark_names.index('right_shoulder')].x +
                       lm_list[landmark_names.index('left_shoulder')].x) * 0.5
        shoulders_y = (lm_list[landmark_names.index('right_shoulder')].y +
                       lm_list[landmark_names.index('left_shoulder')].y) * 0.5

        for lm in lm_list:
            distance = math.sqrt((lm.x - center_x) ** 2 + (lm.y - center_y) ** 2)
            if (distance > max_distance):
                max_distance = distance
        torso_size = math.sqrt((shoulders_x - center_x) **
                               2 + (shoulders_y - center_y) ** 2)
        max_distance = max(torso_size * torso_size_multiplier, max_distance)

        pre_lm = list(np.array([[(landmark.x - center_x) / max_distance, (landmark.y - center_y) / max_distance,
                                 landmark.z / max_distance, landmark.visibility] for landmark in lm_list]).flatten())
        data = pd.DataFrame([pre_lm], columns=col_names)
        predict = model.predict(data)[0]
        if max(predict) > threshold:
            pose_class = class_name[predict.argmax()]

            # Draw skeleton on the image
            annotated_image = img.copy()
            mp_drawing.draw_landmarks(
                annotated_image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Add predicted class name and sports name to the annotated image
            font_scale = 0.5  # Adjust the font scale here

            # Calculate text size
            text_size_class = cv2.getTextSize(f'Pose Class: {pose_class}', cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            text_size_sport = cv2.getTextSize(f'Sport: {sport_name}', cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

            # Get image width and height
            img_height, img_width, _ = annotated_image.shape

            # Define text positions
            text_pos_class = (img_width - text_size_class[0] - 50, 50)
            text_pos_sport = (img_width - text_size_sport[0] - 50, 100)

            cv2.putText(annotated_image, f'Pose : {pose_class}', text_pos_class, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(annotated_image, f'Sport: {sport_name}', text_pos_sport, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
                        cv2.LINE_AA)

            # Save annotated image
            annotated_image_path = f"annotated_image_{model_name}.jpg"
            cv2.imwrite(annotated_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            return predict, pose_class, annotated_image, annotated_image_path
        else:
            pose_class = 'Unknown Pose'
            return predict, pose_class, None, None



def process_video(video_file, threshold, model_name):
    # Save the uploaded video file to the local filesystem
    with open("uploaded_video.mp4", "wb") as f:
        f.write(video_file.read())

    # Open the saved video file using OpenCV
    cap = cv2.VideoCapture("uploaded_video.mp4")

    output_frames = []

    # Load model
    model = models[model_name]
    class_name = class_names[model_name]
    sport_name = sport_names[model_name]

    # Loop through each frame of the video
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process pose estimation on the frame
        result = pose.process(frame_rgb)

        # Draw skeleton on the frame
        if result.pose_landmarks:
            annotated_frame = frame.copy()
            mp_drawing.draw_landmarks(
                annotated_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract landmarks data
            lm_list = []
            for landmarks in result.pose_landmarks.landmark:
                # Preprocessing
                max_distance = 0
                lm_list.append(landmarks)
            center_x = (lm_list[landmark_names.index('right_hip')].x +
                        lm_list[landmark_names.index('left_hip')].x) * 0.5
            center_y = (lm_list[landmark_names.index('right_hip')].y +
                        lm_list[landmark_names.index('left_hip')].y) * 0.5

            shoulders_x = (lm_list[landmark_names.index('right_shoulder')].x +
                           lm_list[landmark_names.index('left_shoulder')].x) * 0.5
            shoulders_y = (lm_list[landmark_names.index('right_shoulder')].y +
                           lm_list[landmark_names.index('left_shoulder')].y) * 0.5

            for lm in lm_list:
                distance = math.sqrt((lm.x - center_x) ** 2 + (lm.y - center_y) ** 2)
                if (distance > max_distance):
                    max_distance = distance
            torso_size = math.sqrt((shoulders_x - center_x) **
                                   2 + (shoulders_y - center_y) ** 2)
            max_distance = max(torso_size * torso_size_multiplier, max_distance)

            pre_lm = list(np.array([[(landmark.x - center_x) / max_distance, (landmark.y - center_y) / max_distance,
                                     landmark.z / max_distance, landmark.visibility] for landmark in lm_list]).flatten())
            data = pd.DataFrame([pre_lm], columns=col_names)

            # Predict pose class
            predict = model.predict(data)[0]
            pose_class = class_name[predict.argmax()]

            # Add predicted class name and sports name to the annotated frame
            cv2.putText(annotated_frame, pose_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f'Sport: {sport_name}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Append annotated frame to output frames list
            output_frames.append(annotated_frame)

    # Release video capture object
    cap.release()

    # Remove the saved video file from the local filesystem
    os.remove("uploaded_video.mp4")

    # Save annotated video frames
    output_frames_path = f"annotated_video_{model_name}"
    os.makedirs(output_frames_path, exist_ok=True)

    for i, frame in enumerate(output_frames):
        cv2.imwrite(os.path.join(output_frames_path, f"frame_{i}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Convert annotated frames to video
    output_video_path = f"annotated_video_{model_name}.mp4"
    frame_rate = 30
    height, width, _ = output_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    for frame in output_frames:
        out.write(frame)
    out.release()

    return output_frames, output_frames_path, output_video_path



# Define sports names for each model
sport_names = {
    "Model1": "Baseball",
    "Model2": "Basketball",
    "Model3": "Football",
    "Model4": "Volleyball",
    "Model5": "Unknown Sport",
}

def main():
    st.title("Pose Prediction App")
    st.sidebar.title("Settings")

    threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    file_type = st.sidebar.radio("Choose Input Type", ("Image", "Video"))

    model_name = st.sidebar.selectbox("Choose Model", list(model_paths.keys()))

    if file_type == "Image":
        image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if image_file is not None:
            st.image(image_file, caption='Uploaded Image', use_column_width=True)
            predict_button = st.button("Predict")
            if predict_button:
                predictions, pose_class, output_img, annotated_image_path = process_image(image_file, threshold, model_name)
                if output_img is not None:
                    st.image(output_img, caption=f'Output Image: {pose_class}', use_column_width=True)
                st.write("Predictions:", predictions)
                st.write("Predicted Pose Class:", pose_class)
                if pose_class in class_names[model_name]:
                    st.write("Pose Class belongs to:", sport_names[model_name])
                    st.write("Detailed Class Name:", class_names[model_name][class_names[model_name].index(pose_class)])
                else:
                    st.write("Pose Class belongs to: Unknown")
                if annotated_image_path is not None:
                    st.write("Annotated Image saved at:", annotated_image_path)

    else:
        video_file = st.file_uploader("Upload Video", type=['mp4'])
        if video_file is not None:
            st.video(video_file)
            predict_button = st.button("Predict")
            if predict_button:
                # Perform prediction on video frames
                output_frames, output_frames_path, output_video_path = process_video(video_file, threshold, model_name)
                if output_video_path:
                    # Display the annotated video
                    st.video(output_video_path)

                if output_frames_path is not None:
                    st.write("Annotated Video Frames saved at:", output_frames_path)

if __name__ == "__main__":
    main()
