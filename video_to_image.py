import cv2
import os


def video_to_images(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize frame count
    frame_count = 0

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # Save the frame as an image
        output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(output_path, frame)

        # Increment frame count
        frame_count += 1

    # Release the video capture object
    cap.release()


# Example usage
video_path = "C:/Users/USER/Downloads/5586712-hd_1080_1920_25fps.mp4"
output_folder = "Basketball_images"
video_to_images(video_path, output_folder)
