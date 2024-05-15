import cv2
import random
from ultralytics import YOLO
from helpers import predict_and_detect

model = YOLO("yolov9c.pt")

def process_video(input_video_path, output_video_path, model, classes=[], conf=0.5):
  """
  Process a video by applying a model to each frame and saving the result to an output video file.

  Args:
    input_video_path (str): The path to the input video file.
    output_video_path (str): The path to save the output video file.
    model: The model to apply to each frame of the video.
    classes (list): A list of class ids to detect in the video frames. Default is an empty list.
    conf (float): The confidence threshold for object detection. Default is 0.5.

  Returns:
    None
  """
  # Open the video file
  cap = cv2.VideoCapture(input_video_path)

  # Get video properties
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS)

  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    # Process the frame
    result_frame, _ = predict_and_detect(model, frame, classes, conf)

    # Write the processed frame to the output video
    out.write(result_frame)

    # Optionally, display the frame for debugging purposes
    cv2.imshow('Frame', result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Release everything if job is finished
  cap.release()
  out.release()
  cv2.destroyAllWindows()

# Example usage
input_video_path = './samples/video/people.mp4'

# Output video path
output_video_path = './samples/video/output.mp4'

# Process the video
process_video(input_video_path, output_video_path, model, classes=[], conf=0.5)