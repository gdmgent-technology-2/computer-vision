# Computer Vision Code

This is a repository for students to learn more about computer visio

## Inference

There are three basic examples:

- Object detection in images
- Object detection in video
- Object detection in webcam

## Training

Install the needed library in conda environment

`pip install ultralytics`

Start training the model via

`yolo detect train data=data.yaml model=yolov9c.pt epochs=10 imgsz=640 batch=8`

## References

- [Open CV](https://opencv.org/)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [Computer Vision Datasets](https://public.roboflow.com/)

## Author

- Tim De Paepe (tim.depaepe@arteveldehs.be)
