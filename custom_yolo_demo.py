import time
import cv2
import numpy as np

from rtmlib import PoseTracker, Custom, RTMPose, draw_skeleton
from yolov8 import YOLOv8Class  # Import YOLOv8Class from testonnx2.py

# ---------- Config ----------
device = 'cpu'
backend = 'onnxruntime'  # alternatives: 'opencv', 'openvino'
openpose_skeleton = False  # True for openpose-style, False for mmpose-style

# Local paths
onnx_model_path = 'newyolov8m.onnx'  # Path to your YOLOv8 ONNX model
pose_model_path = (
    'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/'
    'rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip'
)

# Optional: class name lookup (edit if you have a custom dataset)
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# ---------- Detector (local ONNX) ----------
yolov8_detector = YOLOv8Class(onnx_model_path, conf_thres=0.2, iou_thres=0.3)

# ---------- Pose model (RTMPose) via Custom pipeline ----------
custom_pipeline = Custom(
    det_class=None,          # we'll inject detector manually
    det=None,
    pose_class='RTMPose',
    pose=pose_model_path,
    pose_input_size=(288, 384),
    to_openpose=openpose_skeleton,
    backend=backend,
    device=device
)
# Inject the detector and ensure two-stage mode is used
custom_pipeline.det_model = yolov8_detector
custom_pipeline.one_stage = False  # important: pipeline should call det -> pose

# ---------- Tracker (wrap the prebuilt pipeline) ----------
wholebody = PoseTracker(
    solution=lambda **_: custom_pipeline,
    mode='custom',  # important: tell PoseTracker we are using a custom pipeline
    det_frequency=7,
    to_openpose=openpose_skeleton,
    backend=backend,
    device=device,
    tracking=False
)

# ---------- Video I/O ----------
cap = cv2.VideoCapture(0)  # 0 for webcam, or put a video file path

# Function to format the detections output
def format_detections(boxes, scores, class_ids):
    detections = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        detections.append([*box, score, class_id])
    return np.array(detections)

# Function to draw all detections
def draw_all_detections(img, dets, class_names=None):
    if dets is None or dets.size == 0:
        return img
    for x1, y1, x2, y2, score, cls_id in dets.astype(float):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_id = int(cls_id)
        label = class_names[cls_id] if class_names and 0 <= cls_id < len(class_names) else f"cls={cls_id}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (90, 200, 255), 2)
        cv2.putText(img, f"{label} {score:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 200, 255), 1, cv2.LINE_AA)
    return img

# Main loop for processing video frames
print("Press 'q' or close the window to quit.")
window_name = "RTMPose + CUSTOMYOLOONNX (local ONNX)"

while cap.isOpened():
    ok, frame = cap.read() 
    frame = cv2.imread('demo.jpg')  # For testing with a static image, comment out when using webcam
    if not ok:
        break

    t0 = time.time()
    boxes, scores, class_ids = yolov8_detector(frame)  # Get detections using the YOLOv8 detector
    dt = time.time() - t0
    print(f"Frame time: {dt:.3f}s")

    # Visualize
    vis = frame.copy()

    # Format detections and draw them
    all_dets = format_detections(boxes, scores, class_ids)
    vis = draw_all_detections(vis, all_dets, class_names=COCO_NAMES)

    # Pass the bounding boxes and image to PoseTracker
    keypoints, scores = wholebody(frame)  # PoseTracker calls custom_pipeline -> (det -> pose)

    # Draw skeletons for people (keypoints/scores from RTMPose)
    vis = draw_skeleton(
        vis,
        keypoints,
        scores,
        openpose_skeleton=openpose_skeleton,
        kpt_thr=0.4
    )

    vis = cv2.resize(vis, (960, 540))
    cv2.imshow(window_name, vis)

    # --- Exit conditions ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
