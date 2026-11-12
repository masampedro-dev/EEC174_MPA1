import argparse
import os
import time
import cv2
import numpy as np
from sort import Sort

#notes for others: to run the following script, run in terminal
# 1. Run person tracking and counting using default model and video paths:
#    python yolo_counter.py
# 2. Run with explicit model, video, and output arguments:
#    python yolo_counter.py --weights yolo_files/yolov3.weights --cfg yolo_files/yolov3.cfg --labels yolo_files/coco.names --video mot_vid/MOTS20-09-raw.mp4 --output mot_vid/MOTS20-09-result.mp4

#will show something like the following:
#Frame 0524 | Processing Time: 0.3754s | People Counted: 107
#Frame 0525 | Processing Time: 0.3811s | People Counted: 107

#Phase 2 Complete.
#Total People Counted: 107
#Annotated video saved as: mot_vid/MOTS20-09-result.mp4

#Tracks cumulative unique people by recording the highest SORT track ID assigned so far

def parse_arguments():
    #set up and read command-line flags and default file paths for ease of calling the script
    parser = argparse.ArgumentParser(
        description="EEC174 Phase 2: YOLOv3 Object Tracking & People Counter"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo_files/yolov3.weights",
        help="Path to YOLOv3 weights file (default: yolo_files/yolov3.weights)",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="yolo_files/yolov3.cfg",
        help="Path to YOLOv3 configuration file (default: yolo_files/yolov3.cfg)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="yolo_files/coco.names",
        help="Path to class labels file (default: yolo_files/coco.names)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="mot_vid/MOTS20-09-raw.mp4",
        help="Path to input video (default: mot_vid/MOTS20-09-raw.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mot_vid/MOTS20-09-result.mp4",
        help="Path to save annotated output video (default: mot_vid/MOTS20-09-result.mp4)",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)",
    )
    parser.add_argument(
        "--nms_thresh",
        type=float,
        default=0.3,
        help="Non-Maximum Suppression threshold (default: 0.3)",
    )
    return parser.parse_args()


def load_model(cfg_path, weights_path):
    #again load the model and find its output layers just like phase 1
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

    try:
        layers = net.getLayerNames()
        out_indices = np.array(net.getUnconnectedOutLayers()).flatten()
        output_layers = [layers[i - 1] for i in out_indices]
    except Exception:
        output_layers = ["yolo_82", "yolo_94", "yolo_106"]

    return net, output_layers


def load_classes(labels_path):
    #read class names from coco.names
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    with open(labels_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes


def main():
    args = parse_arguments()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Input video not found: {args.video}")

    #ensure output folder exists (mot_vid)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # 1. Load network and identify 'person' class ID
    classes = load_classes(args.labels)
    person_class_id = classes.index("person") if "person" in classes else 0
    net, output_layers = load_model(args.cfg, args.weights)

    # 2. Setup video input and output streams
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Could not open input video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    #setup the output video writer (.mp4 format)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # 3. Initialize SORT tracker and highest track ID counter
    mot_tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
    max_id = 0
    frame_number = 0

    print(f"Tracking started on: {args.video}")
    print(f"Target class: 'person' (ID: {person_class_id}) | Saving to: {args.output}\n")

    #read and process the video one frame at a time
    while True:
        ret, frame = cap.read()
        if not ret:
            break #end of video reached

        frame_number += 1
        t_start = time.perf_counter()

        # Generate a normalized 416x416 RGB blob for YOLO
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)

        boxes = []
        confidences = []

        # Parse detections and filter strictly for 'person' above the confidence threshold
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if class_id == person_class_id and confidence > args.conf_thresh:
                    box = detection[0:4] * np.array([width, height, width, height])
                    x_c, y_c, b_w, b_h = box.astype("int")
                    x_min = int(x_c - (b_w / 2))
                    y_min = int(y_c - (b_h / 2))

                    boxes.append([x_min, y_min, int(b_w), int(b_h)])
                    confidences.append(confidence)

        # Apply Non-Maximum Suppression (remove duplicate overlapping boxes)
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, args.conf_thresh, args.nms_thresh
        )

        # Convert [x, y, w, h] to SORT format [x1, y1, x2, y2, score]
        dets_for_sort = []
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                x, y, w, h = boxes[i]
                dets_for_sort.append([
                    float(x),
                    float(y),
                    float(x + w),
                    float(y + h),
                    float(confidences[i]),
                ])

        dets_array = np.array(dets_for_sort) if dets_for_sort else np.empty((0, 5))

        # Update SORT tracker to match people across frames and assign persistent IDs
        trackers = mot_tracker.update(dets_array)

        # Draw yellow bounding boxes and numeric track IDs onto the frame
        for d in trackers:
            x1, y1, x2, y2, track_id = d
            #update highest ID seen so the overhead counter matches the box IDs
            track_id = int(track_id)
            max_id = max(max_id, track_id)

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 255),  # Yellow in BGR
                2,
            )
            cv2.putText(
                frame,
                str(track_id),
                (int(x1), max(int(y1) - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),  # Yellow in BGR
                2,
            )

        # Draw green cumulative person count overlay
        cv2.putText(
            frame,
            f"People: {max_id}",
            (550, 45), #placed like center right of the video
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),  # Green in BGR
            3,
        )

        #save the annotated frame to the output video file
        out.write(frame)
        t_frame = time.perf_counter() - t_start

        print(
            f"Frame {frame_number:04d} | "
            f"Processing Time: {t_frame:.4f}s | "
            f"People Counted: {max_id}"
        )

    #release video file handles once all frames are processed
    cap.release()
    out.release()

    print("\nPhase 2 Complete.")
    print(f"Total People Counted: {max_id}")
    print(f"Annotated video saved as: {args.output}")


if __name__ == "__main__":
    main()