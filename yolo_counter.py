import argparse
import os
import sys
import time
import cv2
import numpy as np

#run this code
#python yolo_counter.py --weights yolo_files/yolov3.weights --cfg yolo_files/yolov3.cfg --labels yolo_files/coco.names --video mot_vid/MOTS20-09-raw.mp4 --output mot_vid/MOTS20-09-result.mp4

#then it will output a frame as the following,
#Frame 0064 | Processing Time: 0.5401s | People Counted: 9
#Frame 0065 | Processing Time: 0.4631s | People Counted: 9
#etc. 

#once it finishes, it will save the output video into mot_vid folder
#will be saved as the same name except -result.mp4

# Ensure sort.py can be imported from root or src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from sort import Sort
except ImportError:
    try:
        from src.sort import Sort
    except ImportError as e:
        raise ImportError(
            "Could not locate sort.py. Ensure sort.py is in the root directory or src/."
        ) from e


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="EEC174 Phase 2: YOLOv3 Object Tracking & People Counter"
    )
    # Required inputs per README
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to YOLOv3 weights file (e.g., yolo_files/yolov3.weights)",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to YOLOv3 configuration file (e.g., yolo_files/yolov3.cfg)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to class labels file (e.g., yolo_files/coco.names)",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video (.mp4) (e.g., mot_vid/MOTS20-09-raw.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mot_vid/MOTS20-09-result.mp4",
        help="Path to save annotated output video (default: mot_vid/MOTS20-09-result.mp4)",
    )

    # Thresholds matching lab specifications
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (probability_minimum = 0.5)",
    )
    parser.add_argument(
        "--nms_thresh",
        type=float,
        default=0.3,
        help="Non-Maximum Suppression threshold (threshold = 0.3)",
    )

    return parser.parse_args()


def load_model(cfg_path, weights_path):
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
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    with open(labels_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes


def main():
    args = parse_arguments()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Input video not found: {args.video}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # 1. Load network and identify target 'person' class ID
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # 3. Initialize SORT tracker and unique count registry
    mot_tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
    unique_ids = set()
    frame_number = 0

    print(f"Tracking started on: {args.video}")
    print(f"Target class: 'person' (ID: {person_class_id}) | Saving to: {args.output}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        t_start = time.perf_counter()

        # Generate 416x416 blob for Darknet
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)

        boxes = []
        confidences = []

        # Parse detections and filter strictly for the 'person' class
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

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, args.conf_thresh, args.nms_thresh
        )

        # Convert NMS boxes [x, y, w, h] to SORT format [x1, y1, x2, y2, score]
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

        # Update SORT tracker
        trackers = mot_tracker.update(dets_array)

      # Draw tracker bounding boxes and numeric IDs 
        for d in trackers:
            x1, y1, x2, y2, track_id = d
            track_id = int(track_id)
            unique_ids.add(track_id)

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                str(track_id),
                (int(x1), max(int(y1) - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

        # Draw updated cumulative person count overlay
     
        cv2.putText(
            frame,
            f"People: {len(unique_ids)}",
            (550, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3,
        )

        out.write(frame)
        t_frame = time.perf_counter() - t_start

        # Print per-frame index and runtime per README requirement
        print(f"Frame {frame_number:04d} | Processing Time: {t_frame:.4f}s | People Counted: {len(unique_ids)}")

    cap.release()
    out.release()

    print("\nPhase 2 Complete.")
    print(f"Total Unique People Counted: {len(unique_ids)}")
    print(f"Annotated video saved as: {args.output}")


if __name__ == "__main__":
    main()