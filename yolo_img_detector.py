import argparse
import glob
import os
import time
from collections import Counter, defaultdict
import cv2
import numpy as np

#run this code within the terminal
# Runs on the entire imgs/ folder using default model files

#python yolo_img_detector.py

# Runs on a single image (replace sample.jpg to something inside imgs folder)
#python yolo_img_detector.py --image imgs/sample.jpg

# Runs with isolated metric flags
#python yolo_img_detector.py -inf
#python yolo_img_detector.py -classes_all

#should see a bunch of outputs like this,
"""
Average Inference Time: 0.51770 seconds
Total Number of Classes Detected: 19

Total Detection Breakdown
Apple: 4
Banana: 1
Bicycle: 1
Bird: 1
Car: 9
Cat: 2
Diningtable: 1
Dog: 2
Elephant: 2
Fork: 1
Giraffe: 1
Horse: 4
Kite: 4
Knife: 1
Motorbike: 1
Orange: 14
Person: 19
Tie: 1
Zebra: 8

Per Image Breakdown
beach.jpg => Kite: 4, Person: 9
cat.jpg => Dog: 1
cats.jpg => Cat: 2
eagle.jpg => Bird: 1
elephants.jpg => Elephant: 2
farm.jpg => Dog: 1, Horse: 1, Person: 1
forest.jpg => Giraffe: 1
fruits.jpg => Apple: 4, Banana: 1, Diningtable: 1, Orange: 12
horses.jpg => Horse: 3
inception.jpg => Person: 6, Tie: 1
kitchen.jpg => Fork: 1, Knife: 1, Orange: 2
road.jpg => Bicycle: 1, Car: 8, Motorbike: 1, Person: 3
sample.jpg => Car: 1
savannah.jpg => Zebra: 8
"""
#afterwards all the output images with the prediction boxes into the folder out_imgs/
def parse_arguments():
    #set up and read command-line flags and default file paths for ease when calling the script within terminal

    parser = argparse.ArgumentParser(
        description="EEC174 Phase 1: YOLOv3 Object Detection on Images"
    )
    # Required model files
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo_files/yolov3.weights",
        help="Path to YOLOv3 weights file (directory: yolo_files/yolov3.weights)",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="yolo_files/yolov3.cfg",
        help="Path to YOLOv3 configuration file (directory: yolo_files/yolov3.cfg)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="yolo_files/coco.names",
        help="Path to class labels file (directory: yolo_files/coco.names)",
    )

    # Input image options: use can pass --image or --dir (defaults to --dir imgs (the whole folder) ) 
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--image",
        type=str,
        nargs="+",
        help="Path to one or more image files (space-separated)",
    )
    input_group.add_argument(
        "--dir",
        type=str,
        default="imgs",
        help="Path to directory containing input .jpg images",
    )

    # Output and detection parameters
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out_imgs",
        help="Directory to save output annotated images (default: out_imgs)",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5 as per the README)",
    )
    parser.add_argument(
        "--nms_thresh",
        type=float,
        default=0.3,
        help="Non-Maximum Suppression threshold (default: 0.3 as per the README)",
    )

    # Display flags
    parser.add_argument(
        "-inf",
        action="store_true",
        help="Display only the average inference time",
    )
    parser.add_argument(
        "-classes_all",
        action="store_true",
        help="Display only the total number of classes detected",
    )

    return parser.parse_args()


def load_model(cfg_path, weights_path):
    #load the Darknet YOLOv3 NN and find its output layer names
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    #load YOLOv3 and the pretrained weights just like tutorial
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

    # Use project README-specified layer outputs to prevent 1D indexing issues
    #find the names of the three YOLO detection layers, should be 82, 94, 106
    try:
        layers = net.getLayerNames()
        out_indices = np.array(net.getUnconnectedOutLayers()).flatten()
        output_layers = [layers[i - 1] for i in out_indices]
    except Exception:
        #for standard YOLOv3
        output_layers = ["yolo_82", "yolo_94", "yolo_106"]

    return net, output_layers


def load_classes(labels_path):
    #read class names from coco.names
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    classes = []
    with open(labels_path, "r", encoding="utf-8") as file:
        for line in file:
            clean_name = line.strip()
            if clean_name:
                classes.append(clean_name)

    return classes


def collect_image_paths(args):
    # Build a sorted list of valid image file paths
    image_paths = []

    # Case 1: user passed specific image file(s) with --image
    if args.image:
        for path in args.image:
            if os.path.isfile(path):
                normalized = os.path.normpath(path)
                if normalized not in image_paths:
                    image_paths.append(normalized)
            else:
                print(f"[Warning] Image path not found: {path}")

    # Case 2: user passed a directory with --dir (or used the default imgs folder)
    elif args.dir:
        if not os.path.isdir(args.dir):
            raise NotADirectoryError(f"Directory not found: {args.dir}")
        valid_extensions = (".jpg", ".jpeg")
        for filename in sorted(os.listdir(args.dir)):
            full_path = os.path.join(args.dir, filename)
            if os.path.isfile(full_path) and filename.lower().endswith(valid_extensions):
                normalized = os.path.normpath(full_path)
                if normalized not in image_paths:
                    image_paths.append(normalized)

    if not image_paths:
        raise ValueError("No valid input images found to process.")
    return image_paths

 


def process_image(image_path, net, output_layers, classes, conf_thresh, nms_thresh):
    #run YOLOv3 on a single image, draw bounding boxes, and return detections
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Failed to read image: {image_path}")
        return None, 0.0, []

    h, w = image.shape[:2]

    # Convert to 416x416 blob for Darknet YOLOv3
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), swapRB=True, crop=False
    )
    net.setInput(blob)

    # Measure inference timing
    start_time = time.perf_counter()
    layer_outputs = net.forward(output_layers)
    inference_time = time.perf_counter() - start_time

    boxes = []
    confidences = []
    class_ids = []

    #loop through all candidate predictions from the 3 YOLO output layers
    for output in layer_outputs:
        for detection in output:
            #first 5 numbers are box info; indices 5+ are confidence scores for the 80 classes
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            #keep only predictions above our confidence threshold (0.5)
            if confidence > conf_thresh:
                #convert box coordinates into actual image pixel coordinates
                box = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, width, height = box.astype("int")
                x_min = int(x_center - (width / 2))
                y_min = int(y_center - (height / 2))

                boxes.append([x_min, y_min, int(width), int(height)])
                confidences.append(confidence)
                class_ids.append(class_id)
    #remove duplicate overlapping boxes for the same object using NMS or Non-Maximum suppresion (covered in lecture)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

    detected_items = []
    #generate consistent colors for each class label
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

    #draw the final bounding boxes and text labels onto the image
    if len(indices) > 0:
        for i in np.array(indices).flatten():
            cls_id = class_ids[i]
            cls_name = classes[cls_id] if cls_id < len(classes) else str(cls_id)
            conf = confidences[i]
            x, y, bw, bh = boxes[i]

            detected_items.append((cls_name, conf))

            #draw rectangle and confidence label
            color = [int(c) for c in colors[cls_id]]
            cv2.rectangle(image, (x, y), (x + bw, y + bh), color, 3)
            label_text = f"{cls_name.capitalize()}: {conf:.2f}" 
            cv2.putText(
                image,
                label_text,
                (x, max(y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    return image, inference_time, detected_items


def main():
    args = parse_arguments()

    #create output directory and load model assets
    os.makedirs(args.out_dir, exist_ok=True)

    classes = load_classes(args.labels)
    net, output_layers = load_model(args.cfg, args.weights)
    image_paths = collect_image_paths(args)

    total_inference_time = 0.0
    all_detections_counter = Counter()
    per_image_breakdown = defaultdict(Counter)

    #process each image, save annotated output, and record counts
    for path in image_paths:
        annotated_img, inf_time, detections = process_image(
            path, net, output_layers, classes, args.conf_thresh, args.nms_thresh
        )

        if annotated_img is None:
            continue

        total_inference_time += inf_time

        # Save output image with '_out.jpg' suffix
        base_name = os.path.basename(path)
        name_part, _ = os.path.splitext(base_name)
        out_path = os.path.join(args.out_dir, f"{name_part}_out.jpg")
        cv2.imwrite(out_path, annotated_img)

        #tally detected objects both globally and per image
        for cls_name, _ in detections:
            all_detections_counter[cls_name.capitalize()] += 1
            per_image_breakdown[base_name][cls_name.capitalize()] += 1

    #calculate summary metrics
    avg_inference_time = total_inference_time / len(image_paths)
    unique_classes_count = sum(all_detections_counter.values())

    # print output
    if args.inf and not args.classes_all:
        print(f"Average Inference Time: {avg_inference_time:.6f} seconds")
        return

    if args.classes_all and not args.inf:
        print(f"Total Number of Classes Detected: {unique_classes_count}")
        return

    # Default full report
    print(f"Average Inference Time: {avg_inference_time:.5f} seconds")
    print(f"Total Number of Classes Detected: {unique_classes_count}")
    print("\nTotal Detection Breakdown")
    if all_detections_counter:
        for cls_name, count in sorted(all_detections_counter.items()):
            print(f"{cls_name}: {count}")
    else:
        print("None")

    print("\nPer Image Breakdown")
    for img_name in sorted(per_image_breakdown.keys()):
        counts = per_image_breakdown[img_name]
        breakdown_str = ", ".join(
            [f"{cls}: {cnt}" for cls, cnt in sorted(counts.items())]
        )
        print(f"{img_name} => {breakdown_str if breakdown_str else 'None'}")


if __name__ == "__main__":
    main()


