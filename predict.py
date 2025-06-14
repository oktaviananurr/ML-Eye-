from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import time

model = YOLO("runs/train/YOLOv8_EYE_DISEASES/weights/best.pt")

def predict(image_path, ground_truth_label=None):
    start = time.time()
    results = model(image_path)
    inference_time = time.time() - start
    fps = 1.0 / inference_time if inference_time > 0 else 0.0

    boxes = results[0].boxes
    names = results[0].names

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, figsize=(10, 5))
    ax.imshow(img)
    ax.axis("off")

    if len(boxes) == 0:
        ax.set_title("No Detection", fontsize=14)
        plt.show()
        return

    cls_id = int(boxes.cls[0])
    conf = float(boxes.conf[0])
    pred_label = names[cls_id]

    if ground_truth_label:
        is_correct = pred_label.lower() == ground_truth_label.lower()
        title_status = "✓ BENAR" if is_correct else "✗ SALAH"
        color = "green" if is_correct else "red"
    else:
        title_status = ""
        color = "blue"

    # Draw box
    x1, y1, x2, y2 = boxes.xyxy[0].cpu().numpy()
    box = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                            linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(box)

    # Title
    ax.set_title(f"{pred_label} ({conf*100:.2f}%) - {title_status}\n🕒 {inference_time:.2f}s ({fps:.1f} FPS)",
                 fontsize=12, color=color)

    plt.tight_layout()
    plt.show()

# Jalankan
predict("eye_diseases_detect-1/test/images/image-7_jpeg_jpg.rf.ed64c6d73158b8b2c659e2ef216f0ac7.jpg", ground_truth_label="Bulging_Eyes")

