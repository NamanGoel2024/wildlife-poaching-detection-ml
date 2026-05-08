import os
import time
import math
import csv
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ================= CONFIG =================
INPUT_SOURCE = 0  # 0 for webcam OR "sample_video.mp4"
MODEL = "yolov8n.pt"

CONF_THRESH = 0.15
IMG_SIZE = 640
NEAR_PIX = 120

OUTPUT_DIR = "alerts"
LOG_CSV = "events.csv"

ALERT_PERSIST = 3
ALERT_COOLDOWN_SECONDS = 5
TARGET_INFER_FPS = 4.0

WEAPON_LABELS = {"knife", "gun", "pistol", "rifle", "scissors"}

# ================= STATE =================
tracker = DeepSort(
    max_age=30,
    n_init=5,
    max_iou_distance=0.7
)

track_time = {}

persist_count = 0
last_alert_time = 0.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= HELPERS =================
def box_center(bb):

    x1, y1, x2, y2 = bb

    return (
        (x1 + x2) // 2,
        (y1 + y2) // 2
    )

def dist(a, b):

    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )

# ================= IOU =================
def iou(a, b):

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])

    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (area_a + area_b - inter)

# ================= NMS =================
def nms(boxes, thresh=0.3):

    boxes = sorted(
        boxes,
        key=lambda x: x[1],
        reverse=True
    )

    keep = []

    for bb, conf in boxes:

        if all(iou(bb, kb) < thresh for kb, _ in keep):

            keep.append((bb, conf))

    return keep

# ================= SAVE EVIDENCE =================
def save_evidence(frame, frame_id):

    filename = (
        f"{OUTPUT_DIR}/alert_"
        f"{frame_id}_{int(time.time())}.jpg"
    )

    cv2.imwrite(filename, frame)

    with open(LOG_CSV, "a", newline="") as f:

        csv.writer(f).writerow([
            time.time(),
            frame_id,
            filename
        ])

# ================= INFERENCE =================
def run_inference(model, frame, frame_id):

    global persist_count
    global last_alert_time

    result = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        verbose=False
    )[0]

    persons = []
    weapons = []

    # ================= DETECTIONS =================
    for bb, cls, conf in zip(
        result.boxes.xyxy,
        result.boxes.cls,
        result.boxes.conf
    ):

        x1, y1, x2, y2 = map(
            int,
            bb.tolist()
        )

        label = model.names[int(cls)]

        # PERSON
        if label == "person":

            persons.append(
                ((x1, y1, x2, y2), float(conf))
            )

        # WEAPON
        elif label in WEAPON_LABELS or label == "cell phone":

            weapons.append(
                ((x1, y1, x2, y2), float(conf))
            )

    persons = nms(persons)

    # ================= TRACKING =================
    detections = []

    for (x1, y1, x2, y2), conf in persons:

        detections.append(
            (
                [x1, y1, x2 - x1, y2 - y1],
                conf,
                "person"
            )
        )

    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    tracked_persons = []

    for track in tracks:

        if not track.is_confirmed():
            continue

        l, t, r, b = map(
            int,
            track.to_ltrb()
        )

        tracked_persons.append(
            ((l, t, r, b), track.track_id)
        )

    # ================= LOITERING =================
    current_time = time.time()

    loiter_ids = []

    for _, tid in tracked_persons:

        if tid not in track_time:

            track_time[tid] = current_time

        if current_time - track_time[tid] > 10:

            loiter_ids.append(tid)

    # ================= RISK SCORE =================
    risk = 0

    if tracked_persons:
        risk += 10

    if weapons:
        risk += 40

    if len(tracked_persons) > 1:
        risk += 15

    if loiter_ids:
        risk += 30

    # ================= ALERT CHECK =================
    alert = False

    for wbox, _ in weapons:

        wc = box_center(wbox)

        for pbox, _ in tracked_persons:

            pc = box_center(pbox)

            if dist(wc, pc) < NEAR_PIX:

                alert = True
                break

        if alert:
            break

    persist_count = (
        persist_count + 1
        if alert else 0
    )

    now = time.time()

    status = ""

    # ================= STATUS =================
    if (
        persist_count >= ALERT_PERSIST
        and
        (now - last_alert_time)
        > ALERT_COOLDOWN_SECONDS
    ):

        status = (
            "HIGH RISK | "
            "WEAPON DETECTED"
        )

        last_alert_time = now

        persist_count = 0

        save_evidence(frame, frame_id)

    elif weapons and risk > 40:

        status = (
            "HIGH RISK | "
            "WEAPON DETECTED"
        )

    elif risk > 60:

        status = "HIGH RISK"

    elif risk > 30:

        status = "MEDIUM RISK"

    return (
        tracked_persons,
        weapons,
        status
    )

# ================= DRAW =================
def draw(
    frame,
    tracked_persons,
    weapons,
    status
):

    # ================= PERSONS =================
    for (x1, y1, x2, y2), tid in tracked_persons:

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"ID:{tid}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # ================= WEAPONS =================
    for (x1, y1, x2, y2), conf in weapons:

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        label = f"WEAPON {int(conf * 100)}%"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # ================= STATUS BAR =================
    if status:

        cv2.rectangle(
            frame,
            (0, 0),
            (850, 50),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            status,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

# ================= MAIN =================
def main():

    model = YOLO(MODEL)

    cap = cv2.VideoCapture(INPUT_SOURCE)

    frame_id = 0

    last_time = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        now = time.time()

        # ================= FPS CONTROL =================
        if (
            now - last_time
            < 1.0 / TARGET_INFER_FPS
        ):

            time.sleep(0.01)
            continue

        last_time = now

        frame_id += 1

        tracked_persons, weapons, status = run_inference(
            model,
            frame,
            frame_id
        )

        draw(
            frame,
            tracked_persons,
            weapons,
            status
        )

        cv2.imshow(
            "AI Wildlife Poaching Detection System",
            frame
        )

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()

# ================= START =================
if __name__ == "__main__":

    main()