# 🦁 Wildlife Poaching Detection System

A real-time AI-powered surveillance system that detects potential poaching activity using **object detection + behavior analysis + multi-object tracking**.

Built with **YOLOv8**, **DeepSORT**, and **OpenCV**.

---

## 🚀 What's New (Latest Update)

* ✅ **DeepSORT Tracking** → Stable person IDs across frames
* ✅ **Behavior-Based Detection** → Detects suspicious activity even without visible weapons
* ✅ **Loitering Detection** → Identifies prolonged presence in monitored areas
* ✅ **Risk Scoring System** → Multi-factor decision instead of simple rule-based alerts
* ✅ **Improved NMS Filtering** → Removes duplicate detections
* ✅ **Dual Alert System**:

  * Weapon-based alert
  * Behavior-based high-risk alert

---

## 🎯 How It Works

1. Each frame is processed using YOLOv8 to detect:

   * `person`
   * weapons (`knife`, `gun`, `pistol`, `rifle`, `scissors`)
2. Person detections are passed to **DeepSORT** to assign unique IDs
3. The system tracks:

   * Movement over time
   * Number of people
   * Duration of presence (loitering)
4. A **risk score** is calculated based on:

   * Human presence
   * Weapon detection
   * Group activity
   * Loitering behavior
5. Alerts are triggered when:

   * Weapon is detected near a person
   * OR risk score crosses threshold

---

## 🧠 Risk Scoring Logic

```text
Risk Score = 
    +10 (person detected)
    +40 (weapon detected)
    +15 (multiple people)
    +30 (loitering)
```

* `> 60` → 🚨 HIGH RISK
* `> 30` → ⚠ MEDIUM RISK

---

## 📸 Demo

| Normal              | Detection             |
| ------------------- | --------------------- |
| Green Box = Person  | Red Box = Weapon      |
| ID Tracking Enabled | Risk Alerts Displayed |

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/poach-demo.git
cd poach-demo
```

### 2. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install deep-sort-realtime
```

---

### 4. Run

```bash
python detect.py
```

Press **`Q`** to quit.

---

## ⚙️ Configuration

Edit the `CONFIG` section in `detect.py`:

| Variable           | Description                |
| ------------------ | -------------------------- |
| `INPUT_SOURCE`     | Webcam (`0`) or video file |
| `MODEL`            | YOLOv8 model (n/s/m)       |
| `CONF_THRESH`      | Detection confidence       |
| `NEAR_PIX`         | Distance for weapon alert  |
| `TARGET_INFER_FPS` | Processing speed           |
| `ALERT_PERSIST`    | Frames for confirmation    |

---

## 📁 Project Structure

```
poach-demo/
├── detect.py
├── requirements.txt
├── README.md
├── alerts/
└── events.csv
```

---

## 📊 Alerts

System logs:

* Alert frames (images)
* CSV logs with timestamps

---

## 🧠 Technologies Used

* YOLOv8 (Object Detection)
* DeepSORT (Tracking)
* OpenCV (Video Processing)
* Python

---

## 🏗️ Architecture

```
Video Input
    ↓
YOLOv8 Detection
    ↓
NMS Filtering
    ↓
DeepSORT Tracking (IDs)
    ↓
Behavior Analysis
    ↓
Risk Scoring
    ↓
Alert System
    ↓
Logging + Display
```

---

## 💡 Key Advantages

* Works even if weapon is hidden
* Real-time processing
* Behavior-aware detection
* Scalable to surveillance systems

---

## 👨‍💻 Author

**Naman Goel**
B.Tech CSE | GNIOT
[GitHub](https://github.com/YOUR_USERNAME)

---

## 📄 License

MIT License
