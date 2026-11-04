"""
FocusTrack AI
Real-time office activity and work-habit monitoring.

Input: Laptop webcam
Interface: OpenCV window
Models: YOLO11 Pose + YOLO11 Object Detection
Storage: Local CSV and JSON reports
No prerecorded video input. No video recording.

Controls:
    Q / ESC : Quit and export reports
    E       : Export reports now
    R       : Select workstation monitoring region
    X       : Reset workstation region to full camera frame
    M       : Toggle camera mirroring (selfie/standard)
    H       : Toggle HUD dashboard box
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# 1. CONFIGURATION
# ============================================================

APP_NAME = "FocusTrack AI"

POSE_MODEL_NAME = "yolo11n-pose.pt"
OBJECT_MODEL_NAME = "yolo11n.pt"

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MIRROR_CAMERA = True

POSE_IMAGE_SIZE = 512
OBJECT_IMAGE_SIZE = 416

POSE_CONFIDENCE = 0.35
PHONE_CONFIDENCE = 0.25

# Run phone detection once every N frames to reduce computation.
PHONE_DETECTION_INTERVAL = 10

# A person must be absent this long before a departure is confirmed.
AWAY_CONFIRM_SECONDS = 3.0

# A new posture must remain stable this long before being accepted.
POSTURE_CONFIRM_SECONDS = 1.2

# Minimum keypoint confidence for pose-based estimates.
KEYPOINT_CONFIDENCE = 0.25

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

CONFIG_FILE = Path("focustrack_config.json")

WINDOW_NAME = "FocusTrack AI - Live Office Monitor"


# COCO 17-keypoint skeleton.
SKELETON = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

KEYPOINT_NAMES = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


# ============================================================
# 2. GENERAL UTILITIES
# ============================================================

def format_duration(seconds: float) -> str:
    """Convert seconds into HH:MM:SS."""
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def angle_between_points(a, b, c):
    """Calculate the angle ABC in degrees."""
    if a is None or b is None or c is None:
        return None

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-6:
        return None

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def point_inside_roi(point, roi):
    """Check whether a point is inside an ROI."""
    if roi is None:
        return True

    x, y = point
    rx, ry, rw, rh = roi

    return rx <= x <= rx + rw and ry <= y <= ry + rh


def safe_average(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


# ============================================================
# 3. COMPUTER VISION SYSTEM
# ============================================================

class FocusTrackVision:

    def __init__(self, device):
        print("\nLoading YOLO pose model...")
        self.pose_model = YOLO(POSE_MODEL_NAME)

        print("Loading YOLO object detection model...")
        self.object_model = YOLO(OBJECT_MODEL_NAME)

        self.device = device
        self.frame_count = 0

        self.phone_boxes = []

        self.wrist_history = {
            9: deque(maxlen=8),
            10: deque(maxlen=8),
        }

        # Find the COCO cell-phone class.
        names = self.object_model.names

        self.phone_class_ids = [
            class_id
            for class_id, class_name in names.items()
            if "cell phone" in str(class_name).lower()
        ]

        if not self.phone_class_ids:
            # COCO class 67 is cell phone.
            self.phone_class_ids = [67]

        print(f"Using device: {self.device}")
        print(f"Phone class IDs: {self.phone_class_ids}")
        print("Models loaded successfully.\n")

    def get_keypoint(self, xy, confidence, index):
        """Return a keypoint only if its confidence is sufficient."""
        if index >= len(xy):
            return None

        if confidence is not None:
            if index >= len(confidence):
                return None

            if float(confidence[index]) < KEYPOINT_CONFIDENCE:
                return None

        point = xy[index]

        if not np.isfinite(point).all():
            return None

        return (float(point[0]), float(point[1]))

    def estimate_posture(self, points):
        """
        Estimate sitting/standing using knee angles.

        This heuristic works best with a side or three-quarter camera view.
        """
        left_angle = angle_between_points(
            points.get(11),
            points.get(13),
            points.get(15),
        )

        right_angle = angle_between_points(
            points.get(12),
            points.get(14),
            points.get(16),
        )

        knee_angle = safe_average([left_angle, right_angle])

        if knee_angle is None:
            return "UNKNOWN", None

        if knee_angle < 145:
            return "SITTING", knee_angle

        if knee_angle > 155:
            return "STANDING", knee_angle

        return "UNKNOWN", knee_angle

    def estimate_body_alignment(self, points):
        """
        Estimate torso inclination relative to vertical.

        This is an approximate posture indicator, not a medical assessment.
        """
        left_shoulder = points.get(5)
        right_shoulder = points.get(6)
        left_hip = points.get(11)
        right_hip = points.get(12)

        if not all([
            left_shoulder,
            right_shoulder,
            left_hip,
            right_hip,
        ]):
            return None, "UNKNOWN"

        shoulder_mid = (
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2,
        )

        hip_mid = (
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2,
        )

        dx = shoulder_mid[0] - hip_mid[0]
        dy = shoulder_mid[1] - hip_mid[1]

        torso_angle = math.degrees(
            math.atan2(abs(dx), abs(dy) + 1e-6)
        )

        if torso_angle > 28:
            return torso_angle, "LEANING"

        return torso_angle, "UPRIGHT"

    def estimate_head_turn(self, points):
        """
        Estimate possible head turning using nose displacement
        relative to the eye midpoint.

        This does NOT measure gaze or attention.
        """
        nose = points.get(0)
        left_eye = points.get(1)
        right_eye = points.get(2)

        if not all([nose, left_eye, right_eye]):
            return None

        eye_mid_x = (left_eye[0] + right_eye[0]) / 2

        eye_distance = math.dist(left_eye, right_eye)

        if eye_distance < 3:
            return None

        normalized_offset = abs(nose[0] - eye_mid_x) / eye_distance

        return normalized_offset > 0.28

    def estimate_hand_activity(self, points, posture, person_height):
        """
        Infer coarse hand-activity indicators from wrist movement.

        These labels are estimates:
        - POSSIBLE_TYPING
        - POSSIBLE_WRITING
        - LOW_MOTION_POSSIBLE_READING
        - OTHER
        """
        valid_wrists = {}

        for wrist_id in (9, 10):
            wrist = points.get(wrist_id)

            if wrist is not None:
                valid_wrists[wrist_id] = wrist
                self.wrist_history[wrist_id].append(wrist)
            else:
                self.wrist_history[wrist_id].append(None)

        if posture != "SITTING":
            return "OTHER"

        movement = {}

        for wrist_id, history in self.wrist_history.items():
            valid = [p for p in history if p is not None]

            if len(valid) >= 4:
                displacement = math.dist(valid[0], valid[-1])
                movement[wrist_id] = displacement / max(person_height, 1)

        active_hands = sum(
            1 for value in movement.values()
            if value > 0.045
        )

        if active_hands == 2:
            return "POSSIBLE_TYPING"

        if active_hands == 1:
            return "POSSIBLE_WRITING"

        if len(valid_wrists) >= 1:
            return "LOW_MOTION_POSSIBLE_READING"

        return "OTHER"

    def detect_phones(self, frame, roi):
        """Run phone detection periodically."""
        self.frame_count += 1

        if self.frame_count % PHONE_DETECTION_INTERVAL == 0:
            result = self.object_model.predict(
                frame,
                imgsz=OBJECT_IMAGE_SIZE,
                conf=PHONE_CONFIDENCE,
                classes=self.phone_class_ids,
                device=self.device,
                verbose=False,
            )[0]

            self.phone_boxes = []

            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)

                    center = (
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                    )

                    if point_inside_roi(center, roi):
                        self.phone_boxes.append(
                            (x1, y1, x2, y2)
                        )

        return self.phone_boxes

    def process_frame(self, frame, roi):
        """Process one live webcam frame."""
        result = self.pose_model.predict(
            frame,
            imgsz=POSE_IMAGE_SIZE,
            conf=POSE_CONFIDENCE,
            device=self.device,
            verbose=False,
        )[0]

        selected_box = None
        selected_points = {}
        selected_confidence = -1

        if result.boxes is not None and result.keypoints is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            box_confidences = result.boxes.conf.cpu().numpy()

            keypoints_xy = result.keypoints.xy.cpu().numpy()

            keypoint_confidences = None

            if result.keypoints.conf is not None:
                keypoint_confidences = (
                    result.keypoints.conf.cpu().numpy()
                )

            for person_index, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)

                center = (
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                )

                if not point_inside_roi(center, roi):
                    continue

                confidence = float(box_confidences[person_index])

                if confidence <= selected_confidence:
                    continue

                xy = keypoints_xy[person_index]

                kp_conf = None

                if keypoint_confidences is not None:
                    kp_conf = keypoint_confidences[person_index]

                points = {}

                for index in range(17):
                    point = self.get_keypoint(
                        xy,
                        kp_conf,
                        index,
                    )

                    if point is not None:
                        points[index] = point

                selected_box = (x1, y1, x2, y2)
                selected_points = points
                selected_confidence = confidence

        phones = self.detect_phones(frame, roi)

        phone_visible = len(phones) > 0

        if selected_box is None:
            # Do not carry wrist history across a disappearance.
            for history in self.wrist_history.values():
                history.clear()

            return {
                "present": False,
                "box": None,
                "points": {},
                "posture": "UNKNOWN",
                "knee_angle": None,
                "trunk_angle": None,
                "alignment": "UNKNOWN",
                "head_turned": None,
                "activity": "UNKNOWN",
                "phone_visible": phone_visible,
                "phone_boxes": phones,
            }

        x1, y1, x2, y2 = selected_box
        person_height = max(y2 - y1, 1)

        posture, knee_angle = self.estimate_posture(
            selected_points
        )

        trunk_angle, alignment = self.estimate_body_alignment(
            selected_points
        )

        head_turned = self.estimate_head_turn(
            selected_points
        )

        activity = self.estimate_hand_activity(
            selected_points,
            posture,
            person_height,
        )

        return {
            "present": True,
            "box": selected_box,
            "points": selected_points,
            "posture": posture,
            "knee_angle": knee_angle,
            "trunk_angle": trunk_angle,
            "alignment": alignment,
            "head_turned": head_turned,
            "activity": activity,
            "phone_visible": phone_visible,
            "phone_boxes": phones,
        }


# ============================================================
# 4. EVENT TRACKING ENGINE
# ============================================================

class FocusTrackEngine:

    def __init__(self):
        self.session_start = datetime.now()
        self.session_start_mono = time.monotonic()

        self.last_tick = self.session_start_mono

        self.events = []

        self.has_seen_person = False
        self.present = False

        self.absence_since = None
        self.away_since = None

        self.departure_count = 0
        self.return_count = 0

        self.completed_breaks = []

        self.workstation_seconds = 0.0
        self.sitting_seconds = 0.0
        self.standing_seconds = 0.0

        self.phone_visible_seconds = 0.0
        self.head_turned_seconds = 0.0
        self.leaning_seconds = 0.0

        self.continuous_sitting_seconds = 0.0

        self.posture = "UNKNOWN"
        self.posture_candidate = None
        self.posture_candidate_since = None

        self.alignment = "UNKNOWN"
        self.head_turned = False
        self.phone_visible = False

        self.activity = "UNKNOWN"

        self.activity_seconds = {
            "POSSIBLE_TYPING": 0.0,
            "POSSIBLE_WRITING": 0.0,
            "LOW_MOTION_POSSIBLE_READING": 0.0,
            "OTHER": 0.0,
        }

        self.last_event = "Session started"

    def record_event(self, event_name, details=""):
        now = datetime.now()

        event = {
            "timestamp": now.isoformat(timespec="seconds"),
            "event": event_name,
            "details": details,
        }

        self.events.append(event)
        self.last_event = event_name

        print(
            f"[{event['timestamp']}] "
            f"{event_name}: {details}"
        )

    def update_posture(self, raw_posture, now_mono):
        if raw_posture not in ("SITTING", "STANDING"):
            return

        if raw_posture == self.posture:
            self.posture_candidate = None
            self.posture_candidate_since = None
            return

        if raw_posture != self.posture_candidate:
            self.posture_candidate = raw_posture
            self.posture_candidate_since = now_mono
            return

        if (
            self.posture_candidate_since is not None
            and now_mono - self.posture_candidate_since
            >= POSTURE_CONFIRM_SECONDS
        ):
            previous = self.posture
            self.posture = raw_posture

            if self.posture == "SITTING":
                self.continuous_sitting_seconds = 0.0
                self.record_event(
                    "SITTING_STARTED",
                    f"Previous posture: {previous}",
                )

            elif self.posture == "STANDING":
                self.continuous_sitting_seconds = 0.0
                self.record_event(
                    "STANDING_STARTED",
                    f"Previous posture: {previous}",
                )

            self.posture_candidate = None
            self.posture_candidate_since = None

    def update(self, observation, now_mono=None):
        if now_mono is None:
            now_mono = time.monotonic()

        dt = max(
            0.0,
            min(1.0, now_mono - self.last_tick),
        )

        # ----------------------------------------------------
        # Accumulate durations using the previous stable state.
        # ----------------------------------------------------

        if self.present:
            self.workstation_seconds += dt

            if self.posture == "SITTING":
                self.sitting_seconds += dt
                self.continuous_sitting_seconds += dt

            elif self.posture == "STANDING":
                self.standing_seconds += dt
                self.continuous_sitting_seconds = 0.0

        if self.phone_visible:
            self.phone_visible_seconds += dt

        if self.head_turned:
            self.head_turned_seconds += dt

        if self.alignment == "LEANING":
            self.leaning_seconds += dt

        if self.activity in self.activity_seconds:
            self.activity_seconds[self.activity] += dt

        self.last_tick = now_mono

        # ----------------------------------------------------
        # Workstation presence and departure/return tracking.
        # ----------------------------------------------------

        observed_present = observation["present"]

        if observed_present:
            self.absence_since = None

            if not self.present:
                if self.away_since is not None:
                    break_duration = max(
                        0.0,
                        now_mono - self.away_since,
                    )

                    self.completed_breaks.append(
                        break_duration
                    )

                    self.return_count += 1

                    self.record_event(
                        "RETURNED_TO_WORKSTATION",
                        f"Break duration: "
                        f"{format_duration(break_duration)}",
                    )

                    self.away_since = None

                else:
                    self.record_event(
                        "WORKSTATION_DETECTED",
                        "Initial person detection",
                    )

                self.present = True

            self.has_seen_person = True

        else:
            if self.has_seen_person:
                if self.absence_since is None:
                    self.absence_since = now_mono

                if (
                    self.present
                    and now_mono - self.absence_since
                    >= AWAY_CONFIRM_SECONDS
                ):
                    self.present = False

                    # Backdate the break to the first absent frame.
                    self.away_since = self.absence_since

                    self.departure_count += 1

                    self.posture = "UNKNOWN"
                    self.posture_candidate = None
                    self.posture_candidate_since = None

                    self.continuous_sitting_seconds = 0.0

                    self.record_event(
                        "LEFT_WORKSTATION",
                        f"Absence confirmed after "
                        f"{AWAY_CONFIRM_SECONDS:.1f} seconds",
                    )

        # ----------------------------------------------------
        # Posture state machine.
        # ----------------------------------------------------

        if self.present:
            self.update_posture(
                observation.get("posture", "UNKNOWN"),
                now_mono,
            )

        # ----------------------------------------------------
        # Phone visibility.
        # ----------------------------------------------------

        new_phone_state = bool(
            observation.get("phone_visible", False)
        )

        if new_phone_state != self.phone_visible:
            self.phone_visible = new_phone_state

            if self.phone_visible:
                self.record_event(
                    "PHONE_VISIBLE",
                    "Phone detected inside workstation region",
                )
            else:
                self.record_event(
                    "PHONE_NOT_VISIBLE",
                    "Phone detection ended",
                )

        # ----------------------------------------------------
        # Head-turn indicator.
        # ----------------------------------------------------

        new_head_state = observation.get("head_turned")

        if new_head_state is not None:
            new_head_state = bool(new_head_state)

            if new_head_state != self.head_turned:
                self.head_turned = new_head_state

                if self.head_turned:
                    self.record_event(
                        "HEAD_TURN_INDICATOR",
                        "Possible head turn; not a gaze measurement",
                    )
                else:
                    self.record_event(
                        "HEAD_FORWARD_INDICATOR",
                        "Head-turn indicator cleared",
                    )

        # ----------------------------------------------------
        # Posture alignment.
        # ----------------------------------------------------

        new_alignment = observation.get(
            "alignment",
            "UNKNOWN",
        )

        if new_alignment != "UNKNOWN":
            if new_alignment != self.alignment:
                self.alignment = new_alignment

                self.record_event(
                    "BODY_ALIGNMENT_CHANGED",
                    new_alignment,
                )

        # ----------------------------------------------------
        # Activity indicators.
        # ----------------------------------------------------

        new_activity = observation.get(
            "activity",
            "UNKNOWN",
        )

        if new_activity != self.activity:
            self.activity = new_activity

            if new_activity != "UNKNOWN":
                self.record_event(
                    "ACTIVITY_INDICATOR_CHANGED",
                    new_activity,
                )

    def get_summary(self):
        now = datetime.now()
        now_mono = time.monotonic()

        session_duration = max(
            0.0,
            now_mono - self.session_start_mono,
        )

        break_durations = list(self.completed_breaks)

        if self.away_since is not None:
            break_durations.append(
                max(0.0, now_mono - self.away_since)
            )

        total_break_seconds = sum(break_durations)

        average_break = (
            total_break_seconds / len(break_durations)
            if break_durations else 0.0
        )

        longest_break = (
            max(break_durations)
            if break_durations else 0.0
        )

        return {
            "session_date": self.session_start.date().isoformat(),
            "session_start": self.session_start.isoformat(
                timespec="seconds"
            ),
            "session_end": now.isoformat(timespec="seconds"),

            "session_duration_seconds": round(
                session_duration, 2
            ),

            "workstation_seconds": round(
                self.workstation_seconds, 2
            ),

            "sitting_seconds": round(
                self.sitting_seconds, 2
            ),

            "standing_seconds": round(
                self.standing_seconds, 2
            ),

            "away_seconds": round(
                total_break_seconds, 2
            ),

            "break_count": self.departure_count,
            "return_count": self.return_count,

            "average_break_seconds": round(
                average_break, 2
            ),

            "longest_break_seconds": round(
                longest_break, 2
            ),

            "continuous_sitting_seconds": round(
                self.continuous_sitting_seconds, 2
            ),

            "phone_visible_seconds": round(
                self.phone_visible_seconds, 2
            ),

            "head_turned_seconds": round(
                self.head_turned_seconds, 2
            ),

            "leaning_seconds": round(
                self.leaning_seconds, 2
            ),

            "activity_seconds": {
                key: round(value, 2)
                for key, value in self.activity_seconds.items()
            },

            "event_count": len(self.events),

            "current_workstation_state": (
                "PRESENT" if self.present else "AWAY"
            ),

            "current_posture": self.posture,
        }


# ============================================================
# 5. REPORT EXPORT
# ============================================================

def read_session_reports():
    summaries = []

    for path in REPORT_DIR.glob("session_*.json"):
        try:
            with path.open("r", encoding="utf-8") as file:
                summaries.append(json.load(file))
        except (OSError, json.JSONDecodeError):
            continue

    return summaries


def aggregate_reports(summaries, period_name, start_date, end_date):
    selected = []

    for item in summaries:
        try:
            session_date = datetime.fromisoformat(
                item["session_date"]
            ).date()
        except (KeyError, ValueError):
            continue

        if start_date <= session_date <= end_date:
            selected.append(item)

    numeric_fields = [
        "session_duration_seconds",
        "workstation_seconds",
        "sitting_seconds",
        "standing_seconds",
        "away_seconds",
        "break_count",
        "return_count",
        "phone_visible_seconds",
        "head_turned_seconds",
        "leaning_seconds",
    ]

    totals = {
        field: round(
            sum(float(item.get(field, 0)) for item in selected),
            2,
        )
        for field in numeric_fields
    }

    activity_totals = {}

    for item in selected:
        for activity, seconds in item.get(
            "activity_seconds", {}
        ).items():
            activity_totals[activity] = (
                activity_totals.get(activity, 0.0)
                + float(seconds)
            )

    break_count = totals["break_count"]

    totals["average_break_seconds"] = round(
        totals["away_seconds"] / break_count,
        2,
    ) if break_count else 0.0

    totals["longest_break_seconds"] = round(
        max(
            (
                float(item.get("longest_break_seconds", 0))
                for item in selected
            ),
            default=0.0,
        ),
        2,
    )

    return {
        "period": period_name,
        "period_start": start_date.isoformat(),
        "period_end": end_date.isoformat(),
        "sessions": len(selected),
        "totals": totals,
        "activity_seconds": {
            key: round(value, 2)
            for key, value in activity_totals.items()
        },
    }


def export_reports(engine):
    """Export session events, session summary, daily and weekly JSON."""
    REPORT_DIR.mkdir(exist_ok=True)

    summary = engine.get_summary()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    event_path = REPORT_DIR / f"events_{timestamp}.csv"
    session_path = REPORT_DIR / f"session_{timestamp}.json"

    # Event CSV.
    with event_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["timestamp", "event", "details"],
        )

        writer.writeheader()
        writer.writerows(engine.events)

    # Session JSON.
    with session_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # Aggregate the current session with earlier sessions.
    all_sessions = read_session_reports()

    today = datetime.now().date()

    daily = aggregate_reports(
        all_sessions,
        "daily",
        today,
        today,
    )

    daily_path = REPORT_DIR / f"daily_summary_{today.isoformat()}.json"

    with daily_path.open("w", encoding="utf-8") as file:
        json.dump(daily, file, indent=4)

    week_start = today - timedelta(days=6)

    weekly = aggregate_reports(
        all_sessions,
        "last_7_days",
        week_start,
        today,
    )

    weekly_path = (
        REPORT_DIR / f"weekly_summary_{today.isoformat()}.json"
    )

    with weekly_path.open("w", encoding="utf-8") as file:
        json.dump(weekly, file, indent=4)

    print("\n========== REPORTS EXPORTED ==========")
    print(f"Events:          {event_path}")
    print(f"Session summary: {session_path}")
    print(f"Daily summary:   {daily_path}")
    print(f"Weekly summary:  {weekly_path}")
    print("======================================\n")

    return {
        "events": event_path,
        "session": session_path,
        "daily": daily_path,
        "weekly": weekly_path,
    }


# ============================================================
# 6. OPENCV DASHBOARD
# ============================================================

def draw_text(
    frame,
    text,
    x,
    y,
    color=(255, 255, 255),
    scale=0.55,
    thickness=1,
):
    cv2.putText(
        frame,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_pose(frame, points):
    """Draw available pose landmarks and skeleton."""
    for start, end in SKELETON:
        if start in points and end in points:
            p1 = tuple(map(int, points[start]))
            p2 = tuple(map(int, points[end]))

            cv2.line(
                frame,
                p1,
                p2,
                (0, 220, 255),
                2,
            )

    for point in points.values():
        x, y = map(int, point)

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1,
        )


def draw_dashboard(
    frame,
    observation,
    engine,
    roi,
    fps,
    show_hud=True,
):
    h, w = frame.shape[:2]

    # Workstation region.
    if roi is not None:
        rx, ry, rw, rh = roi

        cv2.rectangle(
            frame,
            (rx, ry),
            (rx + rw, ry + rh),
            (255, 200, 0),
            2,
        )

        draw_text(
            frame,
            "WORKSTATION REGION",
            rx + 5,
            max(ry - 8, 20),
            (255, 200, 0),
            0.42,
        )

    # Person bounding box and pose.
    if observation["present"] and observation["box"]:
        x1, y1, x2, y2 = observation["box"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        draw_text(
            frame,
            "PERSON",
            x1,
            max(y1 - 8, 18),
            (0, 255, 0),
            0.42,
        )

        draw_pose(frame, observation["points"])

    # Phone bounding boxes.
    for x1, y1, x2, y2 in observation["phone_boxes"]:
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        draw_text(
            frame,
            "PHONE",
            x1,
            max(y1 - 8, 18),
            (0, 0, 255),
            0.42,
        )

    # Compact Dashboard panel.
    if show_hud:
        panel_width = min(215, w - 16)
        panel_height = min(232, h - 16)

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (8, 8),
            (8 + panel_width, 8 + panel_height),
            (15, 20, 30),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.75,
            frame,
            0.25,
            0,
            frame,
        )

        # Border for clean modern appearance.
        cv2.rectangle(
            frame,
            (8, 8),
            (8 + panel_width, 8 + panel_height),
            (55, 68, 85),
            1,
        )

        summary = engine.get_summary()

        x = 16
        y = 23
        gap = 15

        draw_text(
            frame,
            "FOCUSTRACK AI | MONITOR",
            x,
            y,
            (0, 220, 255),
            0.40,
            1,
        )

        y += gap + 2

        workstation_state = (
            "PRESENT" if engine.present else "AWAY"
        )

        draw_text(
            frame,
            f"Workstation: {workstation_state}",
            x,
            y,
            (0, 255, 0) if engine.present else (0, 120, 255),
            0.36,
        )

        y += gap

        draw_text(
            frame,
            f"Posture: {engine.posture}",
            x,
            y,
            scale=0.36,
        )

        y += gap

        draw_text(
            frame,
            f"Alignment: {engine.alignment}",
            x,
            y,
            scale=0.36,
        )

        y += gap

        draw_text(
            frame,
            f"Activity: {engine.activity}",
            x,
            y,
            scale=0.36,
        )

        y += gap

        draw_text(
            frame,
            f"Workstation: "
            f"{format_duration(summary['workstation_seconds'])}",
            x,
            y,
            scale=0.36,
        )

        y += gap

        draw_text(
            frame,
            f"Sitting: "
            f"{format_duration(summary['sitting_seconds'])}",
            x,
            y,
            scale=0.36,
        )

        y += gap

        draw_text(
            frame,
            f"Cont. sitting: "
            f"{format_duration(engine.continuous_sitting_seconds)}",
            x,
            y,
            scale=0.36,
        )

        y += gap

        draw_text(
            frame,
            f"Standing: "
            f"{format_duration(summary['standing_seconds'])}",
            x,
            y,
            scale=0.36,
        )

        y += gap

        draw_text(
            frame,
            f"Breaks: "
            f"{engine.departure_count} / {engine.return_count} "
            f"({format_duration(summary['away_seconds'])})",
            x,
            y,
            scale=0.36,
        )

        y += gap

        draw_text(
            frame,
            f"Phone visible: "
            f"{format_duration(engine.phone_visible_seconds)}",
            x,
            y,
            (0, 150, 255) if engine.phone_visible else (220, 220, 220),
            scale=0.36,
        )

        y += gap

        draw_text(
            frame,
            f"Head-turn: "
            f"{format_duration(engine.head_turned_seconds)}",
            x,
            y,
            scale=0.36,
        )

        y += gap

        draw_text(
            frame,
            f"Leaning: "
            f"{format_duration(engine.leaning_seconds)}",
            x,
            y,
            scale=0.36,
        )

        y += gap

        draw_text(
            frame,
            f"FPS: {fps:.1f}",
            x,
            y,
            (0, 220, 255),
            scale=0.36,
        )

    # Controls at the bottom.
    controls = "Q: Quit | E: Export | R: ROI | X: Reset | M: Mirror | H: HUD"

    cv2.rectangle(
        frame,
        (8, h - 28),
        (w - 8, h - 6),
        (20, 20, 20),
        -1,
    )

    draw_text(
        frame,
        controls,
        14,
        h - 13,
        (220, 220, 220),
        0.36,
    )

    # Recent events.
    recent = engine.events[-2:]

    if recent:
        event_y = h - 36

        for event in reversed(recent):
            draw_text(
                frame,
                f"{event['timestamp'][11:19]} "
                f"{event['event']}",
                max(w - 330, 10),
                event_y,
                (255, 255, 0),
                0.36,
            )

            event_y -= 16

    return frame


# ============================================================
# 7. CAMERA AND ROI CONFIGURATION
# ============================================================

def open_camera():
    """Open the laptop webcam, preferring the Windows DirectShow backend."""
    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check camera permissions "
            "and make sure another application is not using it."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # Request a reasonable capture rate.
    camera.set(cv2.CAP_PROP_FPS, 30)

    return camera


def load_roi():
    """Load the previously saved ROI, if available."""
    if not CONFIG_FILE.exists():
        return None

    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as file:
            config = json.load(file)

        roi = config.get("roi")

        if roi and len(roi) == 4:
            return tuple(map(int, roi))

    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass

    return None


def save_roi(roi):
    if roi is None:
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
        return

    with CONFIG_FILE.open("w", encoding="utf-8") as file:
        json.dump({"roi": list(roi)}, file, indent=4)


# ============================================================
# 8. MAIN APPLICATION
# ============================================================

def main():
    print("\n======================================")
    print("       FOCUSTRACK AI")
    print(" Real-Time Office Activity Monitor")
    print("======================================")

    if torch.cuda.is_available():
        device = 0
        print("NVIDIA CUDA GPU detected.")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA GPU not detected. Using CPU.")

    print("\nInitializing camera...")

    camera = open_camera()

    ret, first_frame = camera.read()

    if not ret:
        camera.release()
        raise RuntimeError("Webcam opened but no frame was received.")

    mirror_camera = MIRROR_CAMERA
    show_hud = True

    if mirror_camera:
        first_frame = cv2.flip(first_frame, 1)

    frame_height, frame_width = first_frame.shape[:2]

    roi = load_roi()

    # Validate a saved ROI against the current camera dimensions.
    if roi is not None:
        rx, ry, rw, rh = roi

        if (
            rx < 0 or ry < 0
            or rw <= 0 or rh <= 0
            or rx + rw > frame_width
            or ry + rh > frame_height
        ):
            roi = None

    if roi is None:
        roi = (0, 0, frame_width, frame_height)

    vision = FocusTrackVision(device)
    engine = FocusTrackEngine()

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        1280,
        720,
    )

    print("\nLive monitoring started.")
    print("Press R to select the workstation region.")
    print("Press E to export reports.")
    print("Press X to reset the region.")
    print("Press M to toggle camera mirror (selfie mode).")
    print("Press H to toggle HUD dashboard box.")
    print("Press Q or ESC to stop.\n")

    frame_counter = 0
    fps = 0.0
    fps_start = time.monotonic()
    fps_frames = 0

    try:
        while True:
            ret, frame = camera.read()

            if not ret:
                print("Warning: Webcam frame could not be read.")
                time.sleep(0.05)
                continue

            if mirror_camera:
                frame = cv2.flip(frame, 1)

            now_mono = time.monotonic()

            observation = vision.process_frame(
                frame,
                roi,
            )

            engine.update(
                observation,
                now_mono,
            )

            # FPS calculation.
            fps_frames += 1
            elapsed = now_mono - fps_start

            if elapsed >= 1.0:
                fps = fps_frames / elapsed
                fps_frames = 0
                fps_start = now_mono

            display = draw_dashboard(
                frame.copy(),
                observation,
                engine,
                roi,
                fps,
                show_hud=show_hud,
            )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            # Quit (Q, ESC, or window close button).
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            # Export.
            elif key == ord("e"):
                export_reports(engine)

            # Toggle HUD box.
            elif key == ord("h"):
                show_hud = not show_hud
                print(f"HUD box: {'Visible' if show_hud else 'Hidden'}")

            # Toggle camera mirror.
            elif key == ord("m"):
                mirror_camera = not mirror_camera
                print(f"Camera mirroring: {'Enabled' if mirror_camera else 'Disabled'}")

            # Select a workstation ROI.
            elif key == ord("r"):
                print(
                    "\nSelect the workstation region with your mouse."
                )

                selected = cv2.selectROI(
                    WINDOW_NAME,
                    frame,
                    fromCenter=False,
                    showCrosshair=True,
                )

                rx, ry, rw, rh = map(int, selected)

                if rw > 0 and rh > 0:
                    roi = (rx, ry, rw, rh)
                    save_roi(roi)

                    print(f"New workstation ROI: {roi}")

                else:
                    print("ROI selection cancelled.")

            # Reset ROI.
            elif key == ord("x"):
                roi = (0, 0, frame_width, frame_height)
                save_roi(None)

                print(
                    "ROI reset to the full camera frame."
                )

            frame_counter += 1

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")

    finally:
        print("\nStopping camera...")

        camera.release()
        cv2.destroyAllWindows()

        print("Exporting final reports...")
        export_reports(engine)

        print("\nFinal session summary:")

        summary = engine.get_summary()

        print(
            f"Workstation time: "
            f"{format_duration(summary['workstation_seconds'])}"
        )

        print(
            f"Sitting time: "
            f"{format_duration(summary['sitting_seconds'])}"
        )

        print(
            f"Standing time: "
            f"{format_duration(summary['standing_seconds'])}"
        )

        print(
            f"Breaks: {summary['break_count']}"
        )

        print(
            f"Phone visible: "
            f"{format_duration(summary['phone_visible_seconds'])}"
        )

        print("\nFocusTrack AI stopped successfully.")


if __name__ == "__main__":
    main()