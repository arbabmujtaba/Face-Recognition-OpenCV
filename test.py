
import cv2
import time
import numpy as np
import psutil
import json
import platform
from datetime import datetime

CAMERA_INDEX = 1
TEST_DURATION = 180
WIDTH = 640
HEIGHT = 480

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit()

print("Camera opened successfully!")
print("Starting 3-minute hardware diagnostic...")

print("Backend:", cap.getBackendName())
print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))

start_time = time.time()
last_frame_time = start_time
frame_count = 0
failed_frames = 0
frozen_frames = 0
frame_gaps = []
read_times = []
previous_frame = None
last_report = start_time

report = {
    "system": platform.platform(),
    "camera": CAMERA_INDEX,
    "start": datetime.now().isoformat(),
    "events": []
}

while time.time() - start_time < TEST_DURATION:

    read_start = time.perf_counter()
    success, frame = cap.read()
    read_end = time.perf_counter()

    read_time = (read_end - read_start) * 1000
    read_times.append(read_time)

    current_time = time.time()

    if not success or frame is None:
        failed_frames += 1
        print("CAMERA READ FAILED!")
        report["events"].append({
            "time": round(current_time - start_time, 2),
            "event": "read_failed"
        })
        time.sleep(0.1)
        continue

    frame_count += 1

    gap = current_time - last_frame_time
    frame_gaps.append(gap)
    last_frame_time = current_time

    if gap > 1:
        print(f"WARNING: Frame gap of {gap:.2f} seconds")
        report["events"].append({
            "time": round(current_time - start_time, 2),
            "event": "frame_gap",
            "seconds": round(gap, 3)
        })

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 48))

    if previous_frame is not None:
        difference = cv2.absdiff(small, previous_frame).mean()

        if difference < 0.01:
            frozen_frames += 1
        else:
            frozen_frames = 0

        if frozen_frames == 30:
            print("WARNING: 30 identical frames detected!")
            report["events"].append({
                "time": round(current_time - start_time, 2),
                "event": "identical_frames"
            })

    previous_frame = small.copy()

    elapsed = current_time - start_time
    fps = frame_count / elapsed
    age = (time.time() - current_time) * 1000

    display = frame.copy()

    cv2.putText(
        display, f"FPS: {fps:.1f}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2
    )

    cv2.putText(
        display, f"Frames: {frame_count}",
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2
    )

    cv2.putText(
        display, f"Failures: {failed_frames}",
        (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 0, 255), 2
    )

    cv2.putText(
        display, f"Read: {read_time:.1f} ms",
        (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2
    )

    cv2.imshow("Webcam Hardware Test", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

    if current_time - last_report >= 5:
        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory().percent

        print(
            f"Time: {elapsed:.1f}s | "
            f"FPS: {fps:.1f} | "
            f"CPU: {cpu}% | "
            f"RAM: {memory}% | "
            f"Failures: {failed_frames}"
        )

        last_report = current_time

cap.release()
cv2.destroyAllWindows()

duration = time.time() - start_time

report["duration_seconds"] = round(duration, 2)
report["total_frames"] = frame_count
report["failed_frames"] = failed_frames
report["average_fps"] = round(frame_count / max(duration, 0.001), 2)
report["average_read_ms"] = (
    round(float(np.mean(read_times)), 2)
    if read_times else None
)
report["max_read_ms"] = (
    round(float(np.max(read_times)), 2)
    if read_times else None
)
report["max_frame_gap_seconds"] = (
    round(max(frame_gaps), 3)
    if frame_gaps else None
)

with open("webcam_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("\n========== TEST COMPLETE ==========")
print("Duration:", round(duration, 2), "seconds")
print("Total frames:", frame_count)
print("Failed frames:", failed_frames)
print("Average FPS:", report["average_fps"])
print("Average read time:", report["average_read_ms"], "ms")
print("Maximum read time:", report["max_read_ms"], "ms")
print("Maximum frame gap:", report["max_frame_gap_seconds"], "seconds")
print("Report saved to webcam_report.json")