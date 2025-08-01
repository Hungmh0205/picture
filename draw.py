import cv2
import numpy as np
import json

def load_and_draw_from_data(
    json_path="sketch_data.json",
    canvas_size=None,
    delay=1,
    display_scale=0.5,
    line_batch_size=100,
    dot_batch_size=300
):
    with open(json_path, "r") as f:
        data = json.load(f)

    max_x, max_y = 0, 0
    for line in data["lines"]:
        max_x = max(max_x, line["pt1"][0], line["pt2"][0])
        max_y = max(max_y, line["pt1"][1], line["pt2"][1])
    for dot in data["dots"]:
        max_x = max(max_x, dot["position"][0])
        max_y = max(max_y, dot["position"][1])

    if canvas_size:
        w, h = canvas_size
    else:
        w, h = max_x + 10, max_y + 10

    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    window_name = "Truong Tieu Pham"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[..] Drawing lines...")
    line_counter = 0
    for i, line in enumerate(data["lines"]):
        pt1 = tuple(line["pt1"])
        pt2 = tuple(line["pt2"])
        color = tuple(line["color"])
        thickness = line.get("thickness", 1)
        cv2.line(canvas, pt1, pt2, color, thickness)

        line_counter += 1
        if line_counter % line_batch_size == 0 or i == len(data["lines"]) - 1:
            display_img = cv2.resize(canvas, (0, 0), fx=display_scale, fy=display_scale)
            cv2.imshow(window_name, display_img)
            if cv2.waitKey(delay) == 27:
                cv2.destroyAllWindows()
                return

    print("[..] Drawing patches...")
    dot_counter = 0
    for i, dot in enumerate(data["dots"]):
        x, y = dot["position"]
        size = dot.get("patch_size", dot.get("radius", 1))
        color = tuple(dot["color"])
        half = size
        top_left = (x - half, y - half)
        bottom_right = (x + half, y + half)
        cv2.rectangle(canvas, top_left, bottom_right, color, -1)

        dot_counter += 1
        if dot_counter % dot_batch_size == 0 or i == len(data["dots"]) - 1:
            display_img = cv2.resize(canvas, (0, 0), fx=display_scale, fy=display_scale)
            cv2.imshow(window_name, display_img)
            if cv2.waitKey(delay) == 27:
                cv2.destroyAllWindows()
                return

    print("[✔] Replay complete.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

load_and_draw_from_data("sketch_data.json", delay=1)
