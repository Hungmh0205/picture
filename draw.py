import cv2
import numpy as np
import json
import time

def draw_from_json_with_animation(json_path, canvas_size=None, delay=1, display_scale=0.5):
    with open(json_path, "r") as f:
        data = json.load(f)

    if canvas_size:
        w, h = canvas_size
    else:
        max_x = max([max(line["start"][0], line["end"][0]) for line in data["lines"]] +
                    [dot["position"][0] for dot in data["dots"]])
        max_y = max([max(line["start"][1], line["end"][1]) for line in data["lines"]] +
                    [dot["position"][1] for dot in data["dots"]])
        w, h = max_x + 20, max_y + 20

    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    window_name = "Replay Drawing (Step-by-Step)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Draw lines step-by-step
    for idx, line in enumerate(data["lines"]):
        pt1 = tuple(line["start"])
        pt2 = tuple(line["end"])
        color = tuple(line["color"])
        thickness = int(line["thickness"])
        cv2.line(canvas, pt1, pt2, color, thickness)

        if idx % 10 == 0:
            display_img = cv2.resize(canvas, (0, 0), fx=display_scale, fy=display_scale)
            cv2.imshow(window_name, display_img)
            if cv2.waitKey(delay) == 27:  # ESC to stop
                break

    # Draw dots step-by-step
    for idx, dot in enumerate(data["dots"]):
        pos = tuple(dot["position"])
        radius = int(dot["radius"])
        color = tuple(dot["color"])
        cv2.circle(canvas, pos, radius, color, -1)

        if idx % 50 == 0:
            display_img = cv2.resize(canvas, (0, 0), fx=display_scale, fy=display_scale)
            cv2.imshow(window_name, display_img)
            if cv2.waitKey(delay) == 27:
                break

    # Show final result
    display_img = cv2.resize(canvas, (0, 0), fx=display_scale, fy=display_scale)
    cv2.imshow(window_name, display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("reconstructed_with_animation.jpg", canvas)
    print("[✓] Reconstructed image saved as: reconstructed_with_animation.jpg")

# Run
draw_from_json_with_animation("drawing_data.json", delay=1)
