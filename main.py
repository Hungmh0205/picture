import cv2
import numpy as np
import random
import os
import json

def draw_image_with_sketch_color_and_export(
    image_path,
    scale_factor=2,
    dot_spacing=1,  # vẫn có mặt nhưng không còn dùng
    patch_size_base=1,
    max_patch_size=3,
    intensity_threshold=230,
    delay=0,
    display_scale=0.3,
    batch_size=500,
    line_jitter=1,
    line_thickness=2,
    line_step=2,
    line_batch_size=100,
    canny_thresh1=50,
    canny_thresh2=120
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"[>>] Loading image: {image_path}")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_colored = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    resized_gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    h, w = resized_gray.shape

    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    window_name = "Color Sketch"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    data = {
        "lines": [],
        "dots": []
    }

    print("[..] Detecting edges...")
    blurred = cv2.GaussianBlur(resized_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    line_count = 0
    for contour in contours:
        for i in range(line_step, len(contour), line_step):
            pt1 = contour[i - line_step][0]
            pt2 = contour[i][0]

            jitter_pt1 = (
                np.clip(pt1[0] + random.randint(-line_jitter, line_jitter), 0, w - 1),
                np.clip(pt1[1] + random.randint(-line_jitter, line_jitter), 0, h - 1)
            )
            jitter_pt2 = (
                np.clip(pt2[0] + random.randint(-line_jitter, line_jitter), 0, w - 1),
                np.clip(pt2[1] + random.randint(-line_jitter, line_jitter), 0, h - 1)
            )

            color = img_colored[jitter_pt1[1], jitter_pt1[0]].tolist()
            cv2.line(canvas, jitter_pt1, jitter_pt2, color, thickness=line_thickness)

            data["lines"].append({
                "pt1": [int(jitter_pt1[0]), int(jitter_pt1[1])],
                "pt2": [int(jitter_pt2[0]), int(jitter_pt2[1])],
                "color": [int(c) for c in color],
                "thickness": int(line_thickness)
            })

            line_count += 1
            if line_count % line_batch_size == 0:
                display_img = cv2.resize(canvas, (0, 0), fx=display_scale, fy=display_scale)
                cv2.imshow(window_name, display_img)
                if cv2.waitKey(1 if delay <= 0 else delay) == 27:
                    cv2.destroyAllWindows()
                    return

    print(f"[**] Total lines drawn: {line_count}")

    dot_count = 0
    for y in range(h):
        for x in range(w):
            intensity = resized_gray[y, x]
            if intensity > intensity_threshold:
                continue

            darkness = 255 - intensity
            patch_size = int(patch_size_base + (darkness / 255.0) * (max_patch_size - patch_size_base))
            color = img_colored[y, x].tolist()
            half_size = patch_size
            top_left = (x - half_size, y - half_size)
            bottom_right = (x + half_size, y + half_size)
            cv2.rectangle(canvas, top_left, bottom_right, color, -1)

            data["dots"].append({
                "position": [int(x), int(y)],
                "patch_size": int(patch_size),
                "color": [int(c) for c in color]
            })

            dot_count += 1
            if dot_count % batch_size == 0:
                display_img = cv2.resize(canvas, (0, 0), fx=display_scale, fy=display_scale)
                cv2.imshow(window_name, display_img)
                if cv2.waitKey(1 if delay <= 0 else delay) == 27:
                    cv2.destroyAllWindows()
                    return

    print(f"[**] Total patches drawn: {dot_count}")
    display_img = cv2.resize(canvas, (0, 0), fx=display_scale, fy=display_scale)
    cv2.imshow(window_name, display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output_img = "sketch_color_output.jpg"
    output_json = "sketch_data.json"
    cv2.imwrite(output_img, canvas)
    with open(output_json, "w") as f:
        json.dump(data, f)

    print(f"[✔] Image saved to: {output_img}")
    print(f"[✔] Drawing data saved to: {output_json}")


# Gọi hàm
draw_image_with_sketch_color_and_export("input.jpg")
