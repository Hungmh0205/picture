import cv2
import numpy as np
import random
import os
import json

def draw_image_with_sketch_effects_and_save_data(
    image_path,
    scale_factor=2,
    dot_spacing=2,
    dot_size_base=1,
    max_dot_size=3,
    intensity_threshold=230,
    delay=0,
    display_scale=0.3,
    batch_size=500,
    line_jitter=1,
    line_thickness=2,
    line_step=2,
    line_batch_size=100,
    canny_thresh1=50,
    canny_thresh2=120,
    output_json_path="drawing_data.json"
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"[>>] Loading: {image_path}")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    resized_gray = cv2.resize(
        gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_LINEAR
    )
    h, w = resized_gray.shape
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    window_name = "Sketch Output"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # JSON output data
    output_data = {"lines": [], "dots": []}

    # === Draw lines ===
    blurred = cv2.GaussianBlur(resized_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    line_count = 0
    for contour in contours:
        for i in range(line_step, len(contour), line_step):
            pt1 = contour[i - line_step][0]
            pt2 = contour[i][0]

            jitter_pt1 = (
                pt1[0] + random.randint(-line_jitter, line_jitter),
                pt1[1] + random.randint(-line_jitter, line_jitter)
            )
            jitter_pt2 = (
                pt2[0] + random.randint(-line_jitter, line_jitter),
                pt2[1] + random.randint(-line_jitter, line_jitter)
            )

            color = (100, 100, 100)
            cv2.line(canvas, jitter_pt1, jitter_pt2, color, thickness=line_thickness)

            output_data["lines"].append({
                "start": [int(jitter_pt1[0]), int(jitter_pt1[1])],
                "end": [int(jitter_pt2[0]), int(jitter_pt2[1])],
                "color": [int(c) for c in color],
                "thickness": int(line_thickness)
            })

            line_count += 1
            if line_count % line_batch_size == 0:
                print(f"[//] {line_count} lines")
                display_img = cv2.resize(canvas, (0, 0), fx=display_scale, fy=display_scale)
                cv2.imshow(window_name, display_img)
                if cv2.waitKey(1) == 27:
                    cv2.destroyAllWindows()
                    return

    # === Draw dots ===
    dot_count = 0
    total_pixels = 0
    skipped_pixels = 0

    for y in range(0, h, dot_spacing):
        for x in range(0, w, dot_spacing):
            total_pixels += 1
            intensity = resized_gray[y, x]
            if intensity > intensity_threshold:
                skipped_pixels += 1
                continue

            darkness = 255 - intensity
            probability = darkness / 255.0
            if random.random() < probability:
                dot_size = int(dot_size_base + (darkness / 255.0) * (max_dot_size - dot_size_base))
                color = (0, 0, 0)
                cv2.circle(canvas, (x, y), dot_size, color, -1)

                output_data["dots"].append({
                    "position": [int(x), int(y)],
                    "radius": int(dot_size),
                    "color": [int(c) for c in color]
                })

                dot_count += 1

            if dot_count > 0 and dot_count % batch_size == 0:
                print(f"[..] {dot_count} dots | {total_pixels} scanned | {skipped_pixels} skipped")
                display_img = cv2.resize(canvas, (0, 0), fx=display_scale, fy=display_scale)
                key = cv2.waitKey(1 if delay <= 0 else delay)
                cv2.imshow(window_name, display_img)
                if key == 27:
                    cv2.destroyAllWindows()
                    return

    # === Output ===
    display_img = cv2.resize(canvas, (0, 0), fx=display_scale, fy=display_scale)
    cv2.imshow(window_name, display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output_img_path = "sketch_result.jpg"
    cv2.imwrite(output_img_path, canvas)
    print(f"[✓] Image saved at: {output_img_path}")

    # Save JSON
    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
    print(f"[✓] Drawing data saved at: {output_json_path}")

# Run
draw_image_with_sketch_effects_and_save_data("input.jpg")
