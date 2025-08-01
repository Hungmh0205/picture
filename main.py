import cv2
import numpy as np
import random
import os

def draw_image_with_sketch_effects_batch_lines(
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
    canny_thresh2=120
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"[>>] Initiating data stream from: {image_path}")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    print(f"[##] Source dimensions: {width} x {height}")

    resized_gray = cv2.resize(
        gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_LINEAR
    )
    h, w = resized_gray.shape
    print(f"[**] Rescaled input (x{scale_factor}): {w} x {h}")

    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    window_name = "Truong Tieu Pham"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[//] Engaging edge vectorization protocol...")
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

            cv2.line(canvas, jitter_pt1, jitter_pt2, (100, 100, 100), thickness=line_thickness)
            line_count += 1

            if line_count % line_batch_size == 0:
                print(f"[//] {line_count} lines traced... recalibrating visual output")
                display_img = cv2.resize(canvas, (0, 0), fx=display_scale, fy=display_scale)
                cv2.imshow(window_name, display_img)
                if cv2.waitKey(1) == 27:
                    cv2.destroyAllWindows()
                    return

    print(f"[//] Edge vectorization complete. Total lines: {line_count}")

    print("[..] Initiating stochastic dot matrix shading...")
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
                cv2.circle(canvas, (x, y), dot_size, (0, 0, 0), -1)
                dot_count += 1

            if dot_count > 0 and dot_count % batch_size == 0:
                print(f"[..] {dot_count} dots deployed | {total_pixels} scanned | {skipped_pixels} bypassed")
                display_img = cv2.resize(canvas, (0, 0), fx=display_scale, fy=display_scale)
                key = cv2.waitKey(1 if delay <= 0 else delay)
                cv2.imshow(window_name, display_img)
                if key == 27:
                    cv2.destroyAllWindows()
                    return

    print(f"[**] Dot matrix synthesis complete. Total dots: {dot_count}")
    display_img = cv2.resize(canvas, (0, 0), fx=display_scale, fy=display_scale)
    cv2.imshow(window_name, display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output_path = "sketch_fast_lines_dots.jpg"
    cv2.imwrite(output_path, canvas)
    print(f"[!!] Output archived at: {output_path}")

draw_image_with_sketch_effects_batch_lines("input.jpg")
