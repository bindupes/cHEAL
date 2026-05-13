import cv2
import numpy as np
import os

def highlight_cells(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    h, w = binary.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2 - 30, 255, -1)
    masked_binary = cv2.bitwise_and(binary, binary, mask=mask)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(masked_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()
    confirmed = 0
    possible = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 300 < area < 8000 and len(cnt) >= 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            eccentricity = np.sqrt(1 - (min(MA, ma) / max(MA, ma))**2)
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(cnt)
            if eccentricity > 0.88 and len(approx) < 7:
                confirmed += 1
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 3)  # RED
            elif 0.82 < eccentricity <= 0.88 and len(approx) < 9:
                possible += 1
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 165, 255), 3)  # ORANGE
    
    # Ensure static directory exists
    os.makedirs("static", exist_ok=True)
    output_path = os.path.join("static", "highlighted.png")
    cv2.imwrite(output_path, output)
    
    print(f"Highlighted image saved to: {output_path}")
    print(f"Confirmed sickle cells: {confirmed}")
    print(f"Possible sickle cells: {possible}")
    
    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python highlight_cells.py <image_path>")
    else:
        result_path = highlight_cells(sys.argv[1])
        print(f"Output saved to: {result_path}")