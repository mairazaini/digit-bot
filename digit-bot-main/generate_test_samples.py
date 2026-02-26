import cv2
import numpy as np
import os

# 1. Create a folder for the images
output_dir = "test_samples"
os.makedirs(output_dir, exist_ok=True)

# 2. Define settings (White background, Black text)
canvas_size = 300
font_scale = 10
thickness = 25
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX  # Looks a bit like handwriting

print(f"Generating 10 images in '{output_dir}'...")

for digit in range(10):
    # Create white image
    img = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
    
    # Calculate text size to center it
    text = str(digit)
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = (canvas_size - text_w) // 2
    y = (canvas_size + text_h) // 2

    # Draw the digit in Black
    cv2.putText(img, text, (x, y), font, font_scale, (0), thickness, cv2.LINE_AA)
    
    # Save
    filename = os.path.join(output_dir, f"digit_{digit}.png")
    cv2.imwrite(filename, img)
    print(f"Saved: {filename}")

print("Done! Upload these images in GUI.")