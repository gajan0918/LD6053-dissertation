#!/usr/bin/env python3
from PIL import Image
import numpy as np
import sys

def validate_dog_skin_image(image_pil):
    try:
        img = image_pil.convert("RGB")
        img_array = np.array(img)
        
        avg_color = np.mean(img_array, axis=(0, 1))
        r, g, b = avg_color[0], avg_color[1], avg_color[2]
        
        blue_dominance = b > (r + 10) and b > (g + 10)
        if blue_dominance:
            return False, "Blue dominance - human face"
        
        std_dev = np.std(img_array)
        if std_dev > 100:
            return False, f"High variation - facial features ({std_dev:.0f})"
        
        if np.max(img_array) < 50 or np.min(img_array) > 240:
            return False, "Extreme brightness"
        
        return True, "Valid dog skin"
    except Exception as e:
        return True, f"Error: {str(e)}"

# Test
print("="*50)
print("TESTING VALIDATION WITH DOG IMAGE")
print("="*50)
dog_img = Image.open("Dataset/test/Healthy/075_jpg.rf.f10d5301e05f89372e15d834b4ed7cee.jpg")
is_valid, msg = validate_dog_skin_image(dog_img)
print(f"Result: {'✅ PASS' if is_valid else '❌ FAIL'}")
print(f"Message: {msg}")

img_array = np.array(dog_img.convert("RGB"))
r, g, b = np.mean(img_array, axis=(0, 1))
print(f"Colors - R:{r:.0f}, G:{g:.0f}, B:{b:.0f}")
print(f"Std Dev: {np.std(img_array):.0f}")
