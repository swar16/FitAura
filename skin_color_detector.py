import pandas as pd
import requests
import cv2
import numpy as np
import io
from sklearn.cluster import KMeans
import os
from urllib.parse import urlparse

INPUT_CSV_PATH = './data/myntra_data_updated_front_facing.csv'
OUTPUT_CSV_PATH = './data/myntra_data_with_skin_color.csv'
IMAGE_COLUMN = 'new_model_image_url'
OUTPUT_COLUMN = 'detected_skin_color_rgb'

LOWER_SKIN_HSV = np.array([0, 40, 50], dtype="uint8")
UPPER_SKIN_HSV = np.array([25, 150, 255], dtype="uint8")
MIN_SKIN_PIXELS = 500

def get_dominant_skin_color(image_url):
    if not image_url or not isinstance(image_url, str):
        return None

    try:
        parsed_url = urlparse(image_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
             return None
    except ValueError:
        return None

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(image_url, stream=True, timeout=15, headers=headers)
        response.raise_for_status()

        image_data = io.BytesIO(response.content)
        image_np = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)

        if image_np is None:
            print(f"Failed to decode image from URL: {image_url}")
            return None

        hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

        skin_mask = cv2.inRange(hsv_image, LOWER_SKIN_HSV, UPPER_SKIN_HSV)

        skin_pixels_img = cv2.bitwise_and(image_np, image_np, mask=skin_mask)

        skin_pixels_bgr = image_np[skin_mask > 0]

        if len(skin_pixels_bgr) < MIN_SKIN_PIXELS:
            return None

        kmeans = KMeans(n_clusters=1, random_state=0, n_init=10)
        kmeans.fit(skin_pixels_bgr)

        dominant_bgr = kmeans.cluster_centers_[0].astype(int)

        dominant_rgb = (dominant_bgr[2], dominant_bgr[1], dominant_bgr[0])

        return f"({dominant_rgb[0]}, {dominant_rgb[1]}, {dominant_rgb[2]})"

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {image_url}: {e}")
        return None
    except cv2.error as e:
         print(f"OpenCV error processing {image_url}: {e}")
         return None
    except Exception as e:
        print(f"An unexpected error occurred processing {image_url}: {e}")
        return None

try:
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_CSV_PATH}")
except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_CSV_PATH}")
    exit()
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

if IMAGE_COLUMN not in df.columns:
    print(f"Error: Column '{IMAGE_COLUMN}' not found in the input CSV.")
    exit()

df[OUTPUT_COLUMN] = None

total_rows = len(df)
for index, row in df.iterrows():
    print(f"Processing Skin Color {index + 1}/{total_rows}: Product {row.get('product_id', 'N/A')}")

    image_url = row[IMAGE_COLUMN]

    if pd.isna(image_url) or not isinstance(image_url, str) or not image_url.strip():
        print("--> Skipping row due to missing or invalid image URL.")
        df.loc[index, OUTPUT_COLUMN] = "Invalid URL"
        continue

    dominant_color = get_dominant_skin_color(image_url)

    if dominant_color:
        print(f"--> Detected dominant skin color: {dominant_color}")
        df.loc[index, OUTPUT_COLUMN] = dominant_color
    else:
        print("--> Could not detect dominant skin color.")
        df.loc[index, OUTPUT_COLUMN] = "Not Detected"

try:
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nProcessing complete. Updated data with skin color saved to {OUTPUT_CSV_PATH}")
except Exception as e:
    print(f"Error saving CSV: {e}")

