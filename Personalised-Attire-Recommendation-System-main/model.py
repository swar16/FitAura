import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
import re
import os

DATASET_PATH = './data/final.csv'
GENDER_COLUMN = 'detected_gender_freq'
SKIN_COLOR_COLUMN = 'detected_skin_color_rgb'
MODEL_IMAGE_COLUMN = 'new_model_image_url'
URL_COLUMN = 'product_url'
PRICE_COLUMN = 'price'
TOP_N_RECOMMENDATIONS = 5

LOWER_SKIN_HSV = np.array([0, 40, 50], dtype="uint8")
UPPER_SKIN_HSV = np.array([25, 150, 255], dtype="uint8")
MIN_SKIN_PIXELS = 300

def parse_rgb_string(rgb_str):
    if not isinstance(rgb_str, str):
        return None
    match = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', rgb_str)
    if match:
        try:
            r, g, b = map(int, match.groups())
            if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                return np.array([r, g, b])
            else:
                print(f"Warning: Parsed RGB values out of range: {(r,g,b)} from '{rgb_str}'")
                return None
        except ValueError:
            print(f"Warning: Could not convert parsed values to int in '{rgb_str}'")
            return None
    else:
        return None

def get_dominant_skin_color_from_path(image_path):
    if not os.path.exists(image_path):
        print(f"Error: User image path not found: {image_path}")
        return None
    try:
        image_np = cv2.imread(image_path)
        if image_np is None:
            print(f"Error: Failed to read user image file: {image_path}")
            return None

        hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv_image, LOWER_SKIN_HSV, UPPER_SKIN_HSV)
        skin_pixels_bgr = image_np[skin_mask > 0]

        if len(skin_pixels_bgr) < MIN_SKIN_PIXELS:
            print(f"Warning: Insufficient skin pixels ({len(skin_pixels_bgr)}) detected in user image.")
            return None

        try:
            n_clusters = 1
            if len(np.unique(skin_pixels_bgr, axis=0)) < n_clusters:
                print("Warning: Fewer unique skin pixel colors than clusters, using average.")
                dominant_bgr = np.mean(skin_pixels_bgr, axis=0).astype(int)
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
                kmeans.fit(skin_pixels_bgr)
                dominant_bgr = kmeans.cluster_centers_[0].astype(int)

        except Exception as kmeans_err:
            print(f"Error during KMeans clustering on user image: {kmeans_err}")
            dominant_bgr = np.mean(skin_pixels_bgr, axis=0).astype(int)

        dominant_rgb = np.array([dominant_bgr[2], dominant_bgr[1], dominant_bgr[0]])
        print(f"Detected user dominant skin color (RGB): {dominant_rgb}")
        return dominant_rgb

    except cv2.error as e:
        print(f"OpenCV error processing user image {image_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred processing user image {image_path}: {e}")
        return None

def calculate_rgb_distance(rgb1, rgb2):
    if rgb1 is None or rgb2 is None:
        return float('inf')
    rgb1 = np.asarray(rgb1)
    rgb2 = np.asarray(rgb2)
    return np.linalg.norm(rgb1 - rgb2)

def recommend_products(user_gender, user_image_path, df, top_n=5):
    print("\n--- Starting Recommendation Process ---")
    user_skin_color = get_dominant_skin_color_from_path(user_image_path)
    if user_skin_color is None:
        print("Error: Could not determine user's skin color. Cannot provide recommendations.")
        return None

    filtered_df = df[df[GENDER_COLUMN].str.lower() == user_gender.lower()].copy()
    if filtered_df.empty:
        print(f"Sorry, no products found for the gender '{user_gender}' in the cleaned dataset.")
        return None
    print(f"Found {len(filtered_df)} products matching gender '{user_gender}'.")

    print("Calculating skin color distances...")
    filtered_df['color_distance'] = filtered_df['numeric_skin_color'].apply(
        lambda model_color: calculate_rgb_distance(user_skin_color, model_color)
    )

    recommendations_df = filtered_df.sort_values('color_distance', ascending=True).head(top_n)

    if recommendations_df.empty:
        print("Could not find any products with valid skin color data for the specified gender after filtering.")
        return None

    print(f"\n--- Top {len(recommendations_df)} Recommendations ---")
    output_cols = [MODEL_IMAGE_COLUMN, URL_COLUMN, PRICE_COLUMN, 'color_distance']
    final_recommendations = recommendations_df[output_cols]

    return final_recommendations

if __name__ == "__main__":
    try:
        print(f"Loading dataset from: {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
        print(f"Loaded {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        exit()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    print("Cleaning data...")
    initial_rows = len(df)

    invalid_url_indicators = ["", "Not Processed", "Invalid URL", "Error"]
    invalid_color_indicators = ["", "Not Detected", "Invalid URL", "Error", "Not Processed"]

    required_cols = [MODEL_IMAGE_COLUMN, SKIN_COLOR_COLUMN, GENDER_COLUMN, URL_COLUMN, PRICE_COLUMN]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in the dataset: {missing_cols}")
        exit()

    df.dropna(subset=[MODEL_IMAGE_COLUMN], inplace=True)
    df = df[~df[MODEL_IMAGE_COLUMN].astype(str).str.strip().isin(invalid_url_indicators)]
    print(f"Rows after removing invalid model URLs: {len(df)}")

    df.dropna(subset=[SKIN_COLOR_COLUMN], inplace=True)
    df = df[~df[SKIN_COLOR_COLUMN].astype(str).str.strip().isin(invalid_color_indicators)]
    print(f"Rows after removing invalid skin colors: {len(df)}")

    df['numeric_skin_color'] = df[SKIN_COLOR_COLUMN].apply(parse_rgb_string)

    df.dropna(subset=['numeric_skin_color'], inplace=True)
    print(f"Rows after parsing and removing failed skin colors: {len(df)}")

    rows_removed = initial_rows - len(df)
    print(f"Removed {rows_removed} rows during cleaning.")

    if len(df) == 0:
        print("Error: No valid data remaining after cleaning. Cannot proceed.")
        exit()

    while True:
        user_gender_input = input("Enter your gender (e.g., Men, Women, Boys, Girls): ").strip()
        available_genders = df[GENDER_COLUMN].str.lower().unique()
        if user_gender_input.lower() in available_genders:
            break
        else:
            print(f"Gender '{user_gender_input}' not found in available genders: {available_genders}. Please try again.")

    while True:
        user_image_path_input = input("Enter the full path to your image file: ").strip()
        if os.path.exists(user_image_path_input):
            break
        else:
            print("Invalid file path. Please enter a correct path.")

    recommendations = recommend_products(
        user_gender_input,
        user_image_path_input,
        df,
        top_n=TOP_N_RECOMMENDATIONS
    )

    if recommendations is not None and not recommendations.empty:
        for index, row in recommendations.iterrows():
            print("-" * 20)
            print(f"Recommendation {index + 1}:")
            print(f"  Price: {row[PRICE_COLUMN]}")
            print(f"  Model Image: {row[MODEL_IMAGE_COLUMN]}")
            print(f"  Product URL: {row[URL_COLUMN]}")
            print(f"  Skin Color Distance: {row['color_distance']:.2f}")
        print("-" * 20)
    else:
        print("\nCould not generate recommendations based on the provided input and available data.")