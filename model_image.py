import pandas as pd
import requests
import cv2
import mediapipe as mp
import numpy as np
import io
import os
from urllib.parse import urlparse
import math

INPUT_CSV_PATH = './data/pae_dataset.csv'
OUTPUT_CSV_PATH = './data/myntra_data_updated_front_facing.csv'
VISIBILITY_THRESHOLD = 0.65
MIN_VISIBLE_LANDMARKS_OVERALL = 8
VERTICAL_SPREAD_THRESHOLD = 0.6
MAX_Y_DIFF_RATIO_SHOULDERS = 0.08
MAX_Y_DIFF_RATIO_HIPS = 0.08
MAX_Z_DIFF_SHOULDERS = 0.4
MAX_Z_DIFF_HIPS = 0.4

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True,
                             model_complexity=2,
                             min_detection_confidence=0.6)

LM = mp.solutions.pose.PoseLandmark
FULL_BODY_REQUIRED_LANDMARKS = {
    LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP,
    LM.LEFT_ANKLE, LM.RIGHT_ANKLE
}
UPPER_BODY_REQUIRED_LANDMARKS = {
    LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_ELBOW, LM.RIGHT_ELBOW
}
FRONT_FACING_CHECK_LANDMARKS = {
    LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP
}

def _process_image_url(image_url):
    if not image_url or not isinstance(image_url, str):
        return None, None, None
    try:
        parsed_url = urlparse(image_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
             return None, None, None
    except ValueError:
        return None, None, None
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(image_url, stream=True, timeout=15, headers=headers)
        response.raise_for_status()
        image_data = io.BytesIO(response.content)
        image_np = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)
        if image_np is None:
            print(f"Failed to decode image from URL: {image_url}")
            return None, None, None
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(image_rgb)
        return results, image_url, image_np.shape
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {image_url}: {e}")
        return None, None, None
    except cv2.error as e:
         print(f"OpenCV error processing {image_url}: {e}")
         return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred processing {image_url}: {e}")
        return None, None, None

def get_landmark_if_visible(landmarks, landmark_enum, min_visibility):
    idx = landmark_enum.value
    if idx < len(landmarks) and landmarks[idx].visibility > min_visibility:
        return landmarks[idx]
    return None

def is_front_facing(landmarks, image_shape):
    if not landmarks or not image_shape:
        return False

    image_height = image_shape[0]
    if image_height <= 0: return False

    nose = get_landmark_if_visible(landmarks, LM.NOSE, VISIBILITY_THRESHOLD)
    l_shoulder = get_landmark_if_visible(landmarks, LM.LEFT_SHOULDER, VISIBILITY_THRESHOLD)
    r_shoulder = get_landmark_if_visible(landmarks, LM.RIGHT_SHOULDER, VISIBILITY_THRESHOLD)
    l_hip = get_landmark_if_visible(landmarks, LM.LEFT_HIP, VISIBILITY_THRESHOLD)
    r_hip = get_landmark_if_visible(landmarks, LM.RIGHT_HIP, VISIBILITY_THRESHOLD)

    if not all([nose, l_shoulder, r_shoulder, l_hip, r_hip]):
        return False

    y_diff_shoulders = abs(l_shoulder.y - r_shoulder.y)
    y_diff_hips = abs(l_hip.y - r_hip.y)

    if y_diff_shoulders > MAX_Y_DIFF_RATIO_SHOULDERS:
        return False
    if y_diff_hips > MAX_Y_DIFF_RATIO_HIPS:
        return False

    if any(math.isnan(coord) or math.isinf(coord) for coord in [l_shoulder.z, r_shoulder.z, l_hip.z, r_hip.z]):
        return False

    z_diff_shoulders = abs(l_shoulder.z - r_shoulder.z)
    z_diff_hips = abs(l_hip.z - r_hip.z)

    if z_diff_shoulders > MAX_Z_DIFF_SHOULDERS:
        return False
    if z_diff_hips > MAX_Z_DIFF_HIPS:
        return False

    return True


def check_pose_type(results, image_shape):
    if not results or not results.pose_landmarks or not image_shape:
        return 'None', False

    landmarks = results.pose_landmarks.landmark
    num_landmarks_total = len(landmarks)
    visible_landmarks_count = 0
    visible_full_body_req_count = 0
    visible_upper_body_req_count = 0
    min_y, max_y = float('inf'), float('-inf')
    image_height = image_shape[0]

    for i, lm in enumerate(landmarks):
        if lm.visibility > VISIBILITY_THRESHOLD:
            visible_landmarks_count += 1
            lm_enum = LM(i)
            if lm_enum in FULL_BODY_REQUIRED_LANDMARKS:
                visible_full_body_req_count += 1
                if image_height > 0:
                    lm_y_px = lm.y * image_height
                    min_y = min(min_y, lm_y_px)
                    max_y = max(max_y, lm_y_px)
            if lm_enum in UPPER_BODY_REQUIRED_LANDMARKS:
                 visible_upper_body_req_count += 1

    if visible_landmarks_count < MIN_VISIBLE_LANDMARKS_OVERALL:
        return 'None', False

    front_facing = is_front_facing(landmarks, image_shape)

    is_full_body = False
    if visible_full_body_req_count >= len(FULL_BODY_REQUIRED_LANDMARKS):
        if min_y != float('inf') and max_y != float('-inf') and image_height > 0:
            vertical_spread_px = max_y - min_y
            vertical_spread_ratio = vertical_spread_px / image_height
            if vertical_spread_ratio >= VERTICAL_SPREAD_THRESHOLD:
                is_full_body = True

    if is_full_body:
        return 'Full', front_facing

    is_upper_body = False
    if visible_upper_body_req_count >= len(UPPER_BODY_REQUIRED_LANDMARKS):
        is_upper_body = True

    if is_upper_body:
        return 'Upper', front_facing

    return 'None', False


try:
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_CSV_PATH}")
except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_CSV_PATH}")
    exit()
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

df['new_model_image_url'] = np.nan

total_rows = len(df)
for index, row in df.iterrows():
    print(f"\nProcessing Product {index + 1}/{total_rows}: {row.get('product_id', 'N/A')}")

    candidate_urls = []
    if pd.notna(row['model_image_url']) and isinstance(row['model_image_url'], str):
        candidate_urls.append(row['model_image_url'])
    if pd.notna(row['additional_images']) and isinstance(row['additional_images'], str):
        add_urls = row['additional_images'].replace(';',',').split(',')
        candidate_urls.extend([url.strip() for url in add_urls if url.strip() and url.strip() not in candidate_urls])
    unique_candidate_urls = list(dict.fromkeys(candidate_urls))

    print(f"Found {len(unique_candidate_urls)} unique candidate URLs.")

    candidate_results = []

    for img_url in unique_candidate_urls:
        results, _, img_shape = _process_image_url(img_url)
        if results and img_shape:
            pose_type, front_facing = check_pose_type(results, img_shape)
            if pose_type != 'None':
                print(f"  URL: {img_url} -> Type: {pose_type}, Front: {front_facing}")
                candidate_results.append({'url': img_url, 'type': pose_type, 'front': front_facing})

    selected_url = None
    for res in candidate_results:
        if res['type'] == 'Full' and res['front']:
            selected_url = res['url']
            print(f"--> Selected Priority 1: Full Body, Front Facing ({selected_url})")
            break
    if not selected_url:
        for res in candidate_results:
            if res['type'] == 'Full':
                selected_url = res['url']
                print(f"--> Selected Priority 2: Full Body, Any Orientation ({selected_url})")
                break
    if not selected_url:
        for res in candidate_results:
            if res['type'] == 'Upper' and res['front']:
                selected_url = res['url']
                print(f"--> Selected Priority 3: Upper Body, Front Facing ({selected_url})")
                break
    if not selected_url:
        for res in candidate_results:
            if res['type'] == 'Upper':
                selected_url = res['url']
                print(f"--> Selected Priority 4: Upper Body, Any Orientation ({selected_url})")
                break

    if selected_url:
        df.loc[index, 'new_model_image_url'] = selected_url
        print(f"Final selection for product {row.get('product_id', 'N/A')}: {selected_url}")
    else:
        print(f"No suitable model image found meeting criteria for product {row.get('product_id', 'N/A')}. Leaving blank.")


pose_detector.close()

try:
    df.to_csv(OUTPUT_CSV_PATH, index=False, na_rep='')
    print(f"\nProcessing complete. Front-facing prioritized data saved to {OUTPUT_CSV_PATH}")
except Exception as e:
    print(f"Error saving CSV: {e}")

