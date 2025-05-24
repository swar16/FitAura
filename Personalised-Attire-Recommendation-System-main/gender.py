import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
import re
from collections import defaultdict

INPUT_CSV_PATH = './data/myntra_data_with_skin_color.csv'
OUTPUT_CSV_PATH = './data/myntra_data_with_gender_freq_v2.csv'
URL_COLUMN = 'product_url'
GENDER_COLUMN = 'detected_gender_freq'

GENDER_KEYWORDS = {
    'Girls': ['girl', 'girls'],
    'Boys': ['boy', 'boys'],
    'Women': ['women', 'woman', 'womens', 'ladies', 'female'],
    'Men': ['men', 'man', 'mens', 'male', 'gentlemen'],
    'Kids': ['kids', 'children', 'child', 'junior', 'youth'],
    'Unisex': ['unisex', 'all genders'],
}

def get_gender_by_frequency_targeted(url):
    if not url or not isinstance(url, str) or not url.startswith('http'):
        return "Invalid URL"

    print(f"--- Processing URL: {url} ---")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        response = requests.get(url, headers=headers, timeout=25)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            print(f"Warning: Non-HTML content type '{content_type}' for URL: {url}")

        soup = BeautifulSoup(response.content, 'html.parser')

        search_text = ""

        title_tag = soup.find('title')
        if title_tag:
            search_text += title_tag.get_text(separator=' ', strip=True).lower() + " "

        h1_tag = soup.find('h1', class_='pdp-title')
        if h1_tag:
            search_text += h1_tag.get_text(separator=' ', strip=True).lower() + " "
        else:
             h1_tag = soup.find('h1')
             if h1_tag:
                  search_text += h1_tag.get_text(separator=' ', strip=True).lower() + " "

        breadcrumb_container = soup.find('div', class_='breadcrumbs-container')
        if breadcrumb_container:
            links = breadcrumb_container.find_all('a', class_='breadcrumbs-link')
            for link in links:
                search_text += link.get_text(separator=' ', strip=True).lower() + " "
        else:
             breadcrumb_divs = soup.find_all('div', class_=re.compile(r'breadcrumb', re.I))
             for div in breadcrumb_divs:
                  links = div.find_all('a')
                  for link in links:
                       search_text += link.get_text(separator=' ', strip=True).lower() + " "

        description_div = soup.find('div', class_='pdp-product-description-content')
        if description_div:
            search_text += description_div.get_text(separator=' ', strip=True).lower() + " "
        else:
             desc_divs = soup.find_all('div', attrs={'class': re.compile(r'desc', re.I)})
             for div in desc_divs:
                  if len(div.get_text()) < 1000:
                       search_text += div.get_text(separator=' ', strip=True).lower() + " "

        print(f"Extracted Text Snippet (first 300 chars): {search_text[:300]}")

        if not search_text.strip():
            print("Error: No targeted text found.")
            return "Error - No Text Found"

        gender_counts = defaultdict(int)
        for gender_category, keywords in GENDER_KEYWORDS.items():
            category_count = 0
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.findall(pattern, search_text)
                count = len(matches)
                category_count += count
            gender_counts[gender_category] = category_count

        print(f"Final Counts: {dict(gender_counts)}")

        if not any(gender_counts.values()):
            print("Result: Not Found (Zero Counts)")
            return "Not Found"

        max_count = 0
        for count in gender_counts.values():
             if count > max_count:
                  max_count = count

        winners = [gender for gender, count in gender_counts.items() if count == max_count]
        print(f"Max Count: {max_count}, Winners (pre-tiebreak): {winners}")

        if len(winners) == 1:
            print(f"Result: {winners[0]}")
            return winners[0]
        elif len(winners) > 1:
            priority_order = ['Women', 'Men', 'Girls', 'Boys', 'Unisex', 'Kids']
            for preferred_gender in priority_order:
                if preferred_gender in winners:
                    print(f"Result (Tie-Breaker): {preferred_gender}")
                    return preferred_gender
            print(f"Result (Tie-Fallback): {winners[0]}")
            return winners[0]
        else:
             print("Result: Not Found (Logical Error?)")
             return "Not Found"

    except requests.exceptions.Timeout:
        print(f"Timeout error for URL: {url}")
        return "Error - Timeout"
    except requests.exceptions.RequestException as e:
        print(f"Request error for URL {url}: {e}")
        return f"Error - Request Failed"
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return "Error - Processing Failed"

try:
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_CSV_PATH}")
except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_CSV_PATH}")
    exit()
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

if URL_COLUMN not in df.columns:
    print(f"Error: URL column '{URL_COLUMN}' not found.")
    exit()

df[GENDER_COLUMN] = "Not Processed"

total_rows = len(df)
for index, row in df.iterrows():
    print(f"\nProcessing Gender Freq V2 {index + 1}/{total_rows}: Product {row.get('product_id', 'N/A')}")
    url = row[URL_COLUMN]

    gender = get_gender_by_frequency_targeted(url)
    df.loc[index, GENDER_COLUMN] = gender

    time.sleep(random.uniform(1.8, 4.5))

try:
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nProcessing complete. Targeted frequency-based gender saved to {OUTPUT_CSV_PATH}")
except Exception as e:
    print(f"Error saving CSV: {e}")

