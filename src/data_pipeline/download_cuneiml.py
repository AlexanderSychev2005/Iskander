import json
import os
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

JSON_FILE = "../data/CuneiMLv1.2.json"
DATA_DIR = "../data/images_dataset"
IMG_DIR = os.path.join(DATA_DIR, "images")
LINEART_DIR = os.path.join(DATA_DIR, "linearts")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LINEART_DIR, exist_ok=True)

def download_file(url, filepath):
    if not url:
        return False
    if os.path.exists(filepath):
        return True # Already downloaded
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            with open(filepath, 'wb') as f:
                f.write(response.read())
        return True
    except Exception as e:
        return False

def process_item(item):
    item_id = item.get("id")
    if not item_id:
        return 0
    
    img_url = item.get("img_url")
    lineart_url = item.get("lineart")
    
    count = 0
    if img_url:
        ext = img_url.split('.')[-1]
        img_path = os.path.join(IMG_DIR, f"{item_id}.{ext}")
        if download_file(img_url, img_path):
            count += 1
            
    if lineart_url:
        ext = lineart_url.split('.')[-1]
        lineart_path = os.path.join(LINEART_DIR, f"{item_id}_l.{ext}")
        if download_file(lineart_url, lineart_path):
            count += 1
            
    return count

def main():
    print(f"Loading {JSON_FILE}...")
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} items. Starting download...")
    
    total_downloaded = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_item, item): item for item in data}
        for i, future in enumerate(as_completed(futures)):
            total_downloaded += future.result()
            
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {i+1}/{len(data)} items. Elapsed: {elapsed:.2f}s")
                
    print(f"Done! Processed all items. Downloaded/Verified: {total_downloaded} files.")

if __name__ == "__main__":
    main()
