import os
import urllib.request
import zipfile
import sys

def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100 / total_size
        sys.stdout.write(f"\rDownloading WESAD Dataset: {percent:.1f}%")
        sys.stdout.flush()

def download_and_extract_wesad(target_dir="references/WESAD"):
    url = "https://archive.ics.uci.edu/static/public/464/wesad+wearable+stress+and+affect+detection.zip"
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "wesad.zip")
    
    if not os.path.exists(zip_path):
        print(f"Starting download from {url}...")
        urllib.request.urlretrieve(url, zip_path, reporthook=download_progress)
        print("\nDownload complete.")
    else:
        print("WESAD zip file already exists. Skipping download.")
        
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("Extraction complete. WESAD dataset is ready.")

if __name__ == "__main__":
    download_and_extract_wesad()
