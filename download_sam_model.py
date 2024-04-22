import requests
import os

def download_file(url, save_path):
    """
    Downloads a file from the given URL and saves it to the specified path.

    Args:
        url (str): URL of the file to download.
        save_path (str): Path where the downloaded file should be saved.
    """
    try:
        # Create the destination folder if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception if the request was unsuccessful

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"File downloaded and saved at: {save_path}")
    except requests.RequestException as e:
        print(f"Error downloading the file: {e}")

# Example usage
download_url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
download_path = '/home/ubuntu/projects/python/mayank/translate_object/models/sam_vit_h_4b8939.pth'

download_file(download_url, download_path)
