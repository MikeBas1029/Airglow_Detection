import os
import requests
from bs4 import BeautifulSoup

# Specify the URL of the folder you want to download from
url = "http://optics.gi.alaska.edu/amisr_archive/PKR/DASC/RAW/2018/20180120/"

# Specify the local folder where you want to save the files
save_folder = "PokerFlat_2018_01_20"
os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")

    # Loop through each link in the specified folder and download .FIT files
    for link in soup.find_all("a"):
        file_name = link.get("href")
        if file_name.endswith(".FITS"):  # Look for .FIT files specifically
            file_url = url + file_name
            save_path = os.path.join(save_folder, file_name)  # Save file in the specified folder

            # Download the file
            with requests.get(file_url, stream=True) as r:
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            print(f"Downloaded {file_name} to {save_folder}")
        else:
            print(f"{file_name} is not a FITS")
    print(f"Files has been downloaded")
else:
    print(f"Failed to access URL: {url} (Status code: {response.status_code})")