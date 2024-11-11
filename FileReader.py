import os
import requests
from bs4 import BeautifulSoup
import time

# Specify the URL of the folder you want to download from
url = "http://optics.gi.alaska.edu/amisr_archive/PKR/DASC/RAW/2018/20180419/"

# Specify the local folder where you want to save the files
save_folder = "PokerFlat_2018_04_19"
os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Set a timeout duration and number of retries
timeout_duration = 10  # seconds
max_retries = 3

response = requests.get(url, timeout=timeout_duration)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")

    try:    # Adding a stop action

        # Loop through each link in the specified folder and download .FITS files
        for link in soup.find_all("a"):
            file_name = link.get("href")
            if file_name.endswith(".FITS"):  # Look for .FITS files specifically
                file_url = url + file_name
                save_path = os.path.join(save_folder, file_name)  # Save file in the specified folder

                retries = 0
                while retries < max_retries:
                    try:
                        # Download the file with a timeout
                        with requests.get(file_url, stream=True, timeout=timeout_duration) as r:
                            with open(save_path, "wb") as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    if chunk:  # filter out keep-alive new chunks
                                        f.write(chunk)

                        print(f"Downloaded {file_name} to {save_folder}")
                        break  # Exit the retry loop if download is successful

                    except requests.exceptions.Timeout:
                        retries += 1
                        print(f"Timeout occurred for {file_name}. Retrying {retries}/{max_retries}...")
                        time.sleep(2)  # Wait a bit before retrying
                    except Exception as e:
                        print(f"Failed to download {file_name}: {e}")
                        break  # Break the loop if an error other than timeout occurs

                if retries == max_retries:
                    print(f"Failed to download {file_name} after {max_retries} retries.")
            else:
                print(f"{file_name} is not a FITS")
    except KeyboardInterrupt:
        print(f"Keyboard interrupt occurred")
        exit()

    print("All files processed.")
else:
    print(f"Failed to access URL: {url} (Status code: {response.status_code})")
