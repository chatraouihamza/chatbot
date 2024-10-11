import zipfile
import os

# Path to the ZIP file (change this to where your ZIP file is located)
zip_file_path = r'C:\Users\hp\Downloads\archive.zip'
file_data=r'C:\Users\hp\Documents\Chatbot\myenv\Data'
# Extract the file to the same directory as the ZIP file
extract_to = os.path.dirname(file_data)

# Check if the ZIP file exists
if os.path.exists(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)  # Unzip the file
    print("Extraction completed successfully!")
else:
    print(f"The file {zip_file_path} does not exist.")
