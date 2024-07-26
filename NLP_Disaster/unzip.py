import zipfile
import os

# Create directories to match the paths
os.makedirs('/kaggle/input/nlp-getting-started', exist_ok=True)
os.makedirs('/kaggle/input/disasters-on-social-media', exist_ok=True)

# Unzip nlp-getting-started.zip
with zipfile.ZipFile('nlp-getting-started.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/input/nlp-getting-started')

# Unzip disasters-on-social-media.zip
with zipfile.ZipFile('disasters-on-social-media.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/input/disasters-on-social-media')

print('Files extracted successfully')