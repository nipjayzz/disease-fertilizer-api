import firebase_admin
from firebase_admin import credentials, storage
import os
import datetime

cred = credentials.Certificate('uee-we-53-firebase-adminsdk-n26zj-0c1c860d38.json')

firebase_admin.initialize_app(cred, {
    'storageBucket': 'uee-we-53.appspot.com'
})

bucket = storage.bucket()

def upload_image(image_path, destination_blob_name):
    try:
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(image_path)

        blob.make_public()

        print("Image uploaded successfully.")
        print("Public URL:", blob.public_url)

    except Exception as e:
        print("Error uploading image:", e)

if __name__ == "__main__":
    image_file = "sample.jpg"
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    destination = f'images/{timestamp}_{os.path.basename(image_file)}'

    upload_image(image_file, destination)
