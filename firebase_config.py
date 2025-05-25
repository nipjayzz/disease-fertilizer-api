import firebase_admin
from firebase_admin import credentials, firestore, storage

# Check if Firebase is already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate('uee-we-53-firebase-adminsdk-n26zj-0c1c860d38.json')
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'uee-we-53.appspot.com'
    })

# Initialize Firestore and Storage
db_firestore = firestore.client()
bucket = storage.bucket()
