import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "ChatBot0")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "UserDetails")

if not MONGO_URI:
    raise Exception("MONGO_URI not found in environment variables")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
collection = db[MONGO_COLLECTION]

def test_connection():
    try:
        client.admin.command('ping')
        return True
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return False

connected = test_connection()
if connected:
    print("MongoDB connection successful!")
else:
    print("Warning: MongoDB connection failed. Using fallback mode.")