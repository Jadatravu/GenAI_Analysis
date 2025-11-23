import os

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("MONGO_DB", "appliance_db")
COLLECTION_NAME = os.environ.get("MONGO_COLLECTION", "appliances")
#API_KEY = os.environ.get("MCP_API_KEY","ABCD123456")  # optional
API_KEY="ABCD123456"