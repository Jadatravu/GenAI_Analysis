from pymongo import MongoClient
from bson import ObjectId
import json
from typing import Any, Dict, List
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

# Create a MongoDB client (single client reused)
_client = MongoClient(MONGO_URI)
_db = _client[DB_NAME]
_collection = _db[COLLECTION_NAME]

def _serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    # Convert ObjectId to string for JSON serializability
    doc = dict(doc)
    if '_id' in doc:
        doc['_id'] = str(doc['_id'])
    return doc

def mongo_read(filter: Dict = None, projection: Dict = None, limit: int = 100) -> List[Dict]:
    filter = filter or {}
    cursor = _collection.find(filter, projection).limit(limit)
    results = [_serialize_doc(d) for d in cursor]
    return results

def mongo_insert(documents: List[Dict]) -> Dict:
    if not isinstance(documents, list):
        documents = [documents]
    res = _collection.insert_many(documents)
    return {"inserted_ids": [str(i) for i in res.inserted_ids]}

def mongo_update(filter: Dict, update: Dict, many: bool = False) -> Dict:
    if many:
        res = _collection.update_many(filter, update)
    else:
        res = _collection.update_one(filter, update)
    return {"matched_count": res.matched_count, "modified_count": res.modified_count}

def mongo_delete(filter: Dict, many: bool = False) -> Dict:
    if many:
        res = _collection.delete_many(filter)
    else:
        res = _collection.delete_one(filter)
    return {"deleted_count": res.deleted_count}
