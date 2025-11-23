# MCP Server (Flask)

## Setup
1. Create virtualenv and install requirements:
   ```
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r ../requirements.txt
   ```

2. Set environment variables (optional):
   - `MONGO_URI` (default: mongodb://localhost:27017)
   - `MCP_API_KEY` (optional, for simple auth)

3. Run server:
   ```
   export FLASK_APP=app.py
   flask run --host=0.0.0.0 --port=5000
   ```

## Endpoints

- `POST /tools/mongo_read`
  - Body: `{ "filter": {...}, "projection": {...}, "limit": 100 }`
  - Returns matching documents.

- `POST /tools/mongo_insert`
  - Body: `{ "documents": [ {...}, {...} ] }`
  - Inserts documents and returns inserted ids.

- `POST /tools/mongo_update`
  - Body: `{ "filter": {...}, "update": {...}, "many": false }`
  - Updates documents; returns matched_count and modified_count.

- `POST /tools/mongo_delete`
  - Body: `{ "filter": {...}, "many": false }`
  - Deletes documents; returns deleted_count.

All endpoints require header `Content-Type: application/json`.
If `MCP_API_KEY` is set, include header `X-API-KEY: <key>` on requests.
