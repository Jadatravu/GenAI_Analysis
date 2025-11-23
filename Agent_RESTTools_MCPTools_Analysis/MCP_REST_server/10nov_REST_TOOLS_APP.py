from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
import os
import json
from config import API_KEY
from tools import mongo_read, mongo_insert, mongo_update, mongo_delete

app = Flask(__name__)

def require_api_key(req):
    #if API_KEY:
    #    key = req.headers.get("X-API-KEY")
    #    if not key or key != API_KEY:
    #        return False
    return True

@app.before_request
def check_api_key():
    if not require_api_key(request):
        return jsonify({"error": "Invalid or missing API key"}), 401

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "mcp_server"})

@app.route("/tools", methods=["GET"])
def tools():
    return jsonify({"status": "ok", "tools": ["mongo_read","mongo_insert","mongo_update","mongo_delete"]})

@app.route("/tools/<tool_name>", methods=["POST"])
def call_tool(tool_name):
    try:
        payload = request.get_json(force=True)
    except BadRequest:
        return jsonify({"error": "invalid_json"}), 400

    try:
        if tool_name == "mongo_read":
            filt = payload.get("filter")
            print(filt)
            proj = payload.get("projection")
            print(proj)
            limit = payload.get("limit", 100)
            results = mongo_read(filter=filt, projection=proj, limit=limit)
            return jsonify({"status": "ok", "results": results})
        elif tool_name == "mongo_insert":
            docs = payload.get("documents")
            if docs is None:
                return jsonify({"error": "documents_required"}), 400
            res = mongo_insert(docs)
            return jsonify({"status": "ok", "result": res})
        elif tool_name == "mongo_update":
            filt = payload.get("filter", {})
            update = payload.get("update")
            many = payload.get("many", False)
            if update is None:
                return jsonify({"error": "update_required"}), 400
            res = mongo_update(filt, update, many=many)
            return jsonify({"status": "ok", "result": res})
        elif tool_name == "mongo_delete":
            filt = payload.get("filter", {})
            many = payload.get("many", False)
            res = mongo_delete(filt, many=many)
            return jsonify({"status": "ok", "result": res})
        else:
            return jsonify({"error": "unknown_tool"}), 404
    except Exception as e:
        return jsonify({"error": "exception", "message": str(e)}), 500

if __name__ == '__main__':
    # allow running as python -m mcp_server.app
    app.run(host='127.0.0.1', port=int(os.environ.get("PORT", 5000)),debug=True)
