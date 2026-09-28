from __future__ import annotations

import ast
import operator as op
import re
import time


CUSTOMERS = {
    "C1001": {"customer_id": "C1001", "name": "Ravi", "city": "Hyderabad", "tier": "Gold"},
    "C1002": {"customer_id": "C1002", "name": "Anita", "city": "Bengaluru", "tier": "Silver"},
    "C1003": {"customer_id": "C1003", "name": "John", "city": "Chennai", "tier": "Gold"},
}

ORDERS = {
    "O1001": {"order_id": "O1001", "customer_id": "C1001", "status": "Shipped", "amount": 12500},
    "O1002": {"order_id": "O1002", "customer_id": "C1002", "status": "Delivered", "amount": 8300},
    "O1003": {"order_id": "O1003", "customer_id": "C1003", "status": "Processing", "amount": 4200},
}


def customer_lookup(request):
    match = re.search(r"\bC\d+\b", request.upper())
    if not match:
        return {"error": "Customer ID not found"}
    cid = match.group()
    return CUSTOMERS.get(cid, {"error": f"Customer {cid} not found"})


def order_lookup(request):
    match = re.search(r"\bO\d+\b", request.upper())
    if not match:
        return {"error": "Order ID not found"}
    oid = match.group()
    return ORDERS.get(oid, {"error": f"Order {oid} not found"})


_ALLOWED = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](
            _safe_eval(node.left),
            _safe_eval(node.right),
        )
    raise ValueError("Unsupported arithmetic expression")


def calculator(request):
    expression = re.sub(r"(?i)calculate|compute|what is", "", request).strip().rstrip("?")
    try:
        result = _safe_eval(ast.parse(expression, mode="eval"))
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"error": str(exc)}


def web_search(request):
    return {
        "query": request,
        "message": "Web search stub. Connect this to your web/MCP search tool."
    }


def no_tool(request):
    return {"message": "No external tool required."}


TOOLS = {
    "customer_lookup": customer_lookup,
    "order_lookup": order_lookup,
    "calculator": calculator,
    "web_search": web_search,
    "none": no_tool,
}
