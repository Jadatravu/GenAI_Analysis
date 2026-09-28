# Labeled benchmark dataset.
# expected_tool is the ground-truth label used ONLY for evaluation.

DATASET = [
    # Calculator
    {"id": 1, "request": "Calculate 125 * 24", "expected_tool": "calculator"},
    {"id": 2, "request": "What is 100 + 250?", "expected_tool": "calculator"},
    {"id": 3, "request": "Compute 500 / 25", "expected_tool": "calculator"},
    {"id": 4, "request": "Calculate 75 - 19", "expected_tool": "calculator"},
    {"id": 5, "request": "What is 12 * 12?", "expected_tool": "calculator"},
    {"id": 6, "request": "Compute 1000 / 8", "expected_tool": "calculator"},
    {"id": 7, "request": "Calculate 45 + 55", "expected_tool": "calculator"},
    {"id": 8, "request": "What is 9 * 11?", "expected_tool": "calculator"},

    # Customer
    {"id": 9, "request": "Find customer C1001", "expected_tool": "customer_lookup"},
    {"id": 10, "request": "Show details for customer C1002", "expected_tool": "customer_lookup"},
    {"id": 11, "request": "Look up customer C1003", "expected_tool": "customer_lookup"},
    {"id": 12, "request": "Give me information about C1001", "expected_tool": "customer_lookup"},
    {"id": 13, "request": "Find client C1002", "expected_tool": "customer_lookup"},
    {"id": 14, "request": "Who is customer C1003?", "expected_tool": "customer_lookup"},
    {"id": 15, "request": "Retrieve customer C1001", "expected_tool": "customer_lookup"},
    {"id": 16, "request": "Get customer profile C1002", "expected_tool": "customer_lookup"},

    # Orders
    {"id": 17, "request": "What is the status of order O1001?", "expected_tool": "order_lookup"},
    {"id": 18, "request": "Check order O1002", "expected_tool": "order_lookup"},
    {"id": 19, "request": "Find order O1003", "expected_tool": "order_lookup"},
    {"id": 20, "request": "Show me order O1001", "expected_tool": "order_lookup"},
    {"id": 21, "request": "What happened to shipment O1002?", "expected_tool": "order_lookup"},
    {"id": 22, "request": "Retrieve order O1003", "expected_tool": "order_lookup"},
    {"id": 23, "request": "Give me details of order O1001", "expected_tool": "order_lookup"},
    {"id": 24, "request": "Is O1002 delivered?", "expected_tool": "order_lookup"},

    # Web
    {"id": 25, "request": "Search the web for Nokia AnyJev", "expected_tool": "web_search"},
    {"id": 26, "request": "Search for the latest AI news", "expected_tool": "web_search"},
    {"id": 27, "request": "Look up current information about Nokia", "expected_tool": "web_search"},
    {"id": 28, "request": "Search online for AnyJev documentation", "expected_tool": "web_search"},
    {"id": 29, "request": "Find the latest information on agent tool calling", "expected_tool": "web_search"},
    {"id": 30, "request": "Use web search to find current AI trends", "expected_tool": "web_search"},

    # None
    {"id": 31, "request": "Hello", "expected_tool": "none"},
    {"id": 32, "request": "Explain what an AI agent is", "expected_tool": "none"},
    {"id": 33, "request": "What is RAG?", "expected_tool": "none"},
    {"id": 34, "request": "Explain tool calling", "expected_tool": "none"},
    {"id": 35, "request": "What is machine learning?", "expected_tool": "none"},
    {"id": 36, "request": "Explain the difference between AI and ML", "expected_tool": "none"},
]
