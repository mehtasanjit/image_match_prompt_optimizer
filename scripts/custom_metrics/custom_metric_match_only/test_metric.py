import urllib.request
import json
import ssl

url = "https://us-central1-<PROJECT_ID>.cloudfunctions.net/<FUNCTION_NAME>"

test_cases = [
    {"name": "True Positive (Match)", "target": "Match", "response": '{"product_match": "Match"}'},
    {"name": "True Negative (Not_Match)", "target": "Not_Match", "response": '{"product_match": "Not_Match"}'},
    # Edge case with historical Mismatch labels
    {"name": "True Negative (Mismatch GT)", "target": "Mismatch", "response": '{"product_match": "Not_Match"}'}, 
    {"name": "False Positive", "target": "Not_Match", "response": '{"product_match": "Match"}'},
    {"name": "False Negative", "target": "Match", "response": '{"product_match": "Not_Match"}'},
    # Unexpected model outputs treated as Not_Match
    {"name": "Model outputs Inconclusive (GT Not_Match)", "target": "Not_Match", "response": '{"product_match": "inconclusive"}'},
    {"name": "Model outputs Inconclusive (GT Mismatch)", "target": "Mismatch", "response": '{"product_match": "Inconclusive"}'},
    {"name": "Model outputs Inconclusive (GT Match)", "target": "Match", "response": '{"product_match": "inconclusive"}'},
    {"name": "Model outputs random string (GT Not_Match)", "target": "Not_Match", "response": '{"product_match": "i dont know"}'},
    # GT Inconclusive, model outputs Match
    {"name": "Model outputs Match (GT Inconclusive)", "target": "inconclusive", "response": '{"product_match": "Match"}'},
    {"name": "Model outputs Match (GT Mismatch)", "target": "Mismatch", "response": '{"product_match": "Match"}'},
]

# Bypass SSL context if needed
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

print(f"Testing binary metric endpoint: {url}\n")
for tc in test_cases:
    payload = json.dumps({"target": tc["target"], "response": tc["response"]}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    
    print(f"Test: {tc['name']}")
    print(f"  Input Target:   {tc['target']}")
    print(f"  Input Response: {tc['response']}")
    try:
        with urllib.request.urlopen(req, context=ctx) as response:
            result = response.read().decode('utf-8')
            print(f"  Result Score:   {result.strip()}")
    except urllib.error.HTTPError as e:
        print(f"  Error {e.code}: {e.read().decode('utf-8')}")
    print("-" * 40)
