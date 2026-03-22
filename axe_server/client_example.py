import json

import requests


def read_html_file(file_path: str) -> str:
    """Read and return the contents of an HTML file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def test_single():
    """Test processing a single URL."""
    url = "http://localhost:8000/process"
    data = {
        "input_data": "https://www.google.com",
        "query": "What is the text on the search button?",
    }
    response = requests.post(url, json=data)
    print("Single Request Response:")
    print(json.dumps(response.json(), indent=2))


def test_batch():
    """Test processing multiple URLs in a batch."""
    url = "http://localhost:8000/process_batch"
    data = {
        "items": [
            {"input_data": "https://www.google.com", "query": "Get search button text"},
            {"input_data": "https://www.github.com", "query": "What is the hero text?"},
        ]
    }
    response = requests.post(url, json=data)
    print("\nBatch Request Response:")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    # Note: Server must be running first!
    try:
        test_single()
        test_batch()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Is it running on http://localhost:8000?")
