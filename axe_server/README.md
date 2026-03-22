# AXEtract Server

This folder contains a FastAPI server to run the AXEtract pipeline.

## Setup

Ensure you have the dependencies installed:

```bash
pip install fastapi uvicorn requests
```

## Running the Server

You can run the server using:

```bash
python main.py
```

By default, it runs on `http://0.0.0.0:8000`.

### Environment Variables

- `AXE_USE_VLLM`: Set to `True` to use vLLM for local serving (requires GPU and vLLM). Default: `False`.
- `AXE_PORT`: Port to run the server on. Default: `8000`.
- `AXE_HOST`: Host to bind the server to. Default: `0.0.0.0`.

## API Endpoints

### 1. Health Check
`GET /health`

### 2. Process Single Document
`POST /process`

Payload:
```json
{
  "input_data": "https://example.com",
  "query": "Extract the title",
  "schema_model": null
}
```

### 3. Process Batch
`POST /process_batch`

Payload:
```json
{
  "items": [
    {
      "input_data": "https://google.com",
      "query": "Extract the search button text"
    },
    {
      "input_data": "<html><body><h1>Hello</h1></body></html>",
      "query": "Extract the header"
    }
  ]
}
```
