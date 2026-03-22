# Axe Server (FastAPI)

For teams that prefer to interact with Axetract via HTTP, we provide a lightweight FastAPI server wrapper.

## Quick Start

The server is located in the `axe_server` directory.

### 1. Installation

Ensure you have the required dependencies:

```bash
pip install fastapi uvicorn
```

### 2. Launching

Run the server using Python:

```bash
cd axe_server
python main.py
```

By default, the server will be available at `http://localhost:8000`. You can access the interactive Swagger documentation at `http://localhost:8000/docs`.

## Configuration

The server supports several environment variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `AXE_USE_VLLM` | Enable vLLM background engine | `False` |
| `AXE_PORT` | Port to bind the server | `8000` |
| `AXE_HOST` | Host to bind the server | `0.0.0.0` |

## API Endpoints

### `POST /process`
Process a single extraction request.

**Request Body:**
```json
{
  "input_data": "https://example.com",
  "query": "Extract the main product title"
}
```

### `POST /process_batch`
Process multiple requests in a single call.

**Request Body:**
```json
{
  "items": [
    {
      "input_data": "https://site-a.com",
      "query": "Extract price"
    },
    {
      "input_data": "https://site-b.com",
      "query": "Extract description"
    }
  ]
}
```

## Client Example

We provide a `client_example.py` in the `axe_server` folder showing how to interface with the API using the `requests` library.
