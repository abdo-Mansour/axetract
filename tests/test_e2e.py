import os
from axetract.preprocessor.axe_preprocessor import AXEPreprocessor
from axetract.data_types import AXESample
from axetract.pruner.axe_pruner import AXEPruner
from axetract.llm.vllm_client import LocalVLLMClient
from axetract.extractor.axe_extractor import AXEExtractor
from axetract.prompts.pruner_prompt import PRUNER_PROMPT
from axetract.prompts.qa_prompt import QA_PROMPT
from axetract.prompts.schema_prompt import SCHEMA_PROMPT

def main():
    # --- 1. DUMMY DATA SETUP ---
    dummy_html = """
    <html>
        <head><title>Product Page</title></head>
        <body>
            <nav><ul><li>Home</li><li>Products</li></ul></nav>
            <div id="main-content" class="container" style="background: #fff;">
                <h1>SuperWidget 3000</h1>
                <p class="description">The SuperWidget 3000 is the best tool for <b>AI engineers</b>. 
                It features high-performance processing and low latency.</p>
                <table class="specs-table">
                    <tr><th>Feature</th><th>Value</th></tr>
                    <tr><td>Weight</td><td>1.2kg</td></tr>
                    <tr><td>Price</td><td>$299</td></tr>
                </table>
                <div class="reviews">
                    <h2>User Reviews</h2>
                    <div class="review">"Life changing tool!" - Jane Doe</div>
                    <div class="review">"Fast but expensive." - John Smith</div>
                </div>
            </div>
            <footer>Copyright 2025 WidgetCorp</footer>
        </body>
    </html>
    """
    query = "Extract the price and weight of the SuperWidget 3000 from the following context."

    # --- 2. DATA TYPE INITIALIZATION ---
    test_sample = AXESample(
        id="1",
        query=query,
        is_content_url=False,
        content=dummy_html
    )

    # --- 3. PREPROCESSING (Cleaning & Chunking) ---
    # use_clean_chunker=True is added to avoid the ValueError seen earlier
    preprocessor = AXEPreprocessor(use_clean_chunker=True)
    processed_samples = preprocessor([test_sample])
    
    print(f"Successfully created {len(processed_samples[0].chunks)} chunks.")

    # --- 4. LLM CLIENT INITIALIZATION (vLLM) ---
    # Optimized settings for WSL/limited VRAM
    vllm_config = {
        'model_name': 'Qwen/Qwen3-0.6B',
        'max_tokens': 1024,
        'engine_args': {
            "gpu_memory_utilization": 0.8, # Increased to use more available VRAM
            "max_model_len": 1024,          # Reduced from 6000 to fit in memory
            "enable_lora": True,
            "max_loras": 3,                 # Matches your 3 specific adapters
            "max_lora_rank": 64,            # Fixes Error: LoRA rank 64 is greater than max_lora_rank 16
            "disable_log_stats": True,
        },
        'generation_config': {
            'temperature': 0.0,
            'top_p': 1.0,
        },
        'lora_modules': {
            "pruner": {
                "path": "abdo-Mansour/Pruner_Adaptor_Qwen_3_FINAL_EXTRA",
                "temperature": 0.0  
            },
            "qa": {
                "path": "abdo-Mansour/Extractor_Adaptor_Qwen3_QA_websrc",
                "temperature": 1.0 
            },
            "schema": {
                "path": "abdo-Mansour/Extractor_Adaptor_Qwen3_Final",
                "temperature": 0.0  
            }
        },
    }

    print("Initializing Local vLLM Engine... (This may take a minute)")
    lc = LocalVLLMClient(config=vllm_config)

    # --- 5. PRUNER INITIALIZATION ---
    pru = AXEPruner(
        llm_pruner_client=lc,
        llm_pruner_prompt=PRUNER_PROMPT,
    )

    pruned_samples = pru(processed_samples)
    print("Pruning completed successfully.")
    print(pruned_samples)

    # --- 6. EXTRACTOR INITIALIZATION ---
    ext = AXEExtractor(
        llm_extractor_client=lc,
        schema_generation_prompt_template=SCHEMA_PROMPT,
        query_generation_prompt_template=QA_PROMPT,
    )

    extracted_samples = ext(pruned_samples)
    print("Extraction completed successfully.")
    print(extracted_samples)

    print("Pipeline initialized successfully.")
    
    # Optional: Test execution (Uncomment if you want to run a test immediately)
    # result = ext.extract(processed_samples[0])
    # print(result)

if __name__ == "__main__":
    main()