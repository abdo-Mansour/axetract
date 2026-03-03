import os
from axetract.pipeline import AXEPipeline

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
    query = "what is the weight and the price?"
    schema = """{
        "price": "string",
        "weight": "string"
    }"""

    # --- 2. PIPELINE INITIALIZATION ---
    print("Initializing Professional Pipeline... (This may take a minute)")
    pipeline = AXEPipeline.from_config(use_vllm=False)
    
    # --- 3. PROCESSING ---
    print("Processing dummy sample...")
    result = pipeline.process(
        input_data=dummy_html,
        query=query,
        schema=schema
    )

    print("Pipeline completed successfully.")
    print("Result ID:", result.id)
    print("Prediction:", result.prediction)
    print("XPaths:", result.xpaths)
    print("Status:", result.status)
    if result.error:
        print("Error:", result.error)

if __name__ == "__main__":
    main()