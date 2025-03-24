import os
import time
import requests
from typing import Dict, Any
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration for the Hugging Face models
# MODELS_CONFIG = {
#     "mistral": {
#         "name": "Mistral-7B",
#         "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
#         "temperature": 0.7,
#         "max_tokens": 512
#     },
#     "phi2": {
#         "name": "Phi-2",
#         "repo_id": "TheBloke/phi-2-GGUF",
#         "temperature": 0.3,
#         "max_tokens": 512
#     },
#     "tinyllama": {
#         "name": "TinyLlama",
#         "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
#         "temperature": 1.0,
#         "max_tokens": 512
#     }
# }
MODELS_CONFIG = {
    "mistral": {
        "name": "Mistral-7B-Instruct",
        "repo_id": "mistralai/Mistral-7B-Instruct-v0.1",
        "temperature": 0.7,
        "max_tokens": 512,
        "context_size": 4096
    },
    "phi2": {
        "name": "Phi-2",
        "repo_id": "gpt2",
        "temperature": 0.3,
        "max_tokens": 512,
        "context_size": 2048
    },
    "tinyllama": {
        "name": "gemma",
        "repo_id": "google/gemma-2-2b-it",
        "temperature": 0.6,
        "max_tokens": 512,
        "context_size": 2048
    }
}
def generate_response(model_id: str, prompt: str) -> Dict[str, Any]:
    start_time = time.time()

    if model_id not in MODELS_CONFIG:
        return {"text": "Invalid model ID", "model": model_id, "temperature": 0, "time_taken": 0, "relevance": 0}

    config = MODELS_CONFIG[model_id]
    hf_token = os.getenv("HF_TOKEN")

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": config["temperature"],
            "max_new_tokens": config["max_tokens"]
        }
    }

    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{config['repo_id']}",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        result = response.json()

        response_text = result[0]['generated_text'].replace(prompt, "").strip()
        time_taken = time.time() - start_time
        relevance = min(100, len(response_text) / 10)

        return {
            "text": response_text,
            "model": config["name"],
            "temperature": config["temperature"],
            "time_taken": round(time_taken, 2),
            "relevance": round(relevance)
        }

    except Exception as e:
        return {
            "text": f"Error: {str(e)}",
            "model": config["name"],
            "temperature": config["temperature"],
            "time_taken": round(time.time() - start_time, 2),
            "relevance": 0
        }

@app.route('/')
def index():
    return render_template('index.html', models=MODELS_CONFIG)

@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    results = {}
    for model_id in MODELS_CONFIG:
        results[model_id] = generate_response(model_id, prompt)

    return jsonify(results)

@app.route('/download_models')
def download_models():
    return render_template('download.html', models=MODELS_CONFIG)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)