"""
main.py - Mobile-friendly AI backend (Flask)
Features:
- POST /chat   -> returns AI response (uses OpenAI if OPENAI_API_KEY is set)
- POST /upload -> upload a .jsonl dataset file (adds to datasets/)
- Auto-loads all datasets/*.jsonl on each request (so new files work immediately)
"""

import os
import json
import glob
import time
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# If you have OpenAI Python client available, this will be used if OPENAI_API_KEY is set.
# The code below expects the same style used earlier:
# from openai import OpenAI
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# But we import lazily so server still runs without the openai package.
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

DATA_DIR = "datasets"
ALLOWED_EXT = {".jsonl"}
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB uploads limit


def load_all_datasets():
    """Load all JSONL files under datasets/ and return list of dicts with 'input' and 'response'."""
    data = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.jsonl")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        # Accept either {"input":..., "response":...} or {"prompt":..., "completion":...}
                        if "input" in obj and "response" in obj:
                            data.append({"input": obj["input"], "response": obj["response"], "source_file": os.path.basename(path)})
                        elif "prompt" in obj and "completion" in obj:
                            data.append({"input": obj["prompt"], "response": obj["completion"], "source_file": os.path.basename(path)})
                    except Exception:
                        # skip invalid lines silently
                        continue
        except Exception:
            continue
    return data


def find_best_example(user_text, dataset):
    """Return best matching example from dataset using simple word overlap scoring."""
    if not dataset:
        return None

    user_words = set(w for w in user_text.lower().split() if len(w) > 1)
    best = None
    best_score = 0
    for item in dataset:
        q_words = set(w for w in item["input"].lower().split() if len(w) > 1)
        score = len(user_words & q_words)
        # small bonus if dataset input is substring of user_text
        if item["input"].lower() in user_text.lower():
            score += 2
        if score > best_score:
            best_score = score
            best = item

    return best


def openai_reply(user_text, example):
    """Send messages to OpenAI if API key exists. Returns the AI text or raises on errors."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI")
    if not api_key:
        raise RuntimeError("No OpenAI API key found in environment.")

    if OpenAI is None:
        raise RuntimeError("OpenAI client library not installed. Install 'openai' package or set no API key to use local responses.")

    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": "You are a calm, friendly assistant. Help reduce frustration and suggest simple solutions."},
        {"role": "user", "content": user_text}
    ]

    if example:
        example_prompt = f"When someone said: '{example['input']}', a good calming reply was: '{example['response']}'. Use that tone and help the user."
        messages.append({"role": "assistant", "content": example_prompt})

    # Attempt to call chat completion (keeps same style as earlier examples)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        # Access assistant text
        ai_text = response.choices[0].message["content"]
        return ai_text
    except Exception as e:
        # bubble up error
        raise RuntimeError(f"OpenAI request failed: {e}")


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True, silent=True)
    if not payload or "message" not in payload:
        return jsonify({"error": "Please send JSON with a 'message' field."}), 400

    user_message = str(payload["message"]).strip()
    if not user_message:
        return jsonify({"error": "Empty message."}), 400

    # Reload datasets on each request (simple, reliable)
    dataset = load_all_datasets()
    example = find_best_example(user_message, dataset)

    # Try OpenAI if key present and client available
    try:
        ai_text = None
        if os.environ.get("OPENAI_API_KEY") and OpenAI is not None:
            try:
                ai_text = openai_reply(user_message, example)
                source = "openai"
            except Exception as e:
                # If OpenAI fails, fall back to local dataset
                ai_text = None
                source = f"openai_error: {str(e)}"
        if ai_text is None:
            # Local fallback: use example response if found, else default message
            if example:
                ai_text = example["response"]
                source = "local_dataset"
            else:
                ai_text = "I'm here to help. Tell me more about what's bothering you and I'll try to help."
                source = "local_default"

        return jsonify({"response": ai_text, "source": source})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Upload a new dataset .jsonl file from the mobile app.
    Expecting 'file' form-data field (multipart/form-data).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part. Please send form-data with key 'file'."}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    filename = secure_filename(f.filename)
    _, ext = os.path.splitext(filename)
    if ext.lower() not in ALLOWED_EXT:
        return jsonify({"error": f"Only {ALLOWED_EXT} allowed."}), 400

    save_path = os.path.join(DATA_DIR, filename)
    f.save(save_path)
    # Optionally validate file by trying to parse first lines (light check)
    good = False
    try:
        with open(save_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 5:
                    break
                if not line.strip():
                    continue
                obj = json.loads(line)
                if ("input" in obj and "response" in obj) or ("prompt" in obj and "completion" in obj):
                    good = True
                    break
    except Exception:
        good = False

    if not good:
        return jsonify({"warning": "File saved but seems not to have expected JSONL format (no input/response or prompt/completion found in first lines)."}), 200

    return jsonify({"ok": True, "file": filename, "message": "Uploaded and saved."})


@app.route("/datasets", methods=["GET"])
def list_datasets():
    """Return list of dataset file names and total example count."""
    files = []
    total = 0
    for path in glob.glob(os.path.join(DATA_DIR, "*.jsonl")):
        name = os.path.basename(path)
        count = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for _ in f:
                    count += 1
        except Exception:
            count = 0
        files.append({"file": name, "lines": count})
        total += count
    return jsonify({"files": files, "total_examples": total})


@app.route("/datasets/<path:filename>", methods=["GET"])
def download_dataset(filename):
    """Allow downloading dataset files if needed (useful for debugging)."""
    return send_from_directory(DATA_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    # Helpful startup message
    port = int(os.environ.get("PORT", 5000))
    print("Starting server ...")
    print(f"Dataset folder: {os.path.abspath(DATA_DIR)}")
    if os.environ.get("OPENAI_API_KEY"):
        print("OpenAI API key found in environment -> server will call OpenAI.")
    else:
        print("No OpenAI API key found -> server will use local dataset responses only.")
    print(f"Listening on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
