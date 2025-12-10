import json
import glob
from flask import Flask, request, jsonify
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")
app = Flask(__name__)


# Load all dataset files automatically
def load_all_datasets():
    data = []
    for fname in glob.glob("datasets/*.jsonl"):
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except:
                    pass
    return data


dataset = load_all_datasets()


def find_best_example(user_text):
    best_item = None
    best_score = 0

    for item in dataset:
        question = item["input"].lower()
        score = sum(word in user_text.lower() for word in question.split())

        if score > best_score:
            best_score = score
            best_item = item

    return best_item


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]

    example = find_best_example(user_input)

    messages = [
        {
            "role": "system",
            "content": "You are a calm AI that helps people relax, think clearly, and find good solutions."
        },
        {"role": "user", "content": user_input}
    ]

    if example:
        messages.append({
            "role": "assistant",
            "content": f"Here is a sample calming response from your dataset: '{example['response']}'. Use this tone."
        })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    ai_text = response.choices[0].message["content"]
    return jsonify({"response": ai_text})


app.run(host="0.0.0.0", port=5000)