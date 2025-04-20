from flask import Flask, request, render_template, jsonify
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"],
)

app = Flask(__name__)

# Conversation history
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    conversation_history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        messages=conversation_history,
        model="gpt-4o",
        temperature=1,
        max_tokens=4096,
        top_p=1,
    )

    reply = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": reply})

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
