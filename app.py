from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from huggingface_hub import InferenceClient

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load Hugging Face API token from environment variable
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("HF_API_TOKEN is not set. Export it before running the app.")

# Initialize Hugging Face client
client = InferenceClient(api_key=HF_API_TOKEN)

# Model to use
MODEL_ID = "NousResearch/Hermes-4-70B"

# Home route
@app.route("/")
def home():
    return render_template("chatbot.html")  # Make sure you have chatbot.html in templates/

# Chat API route
@app.route("/chat", methods=["POST"])
def chat():
    # Get user message from request
    user_message = request.get_json().get("message", "").strip()
    if not user_message:
        return jsonify({"reply": "Please enter a message."})

    try:
        # Send message to Hugging Face model
        response = client.chat_completion(
            model=MODEL_ID,
            messages=[{"role": "user", "content": user_message}]
        )
        # Extract bot reply
        bot_reply = response["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ Hugging Face API error:", e)
        bot_reply = f"Error: Unable to reach AI service. ({e})"

    return jsonify({"reply": bot_reply})

# Run app
if __name__ == "__main__":
    print("🚀 Flask app running on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
