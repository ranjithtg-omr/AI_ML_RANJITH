from flask import Flask, render_template, request, jsonify

from chatBot_pdf import rag_chat

app = Flask(__name__)


# =========================================================+
# YOUR CHATBOT FUNCTION
# =========================================================
def chatbot_response(user_message):

    response = rag_chat(user_message)

    return response


# =========================================================
# HOME PAGE
# =========================================================
@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# CHAT API
# =========================================================
@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    user_message = data.get("message", "")

    bot_response = chatbot_response(user_message)

    return jsonify({
        "response": bot_response
    })


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)