from flask import Flask, render_template, request, redirect, jsonify, url_for

# Import the chatbot from chatbot.py
from chatbot import Chatbot

# Run the website with Flask
# =========================
app = Flask(__name__)
chatbot = Chatbot("chatbot_model.h5")

@app.template_filter()
def numberFormat(value):
    value = int(value)
    return format(value, ',d')

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html", chats=chatbot.chats)

@app.route('/get', methods=['GET'])
def chat():
    query = request.args.get('message')
    user_received = chatbot.add_chat("user", query)
    response, items = chatbot.chat(query)
    bot_response = chatbot.add_chat("bot", response, items=items)
    bot_response['user_received_time'] = user_received['time']
    return jsonify(bot_response)

@app.route('/reset', methods=['POST'])
def reset():
    chatbot.reset()
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True, port=5000)