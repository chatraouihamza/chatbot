from flask import Flask, render_template, request, session
from datetime import datetime
from chat import loaded_model,classify,preprocess_input,get_response
from util import tokenize, bag_of_words
app = Flask(__name__)
app.secret_key = '20022'

@app.route('/')
def index():
    # Initialize the session messages if not already present
    if 'user_messages' not in session:
        session['user_messages'] = []
    if 'bot_messages' not in session:
        session['bot_messages'] = []

    chatbox_visible = session.get('chatbox_visible', False)
    return render_template(
        'index.html',
        user_messages=session['user_messages'],
        bot_messages=session['bot_messages'],
        chatbox_visible=chatbox_visible
    )

@app.route('/send', methods=['POST'])
def send_message():
    user_message = request.form.get('message')
    
    if not user_message:
        return render_template(
            'index.html',
            messages=session.get('messages', []),
            error="Message field is empty",
            chatbox_visible=session.get('chatbox_visible', True)
        )
    
    bot_response = generate_bot_response(user_message)  # Generate a bot response

    # Append the user message and the bot response to the same message list
    session['messages'] = session.get('messages', [])
    
       # Store the user message with its type
       # session['messages'].append({'sender': 'user', 'text': user_message})
    
       # # Store the bot response with its type
       # session['messages'].append({'sender': 'bot', 'text': bot_response})
      # Get current time (Hour and Minute only)
    timestamp = datetime.now().strftime("%H:%M")     
    # Append the user message and the bot response with timestamp
    messages = session.get('messages', [])
    messages.append({'sender': 'user', 'text': user_message, 'time': timestamp})
    messages.append({'sender': 'bot', 'text': bot_response, 'time': timestamp})
    
    return render_template(
        'index.html',
        messages=session['messages'],
        chatbox_visible=session.get('chatbox_visible', True)
    )

@app.route('/toggle', methods=['POST'])
def toggle_chatbox():
    # Toggle the chatbox visibility
    chatbox_visible = session.get('chatbox_visible', True)
    session['chatbox_visible'] = not chatbox_visible
    
    # Clear the messages from session when toggling off the chatbox
    if not session['chatbox_visible']:
        session.pop('messages', None)
        
    
    return render_template(
        'index.html',
        user_messages=session.get('user_messages', []),
        bot_messages=session.get('bot_messages', []),
        chatbox_visible=session['chatbox_visible']
    )

def generate_bot_response(user_message):
    # Here you would add your bot's logic; for now, we respond with a simple message
    return get_response(user_message)

if __name__ == '__main__':
    app.run(debug=True)
