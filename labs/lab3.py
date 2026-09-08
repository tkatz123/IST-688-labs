import streamlit as st
import tiktoken
from openai import OpenAI, AuthenticationError

#Rough overhead per message for the role/formatting the API wraps around the content
TOKENS_PER_MESSAGE = 4

#How many complete user/assistant exchanges to keep when not using the token buffer
BUFFER_EXCHANGES = 2

#Behaviour rules for the bot, always sent as the first message of the conversation
SYSTEM_PROMPT = (
    'You are a friendly tutor chatting with a 10 year old. '
    'Explain everything in simple words a 10 year old can understand: short sentences, '
    'everyday examples, and no jargon unless you immediately explain it. '
    '\n\n'
    'Follow this conversation flow exactly:\n'
    '1. When the user asks a question, answer it, then end your reply by asking '
    '"Do you want more info?"\n'
    '2. If the user says yes (or anything meaning yes), give another piece of information '
    'about the same topic, then again end by asking "Do you want more info?"\n'
    '3. Keep repeating step 2 for as long as the user keeps saying yes. Each time, share '
    'something new instead of repeating what you already said.\n'
    '4. If the user says no (or anything meaning no), stop giving more info about that topic '
    'and ask what else you can help them with.\n'
    '5. If the user asks a brand new question at any point, answer it and restart at step 1.'
)


@st.cache_resource
def get_encoder():
    #o200k_base is the encoding used by the current GPT models
    return tiktoken.get_encoding('o200k_base')


def count_tokens(message):
    return len(get_encoder().encode(message['content'])) + TOKENS_PER_MESSAGE


def token_buffer(messages, max_tokens):
    #The system prompt is never trimmed, so it comes off the budget first
    system_message = messages[0]
    system_tokens = count_tokens(system_message)
    history = messages[1:]

    #Walk backwards from the newest message, keeping whatever fits in the budget
    kept = []
    total = 0

    for message in reversed(history):
        cost = count_tokens(message)
        if system_tokens + total + cost > max_tokens:
            break
        kept.append(message)
        total += cost

    kept.reverse()

    #The conversation the model sees should never open on an assistant reply
    while kept and kept[0]['role'] == 'assistant':
        total -= count_tokens(kept.pop(0))

    #Always send the newest message, even if it blows the budget on its own
    if not kept:
        kept = history[-1:]
        total = count_tokens(kept[0])

    return [system_message] + kept, system_tokens + total

# Show title and description.
st.title("ChatBot")


if 'client' not in st.session_state:
    #Get OPENAI API Key from secrets file
    openai_api_key = st.secrets.OPENAI_API_KEY
    st.session_state.client = OpenAI(api_key=openai_api_key)

if 'messages' not in st.session_state:
    #The system prompt is always the first thing in the queue
    st.session_state.messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]

try:

    st.session_state.client.models.list()

except AuthenticationError:
    st.info("🚨 Invalid OpenAI API Key")
    st.stop()

if st.sidebar.checkbox('Token based buffer'):
    token_based_buffer = True
    max_tokens = st.sidebar.slider(label= 'Select max tokens', min_value = 10, max_value = 10000, value = 1000)
else:
    token_based_buffer = False

#Display every message in the conversation so far, apart from the system prompt
for message in st.session_state.messages[1:]:
    with st.chat_message(message['role']):
        st.write(message['content'])

if prompt := st.chat_input('Whats up?'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('user'):
        st.write(prompt)


    #Decide how much of the conversation to send back to the model
    if token_based_buffer:
        conversation, used_tokens = token_buffer(st.session_state.messages, max_tokens)
        st.sidebar.caption(
            f'Sending {len(conversation)} of {len(st.session_state.messages)} messages '
            f'({used_tokens}/{max_tokens} tokens)'
        )
    else:
        #The system prompt, then the newest user message plus BUFFER_EXCHANGES complete exchanges before it
        conversation = st.session_state.messages[:1] + st.session_state.messages[1:][-(2 * BUFFER_EXCHANGES + 1):]

    # Generate an answer using the OpenAI API.
    stream = st.session_state.client.chat.completions.create(
        model='gpt-5.4-mini',
        messages=conversation,
        stream=True,
    )

    with st.chat_message('assistant'):
        response = st.write_stream(stream)

    st.session_state.messages.append({'role': 'assistant', 'content': response})

if st.sidebar.button('Clear conversation'):
    st.session_state.clear()
    st.rerun()
