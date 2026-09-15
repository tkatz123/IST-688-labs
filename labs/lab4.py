import sys

# Swap in pysqlite3 before chromadb loads (needed on Streamlit Cloud; local SQLite is new enough)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
from openai import OpenAI, AuthenticationError
import chromadb
from pathlib import Path
from pypdf import PdfReader

chroma_client = chromadb.PersistentClient(path = './ChromaDB_for_Lab')
collection = chroma_client.get_or_create_collection('Lab4Collection')

if 'open_ai_client' not in st.session_state:
    #Get OPENAI API Key from secrets file
    openai_api_key = st.secrets.OPENAI_API_KEY
    st.session_state.open_ai_client = OpenAI(api_key=openai_api_key)

def add_to_collection(collection, text, file_name):

    client = st.session_state.open_ai_client

    response = client.embeddings.create(
        input = text,
        model = 'text-embedding-3-small'
    )

    embedding = response.data[0].embedding

    collection.add(
        documents = [text],
        ids = [file_name],
        embeddings = [embedding]
    )

def extract_text_from_pdf(pdf_path):
    #Extracting the pages from the uploaded file object
    pages = PdfReader(pdf_path).pages

    pages_text = []

    #Iterate over each page
    for page in pages:

        #Extract text from page
        text = page.extract_text()

        #Append page text to list
        pages_text.append(text)

    # Once loop is done running, combine the text of all the pages together
    return " ".join(pages_text)

def load_pdfs_to_collection(folder_path, collection):

    for file_path in Path(folder_path).glob('*.pdf'):

        file_text = extract_text_from_pdf(file_path)
        add_to_collection(collection, file_text, file_path.stem)
        


if collection.count() == 0:
    loaded = load_pdfs_to_collection('data/Lab-04-Data', collection)

# Show title and description.
st.title("Lab 4: Chatbot Using RAG")

st.divider()

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {'role': 'assistant', 'content': 'Hello there! What can I help you with?'}
    ]

try:

    st.session_state.open_ai_client.models.list()

except AuthenticationError:
    st.info("🚨 Invalid OpenAI API Key")
    st.stop()

#Display every message in the conversation so far
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

user_message = st.chat_input('Ask a question about the course documents...')

if user_message:

    st.session_state.messages.append({'role': 'user', 'content': user_message})

    with st.chat_message('user'):
        st.write(user_message)

    client = st.session_state.open_ai_client

    #Embed the user's question with the same model used for the documents
    response = client.embeddings.create(
        input = user_message,
        model = 'text-embedding-3-small'
    )

    query_embedding = response.data[0].embedding

    #Find the documents most similar to the question
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results = 3
    )

    context = "\n\n".join(results['documents'][0])

    system_prompt = (
        'You are a helpful assistant for a course. Answer the user\'s question using only '
        'the course documents below. If the answer is not in the documents, say that you '
        'could not find it in the course documents.\n\n'
        f'Course documents:\n\n{context}'
    )

    #System prompt with the retrieved documents, then the conversation (skipping the greeting)
    conversation = [{'role': 'system', 'content': system_prompt}] + st.session_state.messages[1:]

    # Generate an answer using the OpenAI API.
    stream = client.chat.completions.create(
        model='gpt-5-mini',
        messages=conversation,
        stream=True,
    )

    with st.chat_message('assistant'):
        answer = st.write_stream(stream)
        st.caption('Sources: ' + ', '.join(results['ids'][0]))

    st.session_state.messages.append({'role': 'assistant', 'content': answer})

if st.sidebar.button('Clear conversation'):
    st.session_state.messages = st.session_state.messages[:1]
    st.rerun()
