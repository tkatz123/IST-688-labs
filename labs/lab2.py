import streamlit as st
from openai import OpenAI, AuthenticationError

# Show title and description.
st.title("MY Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

#Get OPENAI API Key from secrets file
openai_api_key = st.secrets.OPENAI_API_KEY

try:
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    client.models.list()

except AuthenticationError:
    st.info("🚨 Invalid OpenAI API Key")
    st.stop()

#Ask the user how they want their document summarized
summarize_option = st.sidebar.selectbox(
    'How would you like your document to be summarized?',
    ('100 words', '2 connecting paragraphs', '5 bullet points')
)

#Define a prompt for summarization based on option selected above
if summarize_option == '100 words':
    summarize_prompt = 'Summarize this document in 100 words or less. NEVER exceed the 100 word summary limit.'
elif summarize_option == '2 connecting paragraphs':
    summarize_prompt = 'Summarize this document in two individual connecting paragraphs. NEVER exceed the 2 paragraph limit. MAKE SURE the paragraphs flow into one another.'
elif summarize_option == '5 bullet points':
    summarize_prompt = 'Summarize this document in 5 individual bullet points. NEVER exceed the five bullet point limit.'

#Let the user select which model they want to use
if st.sidebar.checkbox('Use advanced model'):
    model = 'gpt-5.4-mini'
else:
    model = 'gpt-5.4-nano'

# Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
    "Upload a document (.txt or .md)", type=("txt", "md")
)

if uploaded_file:

    # Process the uploaded file and question.
    document = uploaded_file.read().decode()
    messages = [
        {
            "role": "user",
            "content": f"Here's a document: {document} \n\n---\n\n {summarize_prompt}",
        }
    ]

    # Generate an answer using the OpenAI API.
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    # Stream the response to the app using `st.write_stream`.
    st.write_stream(stream)
