import streamlit as st

lab1 = st.Page(
    'labs/lab1.py',
    title = 'Lab 1',
)

lab2 = st.Page(
    'labs/lab2.py',
    title = 'Lab 2',
    default = True
)

pg = st.navigation([lab1, lab2])
st.set_page_config(page_title='IST 688 Labs', page_icon=None, layout="centered", initial_sidebar_state="expanded")
pg.run()

