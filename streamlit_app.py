import streamlit as st

lab1 = st.Page(
    'labs/lab1.py',
    title = 'Lab 1',
)

lab2 = st.Page(
    'labs/lab2.py',
    title = 'Lab 2',
)

lab3 = st.Page(
    'labs/lab3.py',
    title = 'Lab 3',
)

lab4 = st.Page(
    'labs/lab4.py',
    title = 'Lab 4',
)

lab5 = st.Page(
    'labs/lab5.py',
    title = 'Lab 5',
    default = True
)

pg = st.navigation([lab1, lab2, lab3, lab4, lab5])
st.set_page_config(page_title='IST 688 Labs', page_icon=None, layout="centered", initial_sidebar_state="expanded")
pg.run()

