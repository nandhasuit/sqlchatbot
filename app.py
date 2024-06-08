import streamlit as st
import database
from database import init_database
from langchain_code import get_response
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

def connection_page():
    st.title("SQL Chat Bot - Connection")

    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")

    host = st.text_input("Host", value="localhost")
    port = st.text_input("Port", value="3306")
    user = st.text_input("User", value="root")
    password = st.text_input("Password", type="password", value="admin")
    database = st.text_input("Database", value="tgf")

    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(user, password, host, port, database)
            st.session_state.db = db
            st.session_state.connect = True
            st.success("Connected to database!")
            st.rerun()

def chat_page():
    st.title("SQL Chat Bot - Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
        ]

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    user_query = st.chat_input("Type a message...")
    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
            st.markdown(response)

        st.session_state.chat_history.append(AIMessage(content=response))

def main():
    pages = {
        "Connection": connection_page,
        "Chat": chat_page,
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    pages[selection]()

if __name__ == "__main__":
    load_dotenv()
    main()
