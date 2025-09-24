import streamlit as st
from chatbot2 import TrulyAutonomousAgent  # import your chatbot class

# Initialize chatbot only once (store in session state)
if "agent" not in st.session_state:
    st.session_state.agent = TrulyAutonomousAgent()

st.title("🤖 NeoSage: An enhanced Autonomous Chatbot")
st.subheader("Connected with Gemini API!")

# Chat input
user_query = st.text_input("Ask me something:")

if st.button("Submit") and user_query.strip():
    with st.spinner("Thinking..."):
        try:
            # Call your agent
            result = st.session_state.agent.solve_autonomously(user_query)

            st.subheader("Reasoning")
            st.write(result.get("reasoning"))

            st.subheader("Code")
            for code_snippet in result.get("code", []):
                st.code(code_snippet, language="python")

            st.subheader("Final Answer")
            st.success(result.get("final_answer"))



        except Exception as e:
            st.error(f"⚠️ Error: {e}")
