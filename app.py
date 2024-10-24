
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

my_key_openai = st.secrets["mykey_openai"]

llm_openai = ChatOpenAI(api_key = my_key_openai, model = "gpt-4o", temperature=0.2, max_tokens=None)
embeddings = OpenAIEmbeddings(api_key = my_key_openai, model="text-embedding-3-large")


st.set_page_config(page_title="AI Lawyer Chatbot", page_icon="🤖", layout="centered")
st.title("AI Lawyer Chatbot :robot_face:")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": "sən, hüquqi mövzuları sadə dildə izah edən robotsan."})
    

def get_final_prompt(prompt):

    new_vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
    )

    retriever = new_vector_store.as_retriever()

    relevant_documents = retriever.invoke(prompt)

    context_data = ""

    for document in relevant_documents:
        context_data = context_data + " " + document.page_content
    
    final_prompt = f"""
    Sənə bir sual verəcəm və cavablandırmaq üçün Azərbaycan Respublikasının qanunvericiliyinə aid məlumatlar təqdim edəcəm. Cavabı hazırlayarkən aşağıdakı təlimatlara əməl et:

    - Cavabını sadə dildə yaz.
    - Lazım olduqda, izahını nümunə ilə dəstəklə.
    - Cavabı mümkün qədər ətraflı və dolğun yaz.
    
    Sual: {prompt}
    Məlumatlar: {context_data}
    """

    return final_prompt

for message in st.session_state.messages[1:]:
   with st.chat_message(message["role"]):
      st.markdown(message["content"])

if prompt := st.chat_input("Sualınızı yazın..."):
    st.chat_message("user", avatar="👨🏻").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🤖"):
        response_text = ""
        response_placeholder = st.empty() 

        for token in llm_openai.stream(st.session_state.messages + [{"role": "user", "content": get_final_prompt(prompt)}]):
            token_content = token.content
            response_text += token_content
            response_placeholder.markdown(response_text) 
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("Created by **Azərbaycan Respublikasının Mərkəzi Bankı** © 2024", unsafe_allow_html=True)


