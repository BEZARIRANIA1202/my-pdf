import os
import traceback
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="مساعد المستندات السريع", page_icon="⚡", layout="wide")
st.title("⚡ مساعد المستندات الفوري")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

embeddings = load_embeddings()

# --- القائمة الجانبية ---
st.sidebar.title("⚙️ الخيارات")

if "GOOGLE_API_KEY" in st.secrets and st.secrets["GOOGLE_API_KEY"]:
    clean_api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key_input = st.sidebar.text_input("أدخل Google API Key هنا:", type="password")
    clean_api_key = api_key_input.strip() if api_key_input else ""

if st.session_state["messages"]:
    st.sidebar.write("---")
    st.sidebar.subheader("💾 حفظ سجل المحادثة")
    
    chat_text = "=== سجل المحادثة والأسئلة ===\n\n"
    for i, msg in enumerate(st.session_state["messages"], 1):
        role_label = "❓ السؤال" if msg["role"] == "user" else "💡 الإجابة"
        chat_text += f"[{i}] {role_label}:\n{msg['content']}\n\n" + "-"*40 + "\n\n"

    st.sidebar.download_button(
        label="📥 تحميل سجل الأسئلة كامل (.txt)",
        data=chat_text.encode('utf-8'),
        file_name="chat_history.txt",
        mime="text/plain; charset=utf-8"
    )

    if st.sidebar.button("مسح المحادثة 🗑️"):
        st.session_state["messages"] = []
        st.rerun()

# --- التطبيق الرئيسي ---
if clean_api_key:
    os.environ["GOOGLE_API_KEY"] = clean_api_key

    uploaded_files = st.file_uploader("قم برفع ملفات PDF:", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        current_files_names = [f.name for f in uploaded_files]
        
        if "retriever" not in st.session_state or st.session_state.get("files_names") != current_files_names:
            with st.spinner("⚡ جاري تحليل وتقسيم المستندات..."):
                all_docs = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    temp_path = f"temp_doc_{idx}.pdf"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    
                    for d in docs:
                        cleaned = str(d.page_content).strip()
                        if cleaned:
                            d.page_content = cleaned
                            all_docs.append(d)
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_documents(all_docs)

                vectorstore = FAISS.from_documents(chunks, embeddings)
                
                st.session_state["retriever"] = vectorstore.as_retriever(search_kwargs={"k": 5})
                st.session_state["files_names"] = current_files_names
                st.session_state["messages"] = []
                
            st.success(f"✅ تم تحليل وتقسيم {len(uploaded_files)} ملفات بنجاح!")

    if "retriever" in st.session_state:
        st.write("---")
        
        for idx, msg in enumerate(st.session_state["messages"]):
            if msg["role"] == "user":
                st.markdown(f"**❓ السؤال:** {msg['content']}")
            else:
                st.markdown("### 💡 الإجابة:")
                st.write(msg["content"])
                st.write("---")

        with st.form(key="query_form", clear_on_submit=True):
            user_query = st.text_input("اطرح سؤالك بأي لغة (عربية، دارجة، فرنسية، إنجليزية...):")
            submit_button = st.form_submit_button(label="إرسال السؤال 🚀")

        if submit_button and user_query:
            st.session_state["messages"].append({"role": "user", "content": user_query})

            with st.spinner("⚡ جاري البحث والإجابة..."):
                try:
                    retriever = st.session_state["retriever"]
                    relevant_docs = retriever.invoke(user_query)
                    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        google_api_key=clean_api_key,
                        temperature=0.3
                    )

                    # تعليمات قوية للنموذج ليفهم الدارجة ويرد بنفس لغة المستخدم
                    prompt_template = ChatPromptTemplate.from_template(
                        "You are an intelligent multi-lingual assistant capable of understanding all languages, including dialects such as Algerian Darja, French, Arabic, and English.\n"
                        "Analyze the provided document context below and answer the user's question or request accurately.\n"
                        "CRITICAL RULE: Respond in the EXACT SAME language or dialect used by the user in their request (e.g., if the user asks in Darja, reply in clear Darja; if in French, reply in French; if in Arabic, reply in Arabic).\n\n"
                        "Document Context:\n{context}\n\n"
                        "User Request/Question: {question}"
                    )

                    chain = prompt_template | llm | StrOutputParser()
                    response_text = chain.invoke({"context": context_text, "question": user_query})

                    st.session_state["messages"].append({"role": "assistant", "content": response_text})
                    st.rerun()

                except Exception as e:
                    error_details = traceback.format_exc()
                    st.error(f"حدث خطأ: {e}")
                    st.code(error_details, language="python")

else:
    st.warning("يرجى إدخال Google API Key في القائمة الجانبية للبدء.")
