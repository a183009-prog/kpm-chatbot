import os
import re
import textwrap
import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# =====================================================
# CONFIG
# =====================================================
MODEL_ID = "mesolitica/Malaysian-Qwen2.5-7B-Instruct"
DATA_DIR = "data"
CHROMA_DIR = "chroma_kpm"
COLLECTION_NAME = "kpm_tahun1"

st.set_page_config(
    page_title="Chatbot Rasmi KPM – Tahun 1",
    layout="wide"
)

# =====================================================
# LOAD LLM
# =====================================================


@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    model.eval()
    return tokenizer, model


# =====================================================
# LOAD DOCUMENTS
# =====================================================
@st.cache_data
def load_documents(data_dir):
    documents = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = file
                    doc.metadata["category"] = os.path.basename(root)
                    doc.metadata["doc_id"] = f"{os.path.basename(root)}/{file}"

                documents.extend(docs)

    return documents


# =====================================================
# VECTOR STORE
# =====================================================
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(CHROMA_DIR):
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME
        )

    documents = load_documents(DATA_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )
    chunks = splitter.split_documents(documents)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )


# =====================================================
# PROMPT
# =====================================================
def build_prompt(context, question):
    return f"""
Anda ialah chatbot rasmi Kementerian Pendidikan Malaysia (KPM).

Arahan penting:
- Jawab HANYA soalan yang diberikan.
- Jawapan mestilah berdasarkan maklumat rujukan sahaja.
- Jika soalan berkaitan SYARAT atau KELAYAKAN, fokus kepada syarat sahaja.
- Jangan menyenaraikan maklumat selain yang diminta.
- Jangan menambah fakta atau tafsiran baharu.
- Jika jawapan berbentuk senarai, guna format bernombor.
- Jangan potong ayat.

Maklumat Rujukan:
{context}

Soalan:
{question}

Jawapan:
"""


# =====================================================
# FORMAT OUTPUT
# =====================================================
def format_answer(answer, width=80):
    formatted_lines = []

    for line in answer.splitlines():
        line = line.strip()
        if re.match(r"^\d+\.\s+", line):
            formatted_lines.append(line)
        else:
            formatted_lines.append(textwrap.fill(line, width))

    return "\n".join(formatted_lines)


# =====================================================
# ASK BOT
# =====================================================
def ask_kpm_bot(question, tokenizer, model, retriever):
    docs = retriever.invoke(question)

    if not docs:
        return "Maklumat berkaitan tidak dinyatakan dalam dokumen rasmi KPM."

    context = "\n\n".join(doc.page_content for doc in docs[:2])
    prompt = build_prompt(context, question)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False
    )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    marker = "Jawapan:"
    if marker in full_output:
        return full_output.split(marker)[-1].strip()

    return full_output.strip()


# =====================================================
# STREAMLIT UI
# =====================================================
st.title("Chatbot Rasmi KPM – Tahun 1")
st.caption("Berdasarkan dokumen rasmi Kementerian Pendidikan Malaysia")

tokenizer, model = load_llm()
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("Taip soalan berkaitan kemasukan Murid Tahun 1...")

if question:
    st.session_state.chat.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.spinner("Menjana jawapan rasmi KPM..."):
        answer = ask_kpm_bot(question, tokenizer, model, retriever)
        answer = format_answer(answer)

    st.session_state.chat.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
