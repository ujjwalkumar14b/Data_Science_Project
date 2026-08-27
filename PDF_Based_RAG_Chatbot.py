import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
chat_history = []
vectorstore = None


def extract_pdf_text(pdf_paths):
    text = ""

    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text

def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    return splitter.split_text(text)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=chunks,embedding=embeddings)
    return vectorstore

def ask_rag(question):
    global vectorstore

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
        You are a PDF assistant.

        Answer only from the provided context.
        If the answer is not in the context, say:
        "I could not find that information in the uploaded PDFs."

        Context:
        {context}

        Question:
        {question}
        """
    llm = ChatOllama(model="llama3", temperature=0)
    response = llm.invoke(prompt)
    return response.content


@app.route("/")
def home():
    return render_template(
        "index.html",
        chat_history=chat_history,
        pdf_files=os.listdir(UPLOAD_FOLDER)
    )

@app.route("/upload", methods=["POST"])
def upload():
    global vectorstore
    files = request.files.getlist("pdf_docs")
    pdf_paths = []

    if not files:
        return redirect("/")

    for file in files:
        if file.filename == "":
            continue
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER,filename)

        file.save(path)
        pdf_paths.append(path)

    raw_text = extract_pdf_text(pdf_paths)
    chunks = create_chunks(raw_text)
    vectorstore = create_vectorstore(chunks)

    return redirect("/")

@app.route("/ask", methods=["POST"])
def ask():
    global vectorstore
    question = request.form["question"]

    if vectorstore is None:
        answer = ("Please upload and process PDFs first.")
    else:
        answer = ask_rag(question)

    chat_history.append({"question": question,"answer": answer})
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)