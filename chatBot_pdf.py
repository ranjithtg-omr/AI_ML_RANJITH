# ----------------------------------------
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import json

# ----------------------------------------
#Extarct text from PDF
def extract_pdf_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

pdf_text = extract_pdf_text("Loan_Details.pdf")
print(pdf_text)


# ----------------------------------------
#Split into chunks
def split_text(text, max_length=500):
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        if len(" ".join(chunk + [word])) > max_length:
            chunks.append(" ".join(chunk))
            chunk = []
        chunk.append(word)

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

chunks = split_text(pdf_text)
print("Total chunks:", len(chunks))


# ----------------------------------------
#Generate Embeddings
# model = SentenceTransformer("intfloat/e5-large-v2")
model = SentenceTransformer("all-MiniLM-L6-v2")
#model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
embeddings.shape

# ----------------------------------------
#Build FAISS Index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings))

print("FAISS index ready!")


# ----------------------------------------

def call_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        if response.status_code != 200:
            print("Error:", response.text)
            return "Model error"

        data = response.json()
        return data.get("response", "Model returned no response")

    except Exception as e:
        print("LLM Error:", str(e))
        return "LLM error"


# ----------------------------------------
# Chat history memory
chat_history = []

# ----------------------------------------
def answer_question(query):
    # Add the user query to memory
    chat_history.append({"role": "user", "content": query})

    # Vector search on PDF chunks
    q_embed = model.encode([query])
    distances, ids = index.search(q_embed, k=3)
    context = "\n\n".join([chunks[i] for i in ids[0]])

    # Build conversation history text
    history_text = ""
    for turn in chat_history:
        role = turn["role"]
        content = turn["content"]
        history_text += f"{role.upper()}: {content}\n"

    prompt = f"""
You are a helpful loan-policy assistant.
Answer ONLY using the given PDF context and chat history.
If something is not in the PDF, reply: "This information is not available in the document."

Chat History:
{history_text}

Context from PDF:
{context}

User Question:
{query}

Answer:
"""

    response = call_llm(prompt)

    # Save model's response into history
    chat_history.append({"role": "assistant", "content": response})

    return response

# ----------------------------------------

def rag_chat(user_query):
    return answer_question(user_query)