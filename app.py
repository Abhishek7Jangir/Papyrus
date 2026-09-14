import torch
import streamlit as st
import re
import faiss

from io import BytesIO
from pypdf import PdfReader
from transformers import pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder


# 1. Load model 
device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_llm():
    try:
        return pipeline(
            "text-generation",
            model="Qwen/WebWorld-8B",
            dtype=torch.bfloat16,
            device_map="auto"
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        st.stop()


@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=device
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")


@st.cache_resource
def load_reranker():
    try:
        return CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            device=device
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        st.stop()




pipe = load_llm()
embedding_model = load_embedding_model()
reranker = load_reranker()

# 2. Initialize session state
# a. Messages
if "messages" not in st.session_state:
  st.session_state["messages"] = [
    {
        "role": "system",
        "content": (
            "You are a very ironic, but helpful assistant "
            "who enjoys giving comedic responses."
        )

    },
    {
        "role": "assistant",
        "content": "Oh! You here again!"
    }
]

# b. chunks
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

# c. faiss
if "faiss_index" not in st.session_state:
    st.session_state["faiss_index"] = faiss.IndexFlatL2(384)

# d. uploaded files
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = set()

# 3. PDF Functions

def extract_pdf_text(uploaded_file):
    try:
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    except Exception as e:
        st.sidebar.error(f"Could not read {uploaded_file.name}: {e}")
        return []

    pages = []

    for page_number, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception:
            text = None
        if text:
            pages.append({
                "page": page_number + 1,
                "text": text
            })
    return pages

def chunk_text(text, chunk_size=250, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[max(i-overlap, 0): i+chunk_size])
        chunks.append(chunk.strip())

    return chunks

st.sidebar.header("Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type = ["pdf"],
    accept_multiple_files = True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        if filename in st.session_state["uploaded_files"]:
            continue
        st.sidebar.write(f"Processing: {filename}")
        pages = extract_pdf_text(uploaded_file)
        
        if not pages:
            st.toast(f"{filename}: no readable text found", icon="⚠️")
            continue
        
        new_chunks = []
        new_texts = []

        #chunks
        for page_data in pages:
            page_number = page_data["page"]
            page_text = page_data["text"]
            chunks = chunk_text(page_text)

            for chunk in chunks:
                new_chunks.append({
                    "text": chunk,
                    "source": filename,
                    "page": page_number
                })
                new_texts.append(chunk)

        #embeddings
        if new_texts:
            embeddings = embedding_model.encode(
                new_texts,
                convert_to_numpy=True,
                batch_size=64,
                show_progress_bar=False
            )
            embeddings = embeddings.astype("float32")

            st.session_state["faiss_index"].add(
                embeddings
            )

            #store chunks
            st.session_state["chunks"].extend(
                new_chunks
            )

        #pdf processed
        st.session_state["uploaded_files"].add(
            filename
        )

        st.sidebar.success(
            f"{filename} added!"
        )

#display document info

st.sidebar.write("---")
st.sidebar.write(
    f"Documents: "
    f"{len(st.session_state['uploaded_files'])}",
)
st.sidebar.write(
    f"Vectors: "
    f"{st.session_state['faiss_index'].ntotal}"
)
    

# Display chat history
for msg in st.session_state["messages"][1:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle new user input
if prompt := st.chat_input("Type your message..."):
    prompt = prompt.strip()
    if not prompt:
        st.stop()
    st.session_state["messages"].append({
        "role": "user",
        "content": prompt
    })
    
    with st.chat_message("user"):
        st.write(prompt)


#retriving
    retrived_chunks = []

    if st.session_state["faiss_index"].ntotal > 0:
        query_embedding = embedding_model.encode(
            [prompt],
            convert_to_numpy=True
        )
        query_embedding = query_embedding.astype(
            "float32"
        )

        k = min(
            20,
            st.session_state["faiss_index"].ntotal
        )

        distances, indices = (
            st.session_state["faiss_index"]
            .search(query_embedding, k)
        )

        candidate_chunks = []

        for index in indices[0]:
            chunk = st.session_state["chunks"][index]
            candidate_chunks.append(chunk)

        pairs = [
            (prompt, chunk["text"])
            for chunk in candidate_chunks
        ]

        scores = reranker.predict(pairs)
        reranked_chunks = sorted(
            zip(scores, candidate_chunks),
            key=lambda x: x[0],
            reverse=True
        )
        retrived_chunks = [
            chunk
            for score, chunk in reranked_chunks[:5]
        ]


    #build context
    if retrived_chunks:
        context_parts = []
        for chunk in retrived_chunks:
            context_parts.append(
                f"Source: {chunk['source']}, "
                f"Page: {chunk['page']}\n"
                f"{chunk['text']}"
            )
        context = "\n\n---\n\n".join(
            context_parts
        )
    else:
        context = "No documents have been uploaded."

    #Augmented prompt
    rag_messages = [
        {
            "role": "system",
            "content": (
                "You are a very ironic, sarcastic assistant "
                "who enjoys giving comedic reply to the user.\n\n"

                "You are also a RAG assistant. Use the provided "
                "document context to answer the user's question.\n\n"

                "If the answer is present in the documents, "
                "use the documents as the primary source.\n\n"

                "If the answer cannot be found in the documents, "
                "say that the information was not found in the "
                "uploaded documents instead of inventing facts.\n\n"

                "DOCUMENT CONTEXT:\n"
                f"{context}"
            )
        }
    ]

    
    #recent conv.
    rag_messages.extend(
        st.session_state["messages"][-6:]
    )

    

    
  # generate the model's response
    model_prompt = pipe.tokenizer.apply_chat_template(
        rag_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    try:
        outputs = pipe(
            model_prompt,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.5,
            top_p=0.9
        )
        full_generated_text = outputs[0]["generated_text"]
    except Exception as e:
        st.error(f"Generation failed: {e}")
        st.stop()

    # Extract only the assistant's new response content.
    # The full_generated_text contains the prompt (which includes the system and user messages)
    # and then the model's generated response. We remove the prompt part to get only the new content.
    response = full_generated_text[len(model_prompt):].strip()
    response = re.sub(
        r"<reason>.*?</reason>|<thinking>.*?</thinking>",
        "",
        response,
        flags=re.DOTALL
    ).strip()

    # save and show the assistant's message
    st.session_state["messages"].append({
        "role": "assistant",
        "content": response
    })
    
    with st.chat_message("assistant"):
        st.write(response)

    if len(st.session_state["messages"]) > 11:
        st.session_state["messages"].pop(1)
        st.session_state["messages"].pop(1)