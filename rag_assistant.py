import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
#from langchain_openai import ChatOpenAI
#from langchain_community.llms import FakeLLM
from transformers import pipeline
from langchain_core.language_models import LLM

class DummyLLM(LLM):
    def _call(self, prompt, stop=None):
        return "This is a dummy response for testing purposes."
    
    @property
    def _llm_type(self):
        return "dummy"




# === Setup ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# === Load PDF ===
pdf_path = "sample_paper.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# === Split text ===
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
docs = splitter.split_documents(pages)

# === Embeddings + Chroma vector DB ===
embedding_model = FakeEmbeddings(size=1536)
vectorstore = Chroma.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# === GPT-4 via LangChain ===
#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = DummyLLM()
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# === Hugging Face RoBERTa ===
hf_qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

# === Ask a question ===
query = "What is the main contribution of this paper?"

# Hugging Face extractive answer
hf_docs = retriever.get_relevant_documents(query)
hf_context = " ".join([doc.page_content for doc in hf_docs])
hf_answer = hf_qa(question=query, context=hf_context)

# GPT-4 generative answer
rag_result = rag_chain(query)

# === Display results ===
print("\n=== Hugging Face Extractive Answer ===")
print(f"Answer: {hf_answer['answer']}")

print("\n=== GPT-4 Generative Answer (via LangChain) ===")
print(f"Answer: {rag_result['result']}")

print("\nSources:")
for i, doc in enumerate(rag_result["source_documents"], start=1):
    print(f"- Source {i}: {doc.page_content[:150]}...")

