from fastapi import FastAPI, Request
from pydantic import BaseModel
from chatbot import load_existing_vector_store, ConversationalRetrievalChain

app = FastAPI()

class Query(BaseModel):
    question: str
    history: list = []

llm, vectordb, _ = load_existing_vector_store()
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

@app.get("/")
def root():
    return {"message": "🚀 Chatbot API is running"}

@app.post("/ask")
def ask(query: Query):
    result = qa_chain({"question": query.question, "chat_history": query.history})
    return {
        "answer": result["answer"],
        "sources": [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]
    }
