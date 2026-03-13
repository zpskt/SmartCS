# ai_service/app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from core.rag import RagService
from core.knowledge_base import KnowledgeBaseService
import pdfplumber
from pydantic import BaseModel

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Vue 项目地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.get("/index")
def index():
    return {"message": "Hello World"}
@app.post("/api/chat")
async def chat(request: ChatRequest):
    rag_service = RagService()
    response = rag_service.chain.invoke(
        {"input": request.message},
        {"configurable": {"session_id": request.session_id}}
    )
    return {"reply": response}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename

    if filename.endswith('.pdf'):
        import io
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages])
    elif filename.endswith(('.doc', '.docx')):
        from docx import Document
        import io
        doc = Document(io.BytesIO(content))
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = content.decode('utf-8')

    kb_service = KnowledgeBaseService()
    result = kb_service.upload_by_str(text, filename)
    return {"message": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
