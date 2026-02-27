from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import io

app = FastAPI()

@app.get("/get-pdf")
async def get_pdf():
    # Читаем PDF файл в байты
    with open("/home/user/Desktop/Примеры чертежей/Компас/А4_4/РНАТ.714352.034.cdw.PDF", "rb") as pdf_file:
        pdf_bytes = pdf_file.read()
    
    # Возвращаем PDF как байты
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=document.pdf"}
    )

@app.get("/get-pdf-stream")
async def get_pdf_stream():
    # Альтернативный вариант с StreamingResponse
    file_like = open("/home/user/Desktop/Примеры чертежей/Компас/А4_4/РНАТ.714352.034.cdw.PDF", "rb")
    return StreamingResponse(
        file_like,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=document.pdf"}
    )