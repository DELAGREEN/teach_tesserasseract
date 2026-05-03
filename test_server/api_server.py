# test_api_server.py

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import uvicorn
from datetime import datetime
import glob

app = FastAPI(title="Test PDF API Server", description="Тестовый API сервер для раздачи PDF файлов")

# Конфигурация
PDF_STORAGE_DIR = os.getenv("PDF_STORAGE_DIR", "./pdf_storage")  # Папка с PDF файлами
AUTH_TOKEN = os.getenv("API_TOKEN", "test_token_12345")
USERS = {
    "admin": "admin123",
    "test_user": "test_pass"
}

# Создаём папку для PDF если её нет
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

# Модели данных
class AuthRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    token: str
    user_id: Optional[int] = None

# Вспомогательные функции
def get_pdf_files():
    """Получает список всех PDF файлов в директории"""
    pdf_files = []
    pdf_paths = glob.glob(os.path.join(PDF_STORAGE_DIR, "*.pdf"))
    
    for i, file_path in enumerate(pdf_paths):
        filename = os.path.basename(file_path)
        stat = os.stat(file_path)
        pdf_files.append({
            "id": i + 1,  # Простая нумерация
            "blobId": f"blob_{i + 1}",
            "name": filename,
            "size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat()
        })
    return pdf_files

def find_pdf_by_id(object_id: int):
    """Находит PDF файл по ID"""
    pdf_files = get_pdf_files()
    for pdf in pdf_files:
        if pdf["id"] == object_id:
            return pdf
    return None

def find_pdf_by_blob_id(blob_id: str):
    """Находит PDF файл по blobId"""
    pdf_files = get_pdf_files()
    for pdf in pdf_files:
        if pdf["blobId"] == blob_id:
            return pdf
    return None

# Middleware для проверки авторизации
async def verify_token(token: str):
    """Проверяет валидность токена"""
    if not token:
        return False
    # Убираем "Bearer " если есть
    if token.startswith("Bearer "):
        token = token[7:]
    return token == AUTH_TOKEN

# Эндпоинты API

@app.get("/")
async def root():
    """Корневой эндпоинт с информацией о сервере"""
    return {
        "service": "Test PDF API Server",
        "version": "1.0.0",
        "pdf_storage_dir": PDF_STORAGE_DIR,
        "pdf_count": len(get_pdf_files()),
        "status": "ready"
    }

@app.post("/core/api/Auth/authenticate")
async def authenticate(auth: AuthRequest):
    """Эндпоинт аутентификации"""
    if auth.username in USERS and USERS[auth.username] == auth.password:
        return AuthResponse(token=AUTH_TOKEN, user_id=1)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

@app.get("/core/api/objects/select")
async def get_objects(authorization: Optional[str] = None):
    """Получение списка всех PDF объектов"""
    # Проверяем авторизацию
    if not await verify_token(authorization):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token"
        )
    
    objects = get_pdf_files()
    return JSONResponse(content=objects)

@app.get("/core/api/download/{blobId}/{object_id}")
async def download_pdf(blobId: str, object_id: int, authorization: Optional[str] = None):
    """Скачивание PDF файла по blobId и id"""
    # Проверяем авторизацию
    if not await verify_token(authorization):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token"
        )
    
    # Ищем файл по blobId или id
    pdf_info = find_pdf_by_blob_id(blobId) or find_pdf_by_id(object_id)
    
    if not pdf_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF not found: blobId={blobId}, id={object_id}"
        )
    
    file_path = os.path.join(PDF_STORAGE_DIR, pdf_info["name"])
    
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {pdf_info['name']}"
        )
    
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=pdf_info["name"]
    )

@app.get("/get-pdf")
async def get_pdf_legacy():
    """Legacy эндпоинт для тестов - возвращает первый PDF из папки"""
    pdf_files = get_pdf_files()
    
    if not pdf_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No PDF files found in storage directory"
        )
    
    first_pdf = pdf_files[0]
    file_path = os.path.join(PDF_STORAGE_DIR, first_pdf["name"])
    
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=first_pdf["name"]
    )

@app.post("/upload-pdf")
async def upload_pdf(file: bytes = None, authorization: Optional[str] = None):
    """Эндпоинт для загрузки новых PDF файлов"""
    if not await verify_token(authorization):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token"
        )
    
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    filename = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    file_path = os.path.join(PDF_STORAGE_DIR, filename)
    
    with open(file_path, "wb") as f:
        f.write(file)
    
    pdf_files = get_pdf_files()
    new_id = len(pdf_files)
    
    return {
        "message": "PDF uploaded successfully",
        "filename": filename,
        "blobId": f"blob_{new_id}",
        "id": new_id
    }

@app.get("/admin/list-files")
async def list_files():
    """Административный эндпоинт для просмотра файлов"""
    files = []
    for filename in os.listdir(PDF_STORAGE_DIR):
        file_path = os.path.join(PDF_STORAGE_DIR, filename)
        if os.path.isfile(file_path) and filename.lower().endswith('.pdf'):
            files.append({
                "name": filename,
                "size": os.path.getsize(file_path),
                "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            })
    return {"files": files, "count": len(files), "directory": PDF_STORAGE_DIR}

@app.delete("/admin/clear-all")
async def clear_all_files():
    """Очистка всех PDF файлов"""
    deleted = []
    for filename in os.listdir(PDF_STORAGE_DIR):
        file_path = os.path.join(PDF_STORAGE_DIR, filename)
        if os.path.isfile(file_path) and filename.lower().endswith('.pdf'):
            os.remove(file_path)
            deleted.append(filename)
    return {"message": f"Deleted {len(deleted)} files", "deleted": deleted}

# Запуск сервера
if __name__ == "__main__":
    print("=" * 60)
    print("Запуск тестового API сервера")
    print("=" * 60)
    print(f"PDF storage directory: {PDF_STORAGE_DIR}")
    print(f"Place your PDF files in: {os.path.abspath(PDF_STORAGE_DIR)}")
    print(f"Auth token: {AUTH_TOKEN}")
    print(f"Test credentials: admin/admin123 or test_user/test_pass")
    print("=" * 60)
    print("\nДоступные эндпоинты:")
    print("  GET  /                          - информация о сервере")
    print("  POST /core/api/Auth/authenticate - аутентификация")
    print("  GET  /core/api/objects/select   - список PDF файлов")
    print("  GET  /core/api/download/{blobId}/{id} - скачать PDF")
    print("  GET  /get-pdf                   - скачать первый PDF")
    print("  POST /upload-pdf                - загрузить новый PDF")
    print("  GET  /admin/list-files          - список файлов (админ)")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )