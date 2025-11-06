# Lệnh chạy dự án

## Backend (Terminal 1)

```bash
cd be/app
source ../venv/bin/activate
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Hoặc nếu venv đã activate:
```bash
cd be/app
../venv/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Frontend (Terminal 2)

```bash
cd fe
npm install  # Chỉ cần chạy lần đầu
npm run dev
```

## Truy cập

- **Frontend**: http://localhost:3000
- **Backend API Docs**: http://localhost:8000/docs
- **Backend API**: http://localhost:8000/api

## Lệnh dừng

```bash
# Dừng backend
pkill -f "uvicorn"

# Dừng frontend
pkill -f "vite"
```

