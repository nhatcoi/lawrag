# RAG Chatbot Frontend

React + TypeScript + Tailwind CSS + shadcn/ui

## Cài đặt

```bash
npm install
```

## Chạy

```bash
npm run dev
```

Mở http://localhost:3000

## Build

```bash
npm run build
```

## Cấu trúc

```
fe/
├── src/
│   ├── components/
│   │   ├── ui/          # shadcn/ui components
│   │   ├── ChatMessage.tsx
│   │   ├── ChatInput.tsx
│   │   └── ChatContainer.tsx
│   ├── lib/
│   │   ├── api.ts      # API client
│   │   └── utils.ts    # Utilities
│   ├── App.tsx
│   └── main.tsx
```

## API Endpoints

- `POST /api/chat` - Chat với RAG
- `POST /api/upload` - Upload và index file
- `GET /api/history` - Lịch sử chat
- `GET /api/sources` - Danh sách nguồn

