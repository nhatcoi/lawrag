// Lấy các element từ DOM
const chat = document.getElementById('chat');
const form = document.getElementById('form');
const input = document.getElementById('query');
const apiBaseInput = document.getElementById('apiBase');
const saveApiBtn = document.getElementById('saveApi');
const statusEl = document.getElementById('status');
const sendBtn = document.getElementById('sendBtn');

/**
 * Lấy tham số từ URL query string
 */
function getQueryParam(name) {
  const url = new URL(window.location.href);
  return url.searchParams.get(name);
}

/**
 * Xác định API base URL theo thứ tự ưu tiên:
 * 1. URL param ?api=
 * 2. localStorage
 * 3. Same origin (nếu chạy từ FastAPI)
 * 4. Fallback localhost:8000
 */
function resolveApiBase() {
  // 1) URL param: ?api=http://host:port
  const fromParam = getQueryParam('api');
  if (fromParam) {
    try { localStorage.setItem('API_BASE', fromParam); } catch {}
    return fromParam;
  }
  // 2) localStorage persisted value
  try {
    const fromStore = localStorage.getItem('API_BASE');
    if (fromStore) return fromStore;
  } catch {}
  // 3) Nếu chạy từ FastAPI dưới /app, dùng same-origin
  if (window.location.pathname.startsWith('/app/')) return window.location.origin;
  // 4) Fallback mặc định
  return 'http://localhost:8000';
}

// Khởi tạo API base URL và UI
const API_BASE = window.API_BASE || resolveApiBase();
apiBaseInput.value = API_BASE;

// Xử lý lưu API base URL
saveApiBtn.addEventListener('click', () => {
  const v = apiBaseInput.value.trim();
  if (v) {
    try { localStorage.setItem('API_BASE', v); } catch {}
    window.location.reload();
  }
});

/**
 * Thêm tin nhắn vào chat
 */
function appendMsg(text, who = 'bot') {
  const div = document.createElement('div');
  div.className = `msg ${who}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

/**
 * Hiển thị thông tin nguồn tài liệu
 */
function appendSources(sources) {
  if (!sources || !sources.length) return;
  const lines = sources.map(s => `#${s.rank} (${s.score}) - ${s.id}`);
  const div = document.createElement('div');
  div.className = 'sources';
  div.textContent = 'Nguồn: ' + lines.join(' | ');
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

// Xử lý submit form chat
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  
  // Hiển thị câu hỏi của user
  appendMsg(q, 'me');
  input.value = '';
  sendBtn.disabled = true;
  statusEl.hidden = false;

  try {
    // Gọi API RAG
    const res = await fetch(`${API_BASE}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: q,
        // Sử dụng defaults từ server cho index_dir, provider, etc.
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Request failed');
    
    // Hiển thị câu trả lời và nguồn
    appendMsg(data.answer || '(không có trả lời)');
    appendSources(data.sources);
  } catch (err) {
    appendMsg('Lỗi: ' + (err?.message || err), 'bot');
  } finally {
    // Khôi phục UI
    sendBtn.disabled = false;
    statusEl.hidden = true;
  }
});


