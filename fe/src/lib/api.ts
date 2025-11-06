import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface ChatRequest {
  query: string
  conversation_id?: string
  provider?: string
  law_type_filter?: string
}

export interface Source {
  rank: number
  score: number
  id: string
  path: string
  text: string
}

export interface ChatResponse {
  answer: string
  sources: Source[]
  query: string
  conversation_id: string
}

export interface HistoryItem {
  id: string
  query: string
  answer: string
  timestamp: string
  sources_count: number
}

export interface ChunkInfo {
  filename: string
  path: string
  size: number
  created_at: string
}

export interface DocumentInfo {
  id: string
  filename: string
  original_name: string
  type: string
  size: number
  uploaded_at: string
  status: string
  chunks_count: number
  chunks: ChunkInfo[]
  law_type?: string
}

export interface SourceInfo {
  id: string
  filename: string
  path: string
  size: number
  type?: string
  uploaded_at?: string
  status?: string
}

export const chatApi = {
  chat: async (data: ChatRequest): Promise<ChatResponse> => {
    const response = await api.post('/chat', data)
    return response.data
  },
  getHistory: async (): Promise<HistoryItem[]> => {
    const response = await api.get('/history')
    return response.data
  },
  getSources: async (): Promise<DocumentInfo[]> => {
    const response = await api.get('/sources')
    return response.data
  },
  getSource: async (sourceId: string): Promise<DocumentInfo> => {
    const response = await api.get(`/sources/${sourceId}`)
    return response.data
  },
  deleteSource: async (sourceId: string): Promise<any> => {
    const response = await api.delete(`/sources/${sourceId}`)
    return response.data
  },
  upload: async (file: File, lawType?: string): Promise<any> => {
    const formData = new FormData()
    formData.append('file', file)
    if (lawType) {
      formData.append('law_type', lawType)
    }
    const response = await api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },
  uploadMultiple: async (files: File[], lawType?: string): Promise<any> => {
    const formData = new FormData()
    files.forEach(file => {
      formData.append('files', file)
    })
    if (lawType) {
      formData.append('law_type', lawType)
    }
    const response = await api.post('/upload/multiple', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },
  getLawTypes: async (): Promise<string[]> => {
    const response = await api.get('/chat/law-types')
    return response.data.law_types || []
  },
}

