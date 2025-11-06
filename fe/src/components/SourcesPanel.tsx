import { useState, useEffect } from 'react'
import { FileText, Download, File, Loader2, Trash2, FolderOpen, ChevronRight, ChevronDown } from 'lucide-react'
import { chatApi, DocumentInfo, ChunkInfo } from '@/lib/api'
import { Button } from './ui/Button'
import { cn } from '@/lib/utils'
import axios from 'axios'

interface SourcesPanelProps {
  refreshTrigger?: number
}

export function SourcesPanel({ refreshTrigger }: SourcesPanelProps) {
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedDoc, setSelectedDoc] = useState<DocumentInfo | null>(null)
  const [expandedDocs, setExpandedDocs] = useState<Set<string>>(new Set())
  const [isDeleting, setIsDeleting] = useState<string | null>(null)

  useEffect(() => {
    loadDocuments()
  }, [refreshTrigger])

  const loadDocuments = async () => {
    try {
      setIsLoading(true)
      const data = await chatApi.getSources()
      setDocuments(data)
      if (data.length > 0 && !selectedDoc) {
        setSelectedDoc(data[0])
      }
    } catch (error) {
      console.error('Error loading documents:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleDelete = async (docId: string, filename: string) => {
    if (!confirm(`Bạn có chắc muốn xóa "${filename}"? Tất cả chunks sẽ bị xóa.`)) {
      return
    }

    try {
      setIsDeleting(docId)
      await axios.delete(`/api/sources/${docId}`)
      
      if (selectedDoc?.id === docId) {
        setSelectedDoc(null)
      }
      
      await loadDocuments()
    } catch (error: any) {
      console.error('Error deleting document:', error)
      alert(error.response?.data?.detail || 'Có lỗi xảy ra khi xóa')
    } finally {
      setIsDeleting(null)
    }
  }

  const toggleExpand = (docId: string) => {
    setExpandedDocs(prev => {
      const newSet = new Set(prev)
      if (newSet.has(docId)) {
        newSet.delete(docId)
      } else {
        newSet.add(docId)
      }
      return newSet
    })
  }

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB'
  }

  const formatDate = (dateStr: string) => {
    try {
      const date = new Date(dateStr)
      return date.toLocaleString('vi-VN')
    } catch {
      return dateStr
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="flex h-full">
      <div className="w-1/3 border-r border-border overflow-y-auto">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
            <FolderOpen className="h-5 w-5" />
            Tài liệu đã upload
          </h2>
          <p className="text-xs text-muted-foreground mt-1">{documents.length} tài liệu</p>
        </div>
        <div className="space-y-1 p-2">
          {documents.length === 0 ? (
            <div className="p-4 text-center text-muted-foreground">
              <p className="text-sm">Chưa có tài liệu</p>
            </div>
          ) : (
            documents.map((doc) => (
              <div key={doc.id} className="space-y-1">
                <div
                  className={cn(
                    'group relative rounded-lg transition-all',
                    selectedDoc?.id === doc.id
                      ? 'bg-primary text-primary-foreground shadow-lg'
                      : 'hover:bg-accent hover:text-accent-foreground'
                  )}
                >
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => toggleExpand(doc.id)}
                      className="p-1 hover:bg-black/10 rounded"
                    >
                      {expandedDocs.has(doc.id) ? (
                        <ChevronDown className="h-4 w-4" />
                      ) : (
                        <ChevronRight className="h-4 w-4" />
                      )}
                    </button>
                    <button
                      onClick={() => setSelectedDoc(doc)}
                      className="flex-1 text-left p-2"
                    >
                      <div className="flex items-start gap-2">
                        <FileText className="h-4 w-4 mt-0.5 shrink-0" />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium truncate">{doc.original_name}</p>
                          <p className="text-xs opacity-70 mt-1">
                            {doc.law_type || 'Khác'} • {doc.chunks_count} chunks • {formatFileSize(doc.size)}
                          </p>
                        </div>
                      </div>
                    </button>
                    <button
                      onClick={() => handleDelete(doc.id, doc.original_name)}
                      disabled={isDeleting === doc.id}
                      className={cn(
                        'p-1.5 rounded opacity-0 group-hover:opacity-100 transition-opacity',
                        'hover:bg-red-500/20 text-red-500',
                        selectedDoc?.id === doc.id && 'opacity-100'
                      )}
                    >
                      {isDeleting === doc.id ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <Trash2 className="h-3 w-3" />
                      )}
                    </button>
                  </div>
                </div>
                
                {expandedDocs.has(doc.id) && doc.chunks.length > 0 && (
                  <div className="ml-6 space-y-1">
                    {doc.chunks.map((chunk, idx) => (
                      <div
                        key={idx}
                        className="p-2 rounded bg-muted/50 text-xs"
                      >
                        <div className="flex items-center gap-2">
                          <File className="h-3 w-3" />
                          <span className="truncate">{chunk.filename}</span>
                          <span className="text-muted-foreground ml-auto">
                            {formatFileSize(chunk.size)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-6">
        {selectedDoc ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  {selectedDoc.original_name}
                </h3>
                <p className="text-xs text-muted-foreground mt-1">
                  {selectedDoc.law_type || 'Khác'} • {selectedDoc.type.toUpperCase()} • {formatFileSize(selectedDoc.size)} • {formatDate(selectedDoc.uploaded_at)}
                </p>
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleDelete(selectedDoc.id, selectedDoc.original_name)}
                  disabled={isDeleting === selectedDoc.id}
                >
                  {isDeleting === selectedDoc.id ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4 mr-2" />
                  )}
                  Xóa
                </Button>
              </div>
            </div>
            
            <div className="rounded-lg bg-card p-4 border border-border">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-foreground">Chunks ({selectedDoc.chunks_count})</p>
                <span className={cn(
                  "text-xs px-2 py-1 rounded",
                  selectedDoc.status === "indexed" 
                    ? "bg-green-500/20 text-green-500" 
                    : "bg-yellow-500/20 text-yellow-500"
                )}>
                  {selectedDoc.status}
                </span>
              </div>
              <div className="space-y-2 mt-4 max-h-96 overflow-y-auto">
                {selectedDoc.chunks.map((chunk, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between p-3 rounded bg-muted/50 border border-border"
                  >
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <File className="h-4 w-4 text-primary shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-foreground truncate">
                          {chunk.filename}
                        </p>
                        <p className="text-xs text-muted-foreground truncate">
                          {chunk.path}
                        </p>
                      </div>
                    </div>
                    <div className="text-xs text-muted-foreground ml-4 shrink-0">
                      {formatFileSize(chunk.size)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-center">
            <div>
              <FileText className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                Chọn một tài liệu để xem chi tiết
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
