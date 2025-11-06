import { useState, useEffect } from 'react'
import { History as HistoryIcon, MessageSquare, Clock } from 'lucide-react'
import { chatApi, HistoryItem } from '@/lib/api'
import { Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

export function HistoryPanel() {
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null)

  useEffect(() => {
    loadHistory()
  }, [])

  const loadHistory = async () => {
    try {
      setIsLoading(true)
      const data = await chatApi.getHistory()
      setHistory(data.reverse())
    } catch (error) {
      console.error('Error loading history:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const minutes = Math.floor(diff / 60000)
    const hours = Math.floor(minutes / 60)
    const days = Math.floor(hours / 24)

    if (minutes < 1) return 'Vừa xong'
    if (minutes < 60) return `${minutes} phút trước`
    if (hours < 24) return `${hours} giờ trước`
    return `${days} ngày trước`
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
            <HistoryIcon className="h-5 w-5" />
            Lịch sử chat
          </h2>
          <p className="text-xs text-muted-foreground mt-1">{history.length} cuộc trò chuyện</p>
        </div>
        <div className="space-y-1 p-2">
          {history.length === 0 ? (
            <div className="p-4 text-center text-muted-foreground">
              <p className="text-sm">Chưa có lịch sử chat</p>
            </div>
          ) : (
            history.map((item) => (
              <button
                key={item.id}
                onClick={() => setSelectedItem(item)}
                className={cn(
                  'w-full text-left p-3 rounded-lg transition-all',
                  'hover:bg-accent hover:text-accent-foreground',
                  selectedItem?.id === item.id
                    ? 'bg-primary text-primary-foreground shadow-lg'
                    : 'text-muted-foreground'
                )}
              >
                <div className="flex items-start gap-2">
                  <MessageSquare className="h-4 w-4 mt-0.5 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{item.query}</p>
                    <div className="flex items-center gap-2 mt-1 text-xs opacity-70">
                      <Clock className="h-3 w-3" />
                      <span>{formatDate(item.timestamp)}</span>
                      {item.sources_count > 0 && (
                        <span className="ml-2">• {item.sources_count} nguồn</span>
                      )}
                    </div>
                  </div>
                </div>
              </button>
            ))
          )}
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-6">
        {selectedItem ? (
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-foreground">Câu hỏi</h3>
                <span className="text-xs text-muted-foreground">
                  {formatDate(selectedItem.timestamp)}
                </span>
              </div>
              <div className="rounded-lg bg-muted p-4 border border-border">
                <p className="text-sm text-foreground">{selectedItem.query}</p>
              </div>
            </div>
            
            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-foreground">Câu trả lời</h3>
              <div className="rounded-lg bg-card p-4 border border-border">
                <p className="text-sm text-foreground whitespace-pre-wrap leading-relaxed">
                  {selectedItem.answer}
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-center">
            <div>
              <HistoryIcon className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                Chọn một cuộc trò chuyện để xem chi tiết
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

