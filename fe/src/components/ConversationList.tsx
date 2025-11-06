import { useState, useEffect } from 'react'
import { MessageSquare, Plus, Trash2, Loader2, Clock } from 'lucide-react'
import { chatApi } from '@/lib/api'
import { Button } from './ui/Button'
import { cn } from '@/lib/utils'
import axios from 'axios'

interface Conversation {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

interface ConversationListProps {
  onSelectConversation: (convId: string | null) => void
  selectedId: string | null
}

export function ConversationList({ onSelectConversation, selectedId }: ConversationListProps) {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isDeleting, setIsDeleting] = useState<string | null>(null)

  useEffect(() => {
    loadConversations()
  }, [])

  const loadConversations = async () => {
    try {
      setIsLoading(true)
      const response = await axios.get('/api/chat/conversations')
      setConversations(response.data)
    } catch (error) {
      console.error('Error loading conversations:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleDelete = async (convId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm('Bạn có chắc muốn xóa cuộc trò chuyện này?')) {
      return
    }

    try {
      setIsDeleting(convId)
      await axios.delete(`/api/chat/conversations/${convId}`)
      if (selectedId === convId) {
        onSelectConversation(null)
      }
      await loadConversations()
    } catch (error: any) {
      console.error('Error deleting conversation:', error)
      alert(error.response?.data?.detail || 'Có lỗi xảy ra khi xóa')
    } finally {
      setIsDeleting(null)
    }
  }

  const formatDate = (dateStr: string) => {
    try {
      const date = new Date(dateStr)
      const now = new Date()
      const diff = now.getTime() - date.getTime()
      const minutes = Math.floor(diff / 60000)
      const hours = Math.floor(minutes / 60)
      const days = Math.floor(hours / 24)

      if (minutes < 1) return 'Vừa xong'
      if (minutes < 60) return `${minutes} phút trước`
      if (hours < 24) return `${hours} giờ trước`
      return `${days} ngày trước`
    } catch {
      return dateStr
    }
  }

  const handleNewConversation = () => {
    onSelectConversation(null)
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full border-r border-border">
      <div className="p-4 border-b border-border">
        <Button
          onClick={handleNewConversation}
          className="w-full"
          size="sm"
        >
          <Plus className="h-4 w-4 mr-2" />
          Cuộc trò chuyện mới
        </Button>
      </div>
      
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {conversations.length === 0 ? (
          <div className="p-4 text-center text-muted-foreground">
            <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">Chưa có cuộc trò chuyện</p>
          </div>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              onClick={() => onSelectConversation(conv.id)}
              className={cn(
                'group relative p-3 rounded-lg cursor-pointer transition-all',
                'hover:bg-accent hover:text-accent-foreground',
                selectedId === conv.id && 'bg-primary text-primary-foreground shadow-lg'
              )}
            >
              <div className="flex items-start gap-2">
                <MessageSquare className="h-4 w-4 mt-0.5 shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{conv.title}</p>
                  <div className="flex items-center gap-2 mt-1 text-xs opacity-70">
                    <Clock className="h-3 w-3" />
                    <span>{formatDate(conv.updated_at)}</span>
                    <span>•</span>
                    <span>{conv.message_count} tin nhắn</span>
                  </div>
                </div>
              </div>
              <button
                onClick={(e) => handleDelete(conv.id, e)}
                disabled={isDeleting === conv.id}
                className={cn(
                  'absolute right-2 top-2 p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity',
                  'hover:bg-red-500/20 text-red-500',
                  selectedId === conv.id && 'opacity-100'
                )}
              >
                {isDeleting === conv.id ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <Trash2 className="h-3 w-3" />
                )}
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

