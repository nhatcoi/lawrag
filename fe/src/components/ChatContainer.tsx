import { useState, useEffect, useRef } from 'react'
import { ChatMessage } from './ChatMessage'
import { ChatInput } from './ChatInput'
import { ConversationList } from './ConversationList'
import { chatApi, ChatResponse } from '@/lib/api'
import { Select } from './ui/Select'
import { Loader2, Sparkles, MessageSquare } from 'lucide-react'
import axios from 'axios'

interface Message {
  id: string
  query: string
  answer: string
  sources: ChatResponse['sources']
  timestamp: Date
}

export function ChatContainer() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [lawTypeFilter, setLawTypeFilter] = useState<string>('Tổng quan')
  const [lawTypes, setLawTypes] = useState<string[]>(['Tổng quan'])
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    loadLawTypes()
  }, [])

  useEffect(() => {
    if (conversationId) {
      loadConversation(conversationId)
    } else {
      setMessages([])
    }
  }, [conversationId])

  const loadLawTypes = async () => {
    try {
      const types = await chatApi.getLawTypes()
      setLawTypes(types)
    } catch (error) {
      console.error('Error loading law types:', error)
    }
  }

  const loadConversation = async (convId: string) => {
    try {
      const response = await axios.get(`/api/chat/conversations/${convId}`)
      const conv = response.data
      
      if (conv.law_type_filter) {
        setLawTypeFilter(conv.law_type_filter)
      }
      
      const loadedMessages: Message[] = []
      for (let i = 0; i < conv.messages.length; i++) {
        const msg = conv.messages[i]
        if (msg.role === 'user') {
          const nextMsg = conv.messages[i + 1]
          if (nextMsg && nextMsg.role === 'assistant') {
            loadedMessages.push({
              id: `${convId}_${i}`,
              query: msg.content,
              answer: nextMsg.content,
              sources: nextMsg.sources || [],
              timestamp: new Date(msg.timestamp),
            })
            i++
          } else {
            loadedMessages.push({
              id: `${convId}_${i}`,
              query: msg.content,
              answer: '',
              sources: [],
              timestamp: new Date(msg.timestamp),
            })
          }
        }
      }
      
      setMessages(loadedMessages)
    } catch (error) {
      console.error('Error loading conversation:', error)
    }
  }

  const handleSend = async (query: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      query,
      answer: '',
      sources: [],
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, userMessage])
    setIsLoading(true)

    try {
      const response = await chatApi.chat({ 
        query,
        conversation_id: conversationId || undefined,
        law_type_filter: lawTypeFilter !== 'Tổng quan' ? lawTypeFilter : undefined
      })
      
      setConversationId(response.conversation_id)
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        query: response.query,
        answer: response.answer,
        sources: response.sources,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error:', error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        query,
        answer: 'Xin lỗi, có lỗi xảy ra. Vui lòng thử lại.',
        sources: [],
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleSelectConversation = (convId: string | null) => {
    setConversationId(convId)
    if (!convId) {
      setMessages([])
    }
  }

  return (
    <div className="flex h-full bg-background">
      <div className="w-64 border-r border-border">
        <ConversationList
          onSelectConversation={handleSelectConversation}
          selectedId={conversationId}
        />
      </div>
      
      <div className="flex-1 flex flex-col">
        <div className="p-4 border-b border-border bg-card">
          <div className="flex items-center gap-3">
            <label className="text-sm font-medium text-foreground whitespace-nowrap">Loại luật:</label>
            <Select
              value={lawTypeFilter}
              onChange={(e) => setLawTypeFilter(e.target.value)}
              className="w-48"
            >
              {lawTypes.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </Select>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-6 space-y-4 lg:px-8">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              <div className="text-center space-y-4 max-w-md">
                <div className="flex justify-center">
                  <div className="rounded-full bg-primary/10 p-4">
                    <Sparkles className="h-8 w-8 text-primary" />
                  </div>
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-foreground mb-2">
                    Chào mừng đến với RAG Chatbot
                  </h3>
                  <p className="text-sm">
                    Đặt câu hỏi về pháp luật để bắt đầu cuộc trò chuyện
                  </p>
                </div>
              </div>
            </div>
          )}
          
          {messages.map((message) => (
            <ChatMessage
              key={message.id}
              query={message.query}
              answer={message.answer}
              sources={message.sources}
              isUser={false}
            />
          ))}
          
          {isLoading && (
            <div className="flex items-center gap-3 text-muted-foreground animate-in fade-in">
              <Loader2 className="h-5 w-5 animate-spin" />
              <span className="text-sm">Đang xử lý...</span>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
        
        <ChatInput onSend={handleSend} isLoading={isLoading} />
      </div>
    </div>
  )
}
