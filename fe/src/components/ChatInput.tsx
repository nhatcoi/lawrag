import { useState, FormEvent } from 'react'
import { Send, Loader2 } from 'lucide-react'
import { Input } from './ui/Input'
import { Button } from './ui/Button'

interface ChatInputProps {
  onSend: (query: string) => void
  isLoading?: boolean
}

export function ChatInput({ onSend, isLoading }: ChatInputProps) {
  const [query, setQuery] = useState('')

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (query.trim() && !isLoading) {
      onSend(query.trim())
      setQuery('')
    }
  }

  return (
    <div className="border-t border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <form onSubmit={handleSubmit} className="flex gap-3 p-4 max-w-4xl mx-auto">
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Đặt câu hỏi về pháp luật..."
          disabled={isLoading}
          className="flex-1 bg-muted/50 border-border focus-visible:ring-primary"
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              handleSubmit(e)
            }
          }}
        />
        <Button 
          type="submit" 
          disabled={isLoading || !query.trim()}
          size="lg"
          className="shrink-0"
        >
          {isLoading ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            <Send className="h-5 w-5" />
          )}
        </Button>
      </form>
    </div>
  )
}
