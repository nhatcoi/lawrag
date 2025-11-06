import { MessageCircle, FileText, User } from 'lucide-react'
import { Source } from '@/lib/api'
import { cn } from '@/lib/utils'
import { Collapsible } from './ui/Collapsible'

interface ChatMessageProps {
  query: string
  answer: string
  sources: Source[]
  isUser?: boolean
}

export function ChatMessage({ query, answer, sources, isUser }: ChatMessageProps) {
  return (
    <div className="space-y-4 mb-6">
      {query && (
        <div className={cn('flex gap-3 animate-in fade-in slide-in-from-bottom-2 duration-300', isUser && 'ml-auto max-w-[85%]')}>
          <div className={cn(
            'flex h-8 w-8 shrink-0 items-center justify-center rounded-full',
            isUser ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
          )}>
            <User className="h-4 w-4" />
          </div>
          <div className="flex-1">
            <div className="rounded-2xl bg-primary px-4 py-3 text-primary-foreground shadow-lg">
              <p className="text-sm leading-relaxed">{query}</p>
            </div>
          </div>
        </div>
      )}
      
      {answer && (
        <div className="flex gap-3 animate-in fade-in slide-in-from-bottom-2 duration-300">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
            <MessageCircle className="h-4 w-4" />
          </div>
          <div className="flex-1 space-y-3">
            <div className="rounded-2xl border border-border bg-card px-4 py-3 shadow-lg">
              <p className="text-sm leading-relaxed text-foreground whitespace-pre-wrap">{answer}</p>
            </div>
            
            {sources.length > 0 && (
              <Collapsible title={`Nguồn tham khảo (${sources.length})`}>
                <div className="space-y-2">
                  {sources.map((source, idx) => (
                    <div
                      key={idx}
                      className="rounded-lg border border-border bg-background p-3 text-xs transition-all hover:border-primary/50"
                    >
                      <div className="font-medium text-foreground mb-1 flex items-center gap-2">
                        <FileText className="h-3 w-3" />
                        {source.id || `Nguồn ${idx + 1}`}
                      </div>
                      <div className="text-muted-foreground line-clamp-2 leading-relaxed mt-1">
                        {source.text}
                      </div>
                    </div>
                  ))}
                </div>
              </Collapsible>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
