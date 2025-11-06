import { useState } from 'react'
import { MessageSquare, Upload, History, FileText, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from './ui/Button'

interface SidebarProps {
  activeTab: 'chat' | 'upload' | 'history' | 'sources'
  onTabChange: (tab: 'chat' | 'upload' | 'history' | 'sources') => void
  isOpen: boolean
  onClose: () => void
}

export function Sidebar({ activeTab, onTabChange, isOpen, onClose }: SidebarProps) {
  const tabs = [
    { id: 'chat' as const, label: 'Chat', icon: MessageSquare },
    { id: 'upload' as const, label: 'Upload', icon: Upload },
    { id: 'history' as const, label: 'History', icon: History },
    { id: 'sources' as const, label: 'Sources', icon: FileText },
  ]

  return (
    <>
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}
      <aside
        className={cn(
          'fixed left-0 top-0 h-full w-64 bg-card border-r border-border z-50 transition-transform duration-300',
          'lg:translate-x-0',
          isOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        <div className="flex flex-col h-full">
          <div className="flex items-center justify-between p-4 border-b border-border">
            <h2 className="text-lg font-semibold text-foreground">Menu</h2>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="lg:hidden"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          
          <nav className="flex-1 p-4 space-y-2">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => {
                    onTabChange(tab.id)
                    onClose()
                  }}
                  className={cn(
                    'w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all',
                    'hover:bg-accent hover:text-accent-foreground',
                    activeTab === tab.id
                      ? 'bg-primary text-primary-foreground shadow-lg'
                      : 'text-muted-foreground'
                  )}
                >
                  <Icon className="h-5 w-5" />
                  <span className="font-medium">{tab.label}</span>
                </button>
              )
            })}
          </nav>
        </div>
      </aside>
    </>
  )
}

