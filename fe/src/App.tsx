import { useState } from 'react'
import { Menu } from 'lucide-react'
import { Sidebar } from './components/Sidebar'
import { ChatContainer } from './components/ChatContainer'
import { UploadFile } from './components/UploadFile'
import { HistoryPanel } from './components/HistoryPanel'
import { SourcesPanel } from './components/SourcesPanel'
import { Card, CardContent } from './components/ui/Card'
import { Button } from './components/ui/Button'

function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'upload' | 'history' | 'sources'>('chat')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [refreshTrigger, setRefreshTrigger] = useState(0)

  const handleUploadSuccess = () => {
    setRefreshTrigger(prev => prev + 1)
    setActiveTab('sources')
  }

  const renderContent = () => {
    switch (activeTab) {
      case 'chat':
        return <ChatContainer />
      case 'upload':
        return <UploadFile onUploadSuccess={handleUploadSuccess} />
      case 'history':
        return <HistoryPanel />
      case 'sources':
        return <SourcesPanel refreshTrigger={refreshTrigger} />
      default:
        return <ChatContainer />
    }
  }

  return (
    <div className="min-h-screen bg-background flex">
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      
      <div className="flex-1 lg:ml-64 transition-all duration-300">
        <div className="h-screen py-6 px-4 lg:px-8">
          <Card className="h-full flex flex-col border-border shadow-2xl">
            <div className="flex items-center gap-3 p-4 border-b border-border bg-card">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="lg:hidden"
              >
                <Menu className="h-5 w-5" />
              </Button>
              <h1 className="text-xl font-semibold text-foreground">
                {activeTab === 'chat' && 'Luật sư AI - MrLawyerAI'}
                {activeTab === 'upload' && 'Upload File'}
                {activeTab === 'history' && 'History'}
                {activeTab === 'sources' && 'Sources'}
              </h1>
            </div>
            <CardContent className="flex-1 overflow-hidden p-0">
              {renderContent()}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

export default App
