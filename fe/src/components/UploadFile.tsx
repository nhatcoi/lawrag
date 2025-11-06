import { useState, useRef } from 'react'
import { Upload, FileText, Loader2, CheckCircle2, AlertCircle, X } from 'lucide-react'
import { Button } from './ui/Button'
import { Select } from './ui/Select'
import { chatApi } from '@/lib/api'
import { cn } from '@/lib/utils'

interface UploadFileProps {
  onUploadSuccess?: () => void
}

const LAW_TYPES = [
  'Luật Doanh nghiệp',
  'Bộ luật Lao động',
  'Luật Dân sự',
  'Luật Hình sự',
  'Luật Thương mại',
  'Luật Đất đai',
  'Luật Hôn nhân và Gia đình',
  'Luật Thuế',
  'Khác'
]

export function UploadFile({ onUploadSuccess }: UploadFileProps) {
  const [files, setFiles] = useState<File[]>([])
  const [lawType, setLawType] = useState<string>('Khác')
  const [isUploading, setIsUploading] = useState(false)
  const [status, setStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [message, setMessage] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || [])
    const validFiles = selectedFiles.filter(file => 
      file.type === 'application/pdf' || file.name.endsWith('.txt')
    )
    
    if (validFiles.length !== selectedFiles.length) {
      setMessage('Chỉ hỗ trợ file PDF hoặc TXT')
      setStatus('error')
    }
    
    if (validFiles.length > 0) {
      setFiles(prev => [...prev, ...validFiles])
      setStatus('idle')
      setMessage('')
    }
  }

  const handleUpload = async () => {
    if (files.length === 0) return

    setIsUploading(true)
    setStatus('idle')
    setMessage('')

    try {
      if (files.length === 1) {
        await chatApi.upload(files[0], lawType)
        setMessage(`File "${files[0].name}" đã được upload và index thành công!`)
      } else {
        await chatApi.uploadMultiple(files, lawType)
        setMessage(`${files.length} files đã được upload và index thành công!`)
      }
      setStatus('success')
      setFiles([])
      setLawType('Khác')
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
      if (onUploadSuccess) {
        onUploadSuccess()
      }
    } catch (error: any) {
      setStatus('error')
      setMessage(error.response?.data?.detail || 'Có lỗi xảy ra khi upload file')
    } finally {
      setIsUploading(false)
    }
  }

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  return (
    <div className="p-6 space-y-6">
      <div className="space-y-2">
        <h2 className="text-2xl font-semibold text-foreground">Upload File</h2>
        <p className="text-sm text-muted-foreground">
          Upload file PDF hoặc TXT để thêm vào knowledge base
        </p>
      </div>

      <div className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">Loại luật</label>
          <Select
            value={lawType}
            onChange={(e) => setLawType(e.target.value)}
            className="w-full"
          >
            {LAW_TYPES.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </Select>
        </div>

        <div
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-border rounded-xl p-8 text-center cursor-pointer hover:border-primary transition-colors"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt"
            multiple
            onChange={handleFileChange}
            className="hidden"
          />
          <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
          <p className="text-sm text-foreground mb-2">
            Click để chọn file hoặc kéo thả vào đây
          </p>
          <p className="text-xs text-muted-foreground">
            PDF, TXT (có thể chọn nhiều file)
          </p>
        </div>

        {files.length > 0 && (
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {files.map((file, index) => (
              <div
                key={index}
                className="flex items-center gap-3 p-3 rounded-lg bg-muted border border-border group"
              >
                <FileText className="h-4 w-4 text-primary shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">{file.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
                <button
                  onClick={() => removeFile(index)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-500/20 rounded transition-opacity"
                >
                  <X className="h-4 w-4 text-red-500" />
                </button>
              </div>
            ))}
          </div>
        )}

        {status !== 'idle' && (
          <div
            className={cn(
              'flex items-center gap-3 p-4 rounded-lg',
              status === 'success' ? 'bg-green-500/10 border border-green-500/20' : 'bg-red-500/10 border border-red-500/20'
            )}
          >
            {status === 'success' ? (
              <CheckCircle2 className="h-5 w-5 text-green-500" />
            ) : (
              <AlertCircle className="h-5 w-5 text-red-500" />
            )}
            <p
              className={cn(
                'text-sm',
                status === 'success' ? 'text-green-500' : 'text-red-500'
              )}
            >
              {message}
            </p>
          </div>
        )}

        <Button
          onClick={handleUpload}
          disabled={files.length === 0 || isUploading}
          className="w-full"
          size="lg"
        >
          {isUploading ? (
            <>
              <Loader2 className="h-5 w-5 mr-2 animate-spin" />
              Đang upload...
            </>
          ) : (
            <>
              <Upload className="h-5 w-5 mr-2" />
              Upload & Index {files.length > 0 && `(${files.length} files)`}
            </>
          )}
        </Button>
      </div>
    </div>
  )
}

