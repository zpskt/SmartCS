import apiClient from '../utils/api'

export interface LoginRequest {
  username: string
  password: string
}

export interface LoginResponse {
  success: boolean
  user_id?: string
  username?: string
  role?: string
  token?: string
  message?: string
}

export interface ChatMessage {
  session_id: string
  message: string
  stream?: boolean
}

export interface ChatResponse {
  session_id: string
  message_id: string
  content: string
  sources?: Array<{
    title: string
    source_type: string
    snippet: string
  }>
}

export interface KnowledgeDocument {
  title: string
  content: string
  source_type: string
  created_by: string
  source_url?: string
}

export interface KnowledgeAddRequest {
  title: string
  content: string
  source_type?: string
  source_url?: string
  metadata?: Record<string, any>
}

export interface KnowledgeSearchRequest {
  query: string
  limit?: number
  source_type?: string
}

export interface KnowledgeDeleteRequest {
  doc_id: string
  user_id: string
}

export interface KnowledgeItem {
  id: string
  title: string
  content: string
  source_type: string
  source_url?: string
  metadata?: Record<string, any>
  created_at: string
  updated_at?: string
  created_by: string
}

export interface SessionInfo {
  session_id: string
  title: string
  created_at: string
}

export interface SessionUpdateRequest {
  title: string
}

export interface UserInfo {
  id: string
  username: string
  role: string
  created_at: string
}

export interface UserCreateRequest {
  user_id: string
  username: string
  password: string
  role: string
  current_user_id: string
}

export const authApi = {
  login(data: LoginRequest): Promise<LoginResponse> {
    return apiClient.post('/auth/login', data)
  },
}

export const chatApi = {
  sendMessage(data: ChatMessage): Promise<ChatResponse> {
    return apiClient.post('/chat', data)
  },
  
  // 流式发送消息
  async sendMessageStream(
    data: ChatMessage,
    onChunk: (chunk: string) => void,
    onComplete: (sources?: Array<{ title: string; source_type: string; snippet: string }>) => void
  ) {
    // 获取 token
    const token = localStorage.getItem('token')
    
    const response = await fetch('http://localhost:8000/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': token ? `Bearer ${token}` : '',
        'x-authorization': token ? `Bearer ${token}` : '',
      },
      body: JSON.stringify({
        ...data,
        stream: true,
      }),
    })

    if (!response.ok) {
      if (response.status === 401) {
        throw new Error('未授权，请重新登录')
      }
      throw new Error(`请求失败: ${response.status}`)
    }

    const reader = response.body?.getReader()
    const decoder = new TextDecoder()
    let sources: any = undefined
    let fullContent = ''

    while (reader) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value, { stream: true })
      const lines = chunk.split('\n')

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6)
          try {
            const parsed = JSON.parse(data)
            
            // 处理内容块
            if (parsed.type === 'chunk' && parsed.content) {
              fullContent = parsed.content
              onChunk(fullContent)
            }
            
            // 处理完成信号
            if (parsed.type === 'done') {
              onComplete(sources)
              return
            }
            
            // 兼容旧格式（直接有 content 字段）
            if (parsed.content && !parsed.type) {
              fullContent = parsed.content
              onChunk(fullContent)
            }
            if (parsed.sources) {
              sources = parsed.sources
            }
          } catch (e) {
            console.error('解析错误:', e)
          }
        }
      }
    }
    
    onComplete(sources)
  },
}

export const knowledgeApi = {
  // 获取知识库列表
  getKnowledgeList(params?: { page?: number; page_size?: number; source_type?: string }): Promise<{ total: number; items: KnowledgeItem[] }> {
    return apiClient.get('/knowledge/list', { params })
  },
  
  // 添加知识库文档
  addDocument(data: KnowledgeAddRequest) {
    return apiClient.post('/knowledge/add', data)
  },
  
  // 更新知识库文档
  updateDocument(id: string, data: Partial<KnowledgeAddRequest>) {
    return apiClient.put(`/knowledge/${id}`, data)
  },
  
  // 删除知识库文档
  deleteDocument(docId: string, userId: string) {
    return apiClient.post('/knowledge/delete', { doc_id: docId, user_id: userId })
  },
  
  // 搜索知识库
  searchKnowledge(data: KnowledgeSearchRequest): Promise<{ results: KnowledgeItem[] }> {
    return apiClient.post('/knowledge/search', data)
  },
  
  // 同步飞书文档
  syncFeishu(token?: string) {
    return apiClient.post('/knowledge/sync-feishu', { token })
  },
  
  // 上传文件
  uploadFile(file: File, metadata?: Record<string, any>) {
    const formData = new FormData()
    formData.append('file', file)
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata))
    }
    return apiClient.post('/knowledge/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  },
  
  // 兼容旧接口
  search(query: string, limit: number = 5) {
    return apiClient.get('/knowledge/search', { params: { query, limit } })
  },
}

export const sessionApi = {
  // 创建新会话
  createSession(userId: string, title: string = '新会话') {
    return apiClient.post('/session/create', { user_id: userId, title })
  },
  // 根据userId获取会话连接
  getSessionList(userId: string): Promise<{ sessions: SessionInfo[] }> {
    return apiClient.get(`/session/list`, { params: { user_id: userId } })
  },
  
  // 获取会话消息列表
  getSessionMessages(sessionId: string): Promise<{ messages: any[] }> {
    return apiClient.get(`/session/${sessionId}/messages`)
  },
  
  // 清空会话消息
  clearSession(sessionId: string) {
    return apiClient.post(`/session/${sessionId}/clear`)
  },
  
  // 更新会话名称
  updateSession(sessionId: string, data: SessionUpdateRequest) {
    return apiClient.put(`/session/${sessionId}`, data)
  },
}

export const userApi = {
  // 获取用户列表（仅 admin）
  getUserList(): Promise<{ users: UserInfo[] }> {
    return apiClient.get('/users/list')
  },
  
  // 获取用户详情（仅 admin）
  getUserDetail(userId: string): Promise<UserInfo> {
    return apiClient.get(`/users/${userId}`)
  },
  
  // 创建用户（仅 admin）
  createUser(data: UserCreateRequest) {
    return apiClient.post('/users/create', data)
  },
  
  // 更新用户（仅 admin）
  updateUser(userId: string, data: Partial<UserCreateRequest>) {
    return apiClient.put(`/users/${userId}`, data)
  },
  
  // 删除用户（仅 admin）
  deleteUser(userId: string) {
    return apiClient.post('/users/delete', { user_id: userId })
  },
}
