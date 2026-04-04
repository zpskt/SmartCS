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

export interface SessionInfo {
  session_id: string
  title: string
  created_at: string
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
}

export const knowledgeApi = {
  addDocument(data: KnowledgeDocument) {
    return apiClient.post('/knowledge', data)
  },
  
  search(query: string, limit: number = 5) {
    return apiClient.get('/knowledge/search', { params: { query, limit } })
  },
}

export const sessionApi = {
  createSession(userId: string, title: string = '新会话') {
    return apiClient.post('/session', { user_id: userId, title })
  },
  
  getSessionList(userId: string): Promise<SessionInfo[]> {
    return apiClient.get(`/sessions/${userId}`)
  },
  
  clearSession(sessionId: string) {
    return apiClient.delete(`/session/${sessionId}/messages`)
  },
}
