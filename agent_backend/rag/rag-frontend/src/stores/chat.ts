import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp?: string
  sources?: Array<{
    title: string
    source_type: string
    snippet: string
  }>
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const sessionId = ref<string>('')
  const isLoading = ref<boolean>(false)

  function setSessionId(id: string) {
    sessionId.value = id
  }

  function addMessage(message: Message) {
    messages.value.push(message)
  }

  function clearMessages() {
    messages.value = []
  }

  function setLoading(loading: boolean) {
    isLoading.value = loading
  }
  
  // 更新最后一条消息的内容（用于流式输出）
  function updateLastMessage(content: string, sources?: Array<{ title: string; source_type: string; snippet: string }>) {
    if (messages.value.length > 0) {
      const lastMessage = messages.value[messages.value.length - 1]
      if (lastMessage.role === 'assistant') {
        lastMessage.content = content
        if (sources) {
          lastMessage.sources = sources
        }
      }
    }
  }
  
  // 追加内容到最后一条消息（用于流式输出）
  function appendToLastMessage(chunk: string) {
    if (messages.value.length > 0) {
      const lastMessage = messages.value[messages.value.length - 1]
      if (lastMessage.role === 'assistant') {
        lastMessage.content += chunk
      }
    }
  }

  return {
    messages,
    sessionId,
    isLoading,
    setSessionId,
    addMessage,
    clearMessages,
    setLoading,
    updateLastMessage,
    appendToLastMessage,
  }
})
