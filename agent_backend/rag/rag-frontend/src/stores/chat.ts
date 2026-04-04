import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface Message {
  role: 'user' | 'assistant'
  content: string
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

  return {
    messages,
    sessionId,
    isLoading,
    setSessionId,
    addMessage,
    clearMessages,
    setLoading,
  }
})
