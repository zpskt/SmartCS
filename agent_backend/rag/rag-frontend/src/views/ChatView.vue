<template>
  <div class="chat-container">
    <!-- 会话选择栏 -->
    <div class="session-bar">
      <div class="session-selector">
        <select v-model="currentSessionId" @change="switchSession" class="session-select">
          <option v-for="session in sessions" :key="session.session_id" :value="session.session_id">
            {{ session.title }}
          </option>
        </select>
        <button @click="createNewSession" class="new-session-btn">+ 新建会话</button>
      </div>
    </div>

    <!-- 主聊天区域 -->
    <main class="chat-main">
      <!-- 消息列表 -->
      <div class="messages-container" ref="messagesContainer">
        <div v-if="chatStore.messages.length === 0" class="welcome-message">
          <h3>欢迎使用企业知识库 RAG 系统</h3>
          <p>您可以问我任何关于产品、服务的问题</p>
        </div>
        
        <div
          v-for="(message, index) in chatStore.messages"
          :key="index"
          :class="['message', message.role]"
        >
          <div class="message-content">
            <strong>{{ message.role === 'user' ? '您' : 'AI 助手' }}:</strong>
            <div v-if="message.role === 'user'" class="message-text">{{ message.content }}</div>
            <div v-else class="message-text markdown-body" v-html="renderMarkdown(message.content)"></div>
            
            <!-- 显示参考来源 -->
            <div v-if="message.sources && message.sources.length > 0" class="sources">
              <h4>📚 参考资料:</h4>
              <div
                v-for="(source, idx) in message.sources"
                :key="idx"
                class="source-item"
              >
                <strong>{{ source.title }}</strong>
                <p>{{ source.snippet }}</p>
              </div>
            </div>
          </div>
        </div>
        
        <div v-if="chatStore.isLoading" class="loading-message">
          AI 正在思考中...
        </div>
      </div>

      <!-- 输入框 -->
      <div class="input-container">
        <textarea
          v-model="inputMessage"
          @keydown.enter.prevent="sendMessage"
          placeholder="输入您的问题... (按 Enter 发送)"
          :disabled="chatStore.isLoading"
          rows="3"
        ></textarea>
        <button
          @click="sendMessage"
          :disabled="!inputMessage.trim() || chatStore.isLoading"
          class="send-btn"
        >
          发送
        </button>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick, watch } from 'vue'
import { useChatStore } from '@/stores/chat'
import { chatApi, sessionApi } from '@/api'
import { marked } from 'marked'

const chatStore = useChatStore()

const inputMessage = ref('')
const messagesContainer = ref<HTMLElement | null>(null)
const sessions = ref<any[]>([])
const currentSessionId = ref('')

onMounted(async () => {
  await loadSessions()
  // 如果没有会话，创建一个新会话
  if (sessions.value.length === 0) {
    await createNewSession()
  } else {
    currentSessionId.value = sessions.value[0].session_id
    chatStore.setSessionId(currentSessionId.value)
  }
})

async function loadSessions() {
  try {
    const userId = localStorage.getItem('userId') || ''
    const response = await sessionApi.getSessionList(userId)
    sessions.value = response.sessions || []
  } catch (err) {
    console.error('加载会话失败:', err)
  }
}

async function createNewSession() {
  try {
    const userId = localStorage.getItem('userId') || ''
    const response = await sessionApi.createSession(userId)
    if (response.success) {
      chatStore.setSessionId(response.session_id)
      chatStore.clearMessages()
      await loadSessions()
      currentSessionId.value = response.session_id
    }
  } catch (err) {
    console.error('创建会话失败:', err)
  }
}

async function switchSession() {
  if (!currentSessionId.value) return
  chatStore.setSessionId(currentSessionId.value)
  chatStore.clearMessages()
  // TODO: 加载历史消息
}

async function sendMessage() {
  if (!inputMessage.value.trim() || chatStore.isLoading) return

  const message = inputMessage.value.trim()
  inputMessage.value = ''

  // 添加用户消息
  chatStore.addMessage({
    role: 'user',
    content: message,
  })

  // 添加空的 AI 消息用于流式更新
  chatStore.addMessage({
    role: 'assistant',
    content: '',
  })

  chatStore.setLoading(true)

  try {
    let lastContent = ''
    await chatApi.sendMessageStream(
      {
        session_id: chatStore.sessionId,
        message: message,
      },
      // 流式内容回调 - 实现打字机效果
      (fullText) => {
        // 计算需要追加的新文本
        const newText = fullText.slice(lastContent.length)
        lastContent = fullText
        
        // 逐字显示新文本
        typeWriterEffect(newText)
      },
      // 完成回调
      (sources) => {
        if (sources && sources.length > 0) {
          chatStore.updateLastMessage(chatStore.messages[chatStore.messages.length - 1].content, sources)
        }
        chatStore.setLoading(false)
      }
    )
  } catch (err: any) {
    console.error('发送消息失败:', err)
    const errorMsg = err.message || '抱歉，发送消息失败，请稍后重试。'
    chatStore.updateLastMessage(errorMsg)
    chatStore.setLoading(false)
  }
}

// 打字机效果
function typeWriterEffect(text: string) {
  const currentIndex = chatStore.messages.length - 1
  let charIndex = 0
  
  function typeChar() {
    if (charIndex < text.length) {
      const currentContent = chatStore.messages[currentIndex].content
      chatStore.messages[currentIndex].content = currentContent + text[charIndex]
      charIndex++
      nextTick(() => scrollToBottom())
      
      // 每个字符延迟 30ms，模拟打字效果
      setTimeout(typeChar, 30)
    }
  }
  
  typeChar()
}

function scrollToBottom() {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

// 渲染 Markdown
function renderMarkdown(content: string) {
  if (!content) return ''
  return marked(content)
}
</script>

<style scoped>
.chat-container {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.session-bar {
  padding: 15px 20px;
  background: white;
  border-bottom: 1px solid #e8e8e8;
}

.session-selector {
  display: flex;
  gap: 10px;
  align-items: center;
}

.session-select {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 14px;
  background: white;
  cursor: pointer;
}

.session-select:focus {
  outline: none;
  border-color: #667eea;
}

.new-session-btn {
  padding: 8px 16px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  white-space: nowrap;
  transition: all 0.2s;
}

.new-session-btn:hover {
  background: #5568d3;
  transform: translateY(-1px);
}

.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.welcome-message {
  text-align: center;
  margin-top: 100px;
  color: #999;
}

.welcome-message h3 {
  margin-bottom: 10px;
}

.message {
  margin-bottom: 20px;
  display: flex;
}

.message.user {
  justify-content: flex-end;
}

.message.assistant {
  justify-content: flex-start;
}

.message-content {
  max-width: 70%;
  padding: 15px;
  border-radius: 10px;
  background: white;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  color: #333;
}

.message.user .message-content {
  background: #667eea;
  color: white;
}

.message.assistant .message-content {
  background: white;
  color: #333;
}

.message-content strong {
  display: block;
  margin-bottom: 5px;
  color: inherit;
}

.message-content p {
  margin: 0;
  line-height: 1.6;
  color: inherit;
}

.message-text {
  line-height: 1.6;
}

.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3),
.markdown-body :deep(h4),
.markdown-body :deep(h5),
.markdown-body :deep(h6) {
  margin: 12px 0 8px 0;
  font-weight: 600;
  line-height: 1.4;
}

.markdown-body :deep(h1) { font-size: 20px; }
.markdown-body :deep(h2) { font-size: 18px; }
.markdown-body :deep(h3) { font-size: 16px; }
.markdown-body :deep(h4) { font-size: 15px; }

.markdown-body :deep(p) {
  margin: 8px 0;
}

.markdown-body :deep(code) {
  padding: 2px 6px;
  background: rgba(0, 0, 0, 0.06);
  border-radius: 3px;
  font-family: 'Courier New', monospace;
  font-size: 13px;
}

.markdown-body :deep(pre) {
  padding: 12px;
  background: #282c34;
  border-radius: 6px;
  overflow-x: auto;
  margin: 10px 0;
}

.markdown-body :deep(pre code) {
  background: none;
  padding: 0;
  color: #abb2bf;
  font-size: 13px;
}

.markdown-body :deep(ul),
.markdown-body :deep(ol) {
  margin: 8px 0;
  padding-left: 24px;
}

.markdown-body :deep(li) {
  margin: 4px 0;
}

.markdown-body :deep(blockquote) {
  margin: 10px 0;
  padding: 10px 15px;
  border-left: 4px solid #667eea;
  background: rgba(102, 126, 234, 0.1);
  border-radius: 4px;
}

.markdown-body :deep(a) {
  color: #667eea;
  text-decoration: none;
}

.markdown-body :deep(a:hover) {
  text-decoration: underline;
}

.markdown-body :deep(table) {
  border-collapse: collapse;
  width: 100%;
  margin: 10px 0;
}

.markdown-body :deep(th),
.markdown-body :deep(td) {
  border: 1px solid #ddd;
  padding: 8px 12px;
  text-align: left;
}

.markdown-body :deep(th) {
  background: #f5f5f5;
  font-weight: 600;
}

.markdown-body :deep(img) {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
}

.sources {
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid #e0e0e0;
}

.sources h4 {
  margin: 0 0 10px 0;
  font-size: 14px;
  color: #666;
}

.source-item {
  margin-bottom: 10px;
  padding: 10px;
  background: #f9f9f9;
  border-radius: 5px;
}

.source-item strong {
  color: #667eea;
  font-size: 13px;
}

.source-item p {
  margin: 5px 0 0 0;
  font-size: 12px;
  color: #666;
}

.loading-message {
  text-align: center;
  color: #999;
  padding: 10px;
}

.input-container {
  padding: 20px;
  background: white;
  border-top: 1px solid #e0e0e0;
  display: flex;
  gap: 10px;
}

.input-container textarea {
  flex: 1;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 5px;
  resize: none;
  font-family: inherit;
  font-size: 14px;
}

.input-container textarea:focus {
  outline: none;
  border-color: #667eea;
}

.send-btn {
  padding: 12px 30px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 14px;
  align-self: flex-end;
}

.send-btn:hover:not(:disabled) {
  background: #5568d3;
}

.send-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
