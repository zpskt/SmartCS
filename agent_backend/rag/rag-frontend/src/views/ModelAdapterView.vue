<template>
  <div class="model-adapter-container">
    <!-- 会话选择栏 -->
    <div class="session-bar">
      <div class="session-selector">
        <select v-model="currentSessionId" @change="switchSession" class="session-select">
          <option v-for="session in sessions" :key="session.session_id" :value="session.session_id">
            {{ session.title }}
          </option>
        </select>
        <button @click="showEditSessionModal = true" class="edit-session-btn" title="修改会话名称">
          ✏️
        </button>
        <button @click="createNewSession" class="new-session-btn">+ 新建会话</button>
        <button 
          @click="deleteCurrentSession" 
          class="delete-session-btn"
          :disabled="sessions.length <= 1"
          title="删除当前会话"
        >
          🗑️
        </button>
      </div>
    </div>

    <!-- 修改会话名称模态框 -->
    <div v-if="showEditSessionModal" class="modal-overlay" @click.self="closeEditSessionModal">
      <div class="modal">
        <div class="modal-header">
          <h3>修改会话名称</h3>
          <button @click="closeEditSessionModal" class="close-btn">×</button>
        </div>
        <div class="modal-body">
          <div class="form-group">
            <label>会话名称</label>
            <input 
              v-model="editSessionTitle" 
              type="text" 
              placeholder="请输入会话名称" 
              @keydown.enter="updateSessionTitle"
            />
          </div>
        </div>
        <div class="modal-footer">
          <button @click="closeEditSessionModal" class="cancel-btn">取消</button>
          <button @click="updateSessionTitle" class="submit-btn" :disabled="!editSessionTitle.trim()">保存</button>
        </div>
      </div>
    </div>

    <!-- 主聊天区域 -->
    <main class="chat-main">
      <!-- 消息列表 -->
      <div class="messages-container" ref="messagesContainer">
        <div v-if="messages.length === 0" class="welcome-message">
          <h3>🔧 型号适配助手</h3>
          <p>专业的产品型号查询和管理工具</p>
          
          <!-- 预设问题 -->
          <div class="preset-questions">
            <button 
              v-for="(question, index) in PRESET_QUESTIONS" 
              :key="index"
              @click="sendPresetQuestion(question)"
              class="preset-question-btn"
            >
              {{ question }}
            </button>
          </div>
        </div>
        
        <div
          v-for="(message, index) in messages"
          :key="index"
          :class="['message', message.role]"
        >
          <div class="message-content">
            <strong>{{ message.role === 'user' ? '您' : '型号助手' }}:</strong>
            <div v-if="message.role === 'user'" class="message-text">{{ message.content }}</div>
            <div v-else class="message-text markdown-body" v-html="renderMarkdown(message.content)"></div>
            
            <!-- 显示时间 -->
            <div v-if="message.timestamp" class="message-time">
              {{ formatTime(message.timestamp) }}
            </div>
          </div>
        </div>
        
        <div v-if="isLoading" class="loading-message">
          正在查询型号信息...
        </div>
      </div>

      <!-- 输入框 -->
      <div class="input-container">
        <textarea
          v-model="inputMessage"
          @keydown.enter.prevent="sendMessage"
          placeholder="询问型号相关问题... (按 Enter 发送)"
          :disabled="isLoading"
          rows="3"
        ></textarea>
        <button
          @click="sendMessage"
          :disabled="!inputMessage.trim() || isLoading"
          class="send-btn"
        >
          发送
        </button>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue'
import { modelAdapterApi, sessionApi } from '@/api'
import { marked } from 'marked'

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp?: string
}

const inputMessage = ref('')
const messages = ref<Message[]>([])
const messagesContainer = ref<HTMLElement | null>(null)
const sessions = ref<any[]>([])
const currentSessionId = ref('')
const showEditSessionModal = ref(false)
const editSessionTitle = ref('')
const isLoading = ref(false)

// 预设问题
const PRESET_QUESTIONS = [
  '有哪些可用的型号？',
  'M001型号适配了哪些功能？',
  '查询M002的详细信息',
  '搜索包含"智能"的型号'
]

onMounted(async () => {
  await loadSessions()
  if (sessions.value.length === 0) {
    await createNewSession()
  } else {
    currentSessionId.value = sessions.value[0].session_id
    await loadSessionMessages(currentSessionId.value)
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

async function loadSessionMessages(sessionId: string) {
  try {
    const response = await sessionApi.getSessionMessages(sessionId)
    const msgs = response.messages || []
    
    messages.value = msgs.map((msg: any) => ({
      role: msg.role,
      content: msg.content,
      timestamp: msg.timestamp
    }))
  } catch (err) {
    console.error('加载消息失败:', err)
  }
}

async function createNewSession() {
  try {
    const userId = localStorage.getItem('userId') || ''
    const response = await sessionApi.createSession(userId, '型号适配会话')
    if (response.success) {
      messages.value = []
      await loadSessions()
      currentSessionId.value = response.session_id
    }
  } catch (err) {
    console.error('创建会话失败:', err)
  }
}

async function switchSession() {
  if (!currentSessionId.value) return
  messages.value = []
  await loadSessionMessages(currentSessionId.value)
}

async function deleteCurrentSession() {
  if (sessions.value.length <= 1) {
    alert('至少保留一个会话')
    return
  }
  
  if (!confirm('确定要删除当前会话吗？')) {
    return
  }
  
  try {
    await sessionApi.clearSession(currentSessionId.value)
    await loadSessions()
    
    if (sessions.value.length > 0) {
      currentSessionId.value = sessions.value[0].session_id
      await loadSessionMessages(currentSessionId.value)
    }
  } catch (err) {
    console.error('删除会话失败:', err)
    alert('删除会话失败')
  }
}

async function updateSessionTitle() {
  if (!editSessionTitle.value.trim()) return
  
  try {
    await sessionApi.updateSession(currentSessionId.value, { title: editSessionTitle.value })
    closeEditSessionModal()
    await loadSessions()
  } catch (err) {
    console.error('更新会话名称失败:', err)
    alert('更新会话名称失败')
  }
}

function closeEditSessionModal() {
  showEditSessionModal.value = false
  editSessionTitle.value = ''
}

async function sendMessage() {
  if (!inputMessage.value.trim() || isLoading.value) return

  const message = inputMessage.value.trim()
  inputMessage.value = ''

  // 添加用户消息
  messages.value.push({
    role: 'user',
    content: message,
    timestamp: new Date().toISOString()
  })

  // 添加空的 AI 消息
  messages.value.push({
    role: 'assistant',
    content: '',
    timestamp: new Date().toISOString()
  })

  isLoading.value = true

  try {
    let lastContent = ''
    await modelAdapterApi.sendMessageStream(
      {
        session_id: currentSessionId.value,
        message: message,
      },
      // 流式内容回调
      (fullText) => {
        const newText = fullText.slice(lastContent.length)
        lastContent = fullText
        typeWriterEffect(newText)
      },
      // 完成回调
      () => {
        isLoading.value = false
      }
    )
  } catch (err: any) {
    console.error('发送消息失败:', err)
    const errorMsg = err.message || '抱歉，发送消息失败，请稍后重试。'
    messages.value[messages.value.length - 1].content = errorMsg
    isLoading.value = false
  }
}

function typeWriterEffect(text: string) {
  const currentIndex = messages.value.length - 1
  let charIndex = 0
  
  function typeChar() {
    if (charIndex < text.length) {
      messages.value[currentIndex].content += text[charIndex]
      charIndex++
      nextTick(() => scrollToBottom())
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

function renderMarkdown(content: string) {
  if (!content) return ''
  return marked(content)
}

function formatTime(timestamp: string) {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  const hours = String(date.getHours()).padStart(2, '0')
  const minutes = String(date.getMinutes()).padStart(2, '0')
  const seconds = String(date.getSeconds()).padStart(2, '0')
  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`
}

function sendPresetQuestion(question: string) {
  inputMessage.value = question
  sendMessage()
}
</script>

<style scoped>
.model-adapter-container {
  height: 100%;
  display: flex;
  flex-direction: column;
  width: 100%;
}

.session-bar {
  position: sticky;
  top: 0;
  z-index: 100;
  padding: 20px 30px;
  background: white;
  border-bottom: 1px solid #e8e8e8;
  flex-shrink: 0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.session-selector {
  display: flex;
  gap: 12px;
  align-items: center;
  width: 100%;
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

.new-session-btn {
  padding: 8px 16px;
  background: #ff6b6b;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  white-space: nowrap;
  transition: all 0.2s;
}

.new-session-btn:hover {
  background: #ee5a52;
  transform: translateY(-1px);
}

.edit-session-btn {
  padding: 8px 12px;
  background: #4caf50;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 16px;
  transition: all 0.2s;
}

.delete-session-btn {
  padding: 8px 12px;
  background: #f44336;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 16px;
  transition: all 0.2s;
}

.delete-session-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 30px;
}

.welcome-message {
  text-align: center;
  margin-top: 80px;
  color: #666;
}

.welcome-message h3 {
  margin-bottom: 15px;
  font-size: 28px;
  color: #ff6b6b;
}

.preset-questions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: center;
  margin-top: 30px;
}

.preset-question-btn {
  padding: 10px 20px;
  background: white;
  border: 1px solid #ff6b6b;
  color: #ff6b6b;
  border-radius: 20px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.preset-question-btn:hover {
  background: #ff6b6b;
  color: white;
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(255, 107, 107, 0.3);
}

.message {
  margin-bottom: 24px;
  display: flex;
}

.message.user {
  justify-content: flex-end;
}

.message.assistant {
  justify-content: flex-start;
}

.message-content {
  max-width: 75%;
  padding: 18px 20px;
  border-radius: 12px;
  background: white;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  color: #333;
  line-height: 1.6;
}

.message.user .message-content {
  background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
  color: white;
}

.message-time {
  margin-top: 8px;
  font-size: 12px;
  color: #999;
  text-align: right;
}

.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3) {
  margin: 12px 0 8px 0;
  font-weight: 600;
}

.markdown-body :deep(code) {
  padding: 2px 6px;
  background: rgba(0, 0, 0, 0.06);
  border-radius: 3px;
}

.markdown-body :deep(pre) {
  padding: 12px;
  background: #282c34;
  border-radius: 6px;
  overflow-x: auto;
}

.loading-message {
  text-align: center;
  color: #999;
  padding: 10px;
}

.input-container {
  padding: 24px 30px;
  background: white;
  border-top: 1px solid #e0e0e0;
  display: flex;
  gap: 15px;
}

.input-container textarea {
  flex: 1;
  padding: 14px 16px;
  border: 1px solid #ddd;
  border-radius: 8px;
  resize: none;
  font-family: inherit;
  font-size: 15px;
}

.send-btn {
  padding: 14px 36px;
  background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 15px;
  font-weight: 500;
  align-self: flex-end;
  transition: all 0.2s;
}

.send-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(255, 107, 107, 0.4);
}

.send-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* 模态框样式 */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal {
  background: white;
  border-radius: 8px;
  width: 90%;
  max-width: 400px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid #e0e0e0;
}

.modal-body {
  padding: 20px;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  font-weight: 500;
}

.form-group input {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  box-sizing: border-box;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 20px;
  border-top: 1px solid #e0e0e0;
}

.cancel-btn, .submit-btn {
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.cancel-btn {
  background: #f5f5f5;
  color: #333;
}

.submit-btn {
  background: #ff6b6b;
  color: white;
}

.submit-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
