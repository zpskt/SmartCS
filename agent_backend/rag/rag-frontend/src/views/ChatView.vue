<template>
  <div class="chat-container">
    <!-- 侧边栏 -->
    <aside class="sidebar">
      <div class="sidebar-header">
        <h2>会话列表</h2>
        <button @click="createNewSession" class="new-session-btn">+ 新建会话</button>
      </div>
      <div class="session-list">
        <div
          v-for="session in sessions"
          :key="session.session_id"
          :class="['session-item', { active: session.session_id === chatStore.sessionId }]"
          @click="switchSession(session.session_id)"
        >
          {{ session.title }}
        </div>
      </div>
      <div class="user-info">
        <span>{{ userStore.username }}</span>
        <button @click="handleLogout" class="logout-btn">退出登录</button>
      </div>
    </aside>

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
            <p>{{ message.content }}</p>
            
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
import { ref, onMounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useUserStore } from '@/stores/user'
import { useChatStore } from '@/stores/chat'
import { chatApi, sessionApi } from '@/api'

const router = useRouter()
const userStore = useUserStore()
const chatStore = useChatStore()

const inputMessage = ref('')
const sessions = ref<any[]>([])
const messagesContainer = ref<HTMLElement | null>(null)

onMounted(async () => {
  console.log('用户ID:', userStore.userId)
  // 加载会话列表
  await loadSessions()
  console.log('会话列表:', sessions.value)
  // 如果没有会话，创建一个新会话
  if (sessions.value.length === 0) {
    console.log('没有会话，创建一个新会话')
    await createNewSession()
  } else {
    console.log('有会话，切换到第一个会话')
    // 切换到第一个会话
    console.log('切换到第一个会话:', sessions.value[0].session_id)
    chatStore.setSessionId(sessions.value[0].session_id)
  }
})
// 加载会话
async function loadSessions() {
  try {
    const response = await sessionApi.getSessionList(userStore.userId)
    sessions.value = response.sessions || []
  } catch (err) {
    console.error('加载会话失败:', err)
  }
}

async function createNewSession() {
  try {
    const response = await sessionApi.createSession(userStore.userId)
    if (response.success) {
      chatStore.setSessionId(response.session_id)
      chatStore.clearMessages()
      await loadSessions()
    }
  } catch (err) {
    console.error('创建会话失败:', err)
  }
}

async function switchSession(sessionId: string) {
  chatStore.setSessionId(sessionId)
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

  chatStore.setLoading(true)

  try {
    const response = await chatApi.sendMessage({
      session_id: chatStore.sessionId,
      message: message,
    })

    // 添加 AI 响应
    chatStore.addMessage({
      role: 'assistant',
      content: response.content,
      sources: response.sources,
    })

    await nextTick()
    scrollToBottom()
  } catch (err) {
    console.error('发送消息失败:', err)
    chatStore.addMessage({
      role: 'assistant',
      content: '抱歉，发送消息失败，请稍后重试。',
    })
  } finally {
    chatStore.setLoading(false)
  }
}

function scrollToBottom() {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

function handleLogout() {
  userStore.clearUserInfo()
  chatStore.clearMessages()
  router.push('/login')
}
</script>

<style scoped>
.chat-container {
  display: flex;
  height: 100vh;
  background: #f5f5f5;
}

.sidebar {
  width: 280px;
  background: white;
  border-right: 1px solid #e0e0e0;
  display: flex;
  flex-direction: column;
}

.sidebar-header {
  padding: 20px;
  border-bottom: 1px solid #e0e0e0;
}

.sidebar-header h2 {
  margin: 0 0 10px 0;
  font-size: 18px;
  color: #333;
}

.new-session-btn {
  width: 100%;
  padding: 10px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 14px;
}

.new-session-btn:hover {
  background: #5568d3;
}

.session-list {
  flex: 1;
  overflow-y: auto;
  padding: 10px;
}

.session-item {
  padding: 12px;
  margin-bottom: 5px;
  border-radius: 5px;
  cursor: pointer;
  transition: background 0.2s;
  color: #555;
}

.session-item:hover {
  background: #f0f0f0;
}

.session-item.active {
  background: #e8eaf6;
  color: #667eea;
  font-weight: 500;
}

.user-info {
  padding: 15px;
  border-top: 1px solid #e0e0e0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.logout-btn {
  padding: 6px 12px;
  background: #f44336;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.logout-btn:hover {
  background: #d32f2f;
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
}

.message.user .message-content {
  background: #667eea;
  color: white;
}

.message-content strong {
  display: block;
  margin-bottom: 5px;
}

.message-content p {
  margin: 0;
  line-height: 1.6;
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
