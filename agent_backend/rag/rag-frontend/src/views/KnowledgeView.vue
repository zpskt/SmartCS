<template>
  <div class="knowledge-container">
    <div class="header">
      <div class="header-content">
        <button @click="goBack" class="back-btn">← 返回</button>
        <h1>知识库管理</h1>
      </div>
    </div>

    <!-- Tab 切换 -->
    <div class="tabs">
      <button 
        :class="['tab-btn', { active: activeTab === 'list' }]" 
        @click="activeTab = 'list'"
      >
        📋 文档列表
      </button>
      <button 
        :class="['tab-btn', { active: activeTab === 'add' }]" 
        @click="activeTab = 'add'"
      >
        ➕ 新增文档
      </button>
    </div>

    <!-- 文档列表 Tab -->
    <div v-if="activeTab === 'list'" class="tab-content">
      <div class="list-header">
        <div class="filter-group">
          <select v-model="filterType" @change="loadKnowledgeList">
            <option value="">全部类型</option>
            <option value="manual">手动添加</option>
            <option value="feishu">飞书文档</option>
            <option value="upload">上传文件</option>
            <option value="web">网页</option>
          </select>
        </div>
        <button @click="loadKnowledgeList" class="refresh-btn">🔄 刷新</button>
      </div>

      <div v-if="loading" class="loading">加载中...</div>
      
      <div v-else-if="knowledgeList.length === 0" class="empty-state">
        <p>暂无文档，点击“新增文档”开始添加</p>
      </div>

      <div v-else class="document-list">
        <div v-for="doc in knowledgeList" :key="doc.id" class="document-item">
          <div class="doc-info">
            <h3>{{ doc.title }}</h3>
            <div class="doc-meta">
              <span class="source-tag">{{ getSourceTypeName(doc.source_type) }}</span>
              <span class="time">{{ formatDate(doc.created_at) }}</span>
            </div>
            <div v-if="doc.metadata && Object.keys(doc.metadata).length > 0" class="metadata-preview">
              <strong>元数据:</strong>
              <span v-for="(value, key) in doc.metadata" :key="key" class="meta-item">
                {{ key }}: {{ value }}
              </span>
            </div>
          </div>
          <div class="doc-actions">
            <button @click="editDocument(doc)" class="edit-btn">编辑</button>
            <button @click="deleteDocument(doc.doc_id)" class="delete-btn">删除</button>
          </div>
        </div>
      </div>

      <!-- 分页 -->
      <div v-if="total > 0" class="pagination">
        <button @click="prevPage" :disabled="currentPage === 1">上一页</button>
        <span>第 {{ currentPage }} / {{ totalPages }} 页 (共 {{ total }} 条)</span>
        <button @click="nextPage" :disabled="currentPage === totalPages">下一页</button>
      </div>
    </div>

    <!-- 新增文档 Tab -->
    <div v-if="activeTab === 'add'" class="tab-content">
      <div class="add-tabs">
        <button 
          :class="['add-tab-btn', { active: addMode === 'manual' }]" 
          @click="addMode = 'manual'"
        >
          ✍️ 手动录入
        </button>
        <button 
          :class="['add-tab-btn', { active: addMode === 'upload' }]" 
          @click="addMode = 'upload'"
        >
          📤 上传文件
        </button>
        <button 
          :class="['add-tab-btn', { active: addMode === 'feishu' }]" 
          @click="addMode = 'feishu'"
        >
          🔗 飞书文档
        </button>
      </div>

      <!-- 手动录入 -->
      <div v-if="addMode === 'manual'" class="add-form-container">
        <form @submit.prevent="handleAddDocument" class="add-form">
          <div class="form-group">
            <label for="title">标题 *</label>
            <input
              id="title"
              v-model="addForm.title"
              type="text"
              placeholder="请输入文档标题"
              required
            />
          </div>
          
          <div class="form-group">
            <label for="content">内容 *</label>
            <textarea
              id="content"
              v-model="addForm.content"
              placeholder="请输入文档内容"
              rows="8"
              required
            ></textarea>
          </div>
          
          <div class="form-row">
            <div class="form-group">
              <label for="sourceType">来源类型</label>
              <select id="sourceType" v-model="addForm.source_type">
                <option value="manual">手动添加</option>
                <option value="web">网页</option>
                <option value="other">其他</option>
              </select>
            </div>
            
            <div class="form-group">
              <label for="sourceUrl">来源链接</label>
              <input
                id="sourceUrl"
                v-model="addForm.source_url"
                type="url"
                placeholder="https://..."
              />
            </div>
          </div>

          <!-- 元数据 -->
          <div class="form-group">
            <label>元数据 (JSON格式，可选)</label>
            <textarea
              v-model="metadataText"
              placeholder='{"category": "技术", "tags": ["AI", "RAG"]}'
              rows="3"
            ></textarea>
            <small>示例: {"category": "技术", "author": "张三"}</small>
          </div>
          
          <button type="submit" :disabled="adding" class="submit-btn">
            {{ adding ? '添加中...' : '添加文档' }}
          </button>
        </form>
      </div>

      <!-- 上传文件 -->
      <div v-if="addMode === 'upload'" class="upload-container">
        <div class="upload-area" @dragover.prevent @drop="handleFileDrop">
          <input
            ref="fileInput"
            type="file"
            @change="handleFileSelect"
            accept=".txt,.pdf,.doc,.docx,.md,.csv"
            style="display: none"
          />
          <button @click="$refs.fileInput.click()" class="upload-btn">
            选择文件
          </button>
          <p>或拖拽文件到此处</p>
          <small>支持: TXT, PDF, DOC, DOCX, MD, CSV</small>
        </div>

        <div v-if="selectedFile" class="file-info">
          <p><strong>已选择:</strong> {{ selectedFile.name }}</p>
          <button @click="clearFile" class="clear-btn">清除</button>
        </div>

        <!-- 文件元数据 -->
        <div v-if="selectedFile" class="form-group">
          <label>元数据 (JSON格式，可选)</label>
          <textarea
            v-model="uploadMetadataText"
            placeholder='{"category": "产品文档"}'
            rows="3"
          ></textarea>
        </div>

        <button 
          v-if="selectedFile" 
          @click="handleUploadFile" 
          :disabled="uploading" 
          class="submit-btn"
        >
          {{ uploading ? '上传中...' : '开始上传' }}
        </button>
      </div>

      <!-- 飞书文档同步 -->
      <div v-if="addMode === 'feishu'" class="feishu-container">
        <div class="form-group">
          <label for="feishuToken">飞书访问令牌 (可选)</label>
          <input
            id="feishuToken"
            v-model="feishuToken"
            type="password"
            placeholder="输入飞书 API Token（如已配置可留空）"
          />
          <small>如果后端已配置飞书应用凭证，可以留空</small>
        </div>

        <div class="feishu-tips">
          <h4>💡 提示</h4>
          <ul>
            <li>系统将自动同步您有权限的飞书文档</li>
            <li>同步过程可能需要几分钟，请耐心等待</li>
            <li>同步完成后可以在“文档列表”中查看</li>
          </ul>
        </div>

        <button @click="handleSyncFeishu" :disabled="syncing" class="submit-btn">
          {{ syncing ? '同步中...' : '开始同步' }}
        </button>
      </div>
    </div>

    <!-- 编辑对话框 -->
    <div v-if="showEditDialog" class="dialog-overlay" @click="closeEditDialog">
      <div class="dialog" @click.stop>
        <div class="dialog-header">
          <h3>编辑文档</h3>
          <button @click="closeEditDialog" class="close-btn">×</button>
        </div>
        <div class="dialog-body">
          <div class="form-group">
            <label>标题</label>
            <input v-model="editForm.title" type="text" />
          </div>
          <div class="form-group">
            <label>来源类型</label>
            <select v-model="editForm.source_type">
              <option value="manual">手动添加</option>
              <option value="feishu">飞书文档</option>
              <option value="upload">上传文件</option>
              <option value="web">网页</option>
              <option value="other">其他</option>
            </select>
          </div>
          <div class="form-group">
            <label>来源链接</label>
            <input v-model="editForm.source_url" type="url" />
          </div>
          <div class="form-group">
            <label>元数据 (JSON)</label>
            <textarea v-model="editMetadataText" rows="4"></textarea>
          </div>
        </div>
        <div class="dialog-footer">
          <button @click="closeEditDialog" class="cancel-btn">取消</button>
          <button @click="saveEdit" :disabled="updating" class="save-btn">
            {{ updating ? '保存中...' : '保存' }}
          </button>
        </div>
      </div>
    </div>

    <!-- 消息提示 -->
    <div v-if="message" :class="['message', messageType]">
      {{ message }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { knowledgeApi, type KnowledgeAddRequest, type KnowledgeItem } from '@/api'
import { useUserStore } from '@/stores/user'

const router = useRouter()
const activeTab = ref('list')
const addMode = ref('manual')

// 文档列表
const knowledgeList = ref<KnowledgeItem[]>([])
const loading = ref(false)
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)
const filterType = ref('')

// 添加表单
const addForm = ref<KnowledgeAddRequest>({
  title: '',
  content: '',
  source_type: 'manual',
  source_url: '',
  metadata: {}
})
const metadataText = ref('')
const adding = ref(false)

// 文件上传
const selectedFile = ref<File | null>(null)
const uploadMetadataText = ref('')
const uploading = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)

// 飞书同步
const feishuToken = ref('')
const syncing = ref(false)

// 编辑对话框
const showEditDialog = ref(false)
const editingDocId = ref('')
const editForm = ref({
  title: '',
  source_type: '',
  source_url: '',
  metadata: {} as Record<string, any>
})
const editMetadataText = ref('')
const updating = ref(false)

const message = ref('')
const messageType = ref<'success' | 'error'>('success')

const totalPages = computed(() => Math.ceil(total.value / pageSize.value))

onMounted(() => {
  loadKnowledgeList()
})

const showMessage = (msg: string, type: 'success' | 'error' = 'success') => {
  message.value = msg
  messageType.value = type
  setTimeout(() => {
    message.value = ''
  }, 3000)
}

// 加载知识库列表
const loadKnowledgeList = async () => {
  loading.value = true
  try {
    const response = await knowledgeApi.getKnowledgeList({
      page: currentPage.value,
      page_size: pageSize.value,
      source_type: filterType.value || undefined
    })
    knowledgeList.value = response.documents || []
    total.value = response.total || 0
  } catch (error: any) {
    showMessage(error.message || '加载失败', 'error')
  } finally {
    loading.value = false
  }
}

const prevPage = () => {
  if (currentPage.value > 1) {
    currentPage.value--
    loadKnowledgeList()
  }
}

const nextPage = () => {
  if (currentPage.value < totalPages.value) {
    currentPage.value++
    loadKnowledgeList()
  }
}

// 添加文档
const handleAddDocument = async () => {
  if (!addForm.value.title || !addForm.value.content) {
    showMessage('请填写标题和内容', 'error')
    return
  }

  // 解析元数据
  if (metadataText.value.trim()) {
    try {
      addForm.value.metadata = JSON.parse(metadataText.value)
    } catch (e) {
      showMessage('元数据格式错误，请使用正确的 JSON 格式', 'error')
      return
    }
  }

  adding.value = true
  try {
    await knowledgeApi.addDocument(addForm.value)
    showMessage('文档添加成功')
    // 清空表单
    addForm.value = {
      title: '',
      content: '',
      source_type: 'manual',
      source_url: '',
      metadata: {}
    }
    metadataText.value = ''
    // 切换到列表页
    activeTab.value = 'list'
    loadKnowledgeList()
  } catch (error: any) {
    showMessage(error.message || '添加失败', 'error')
  } finally {
    adding.value = false
  }
}

// 文件选择
const handleFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement
  if (target.files && target.files[0]) {
    selectedFile.value = target.files[0]
  }
}

const handleFileDrop = (event: DragEvent) => {
  event.preventDefault()
  if (event.dataTransfer?.files && event.dataTransfer.files[0]) {
    selectedFile.value = event.dataTransfer.files[0]
  }
}

const clearFile = () => {
  selectedFile.value = null
  uploadMetadataText.value = ''
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

// 上传文件
const handleUploadFile = async () => {
  if (!selectedFile.value) {
    showMessage('请选择文件', 'error')
    return
  }

  let metadata: Record<string, any> | undefined
  if (uploadMetadataText.value.trim()) {
    try {
      metadata = JSON.parse(uploadMetadataText.value)
    } catch (e) {
      showMessage('元数据格式错误', 'error')
      return
    }
  }

  uploading.value = true
  try {
    await knowledgeApi.uploadFile(selectedFile.value, metadata)
    showMessage('文件上传成功')
    clearFile()
    activeTab.value = 'list'
    loadKnowledgeList()
  } catch (error: any) {
    showMessage(error.message || '上传失败', 'error')
  } finally {
    uploading.value = false
  }
}

// 飞书同步
const handleSyncFeishu = async () => {
  syncing.value = true
  try {
    await knowledgeApi.syncFeishu(feishuToken.value || undefined)
    showMessage('飞书文档同步成功，请稍后刷新查看')
    feishuToken.value = ''
  } catch (error: any) {
    showMessage(error.message || '同步失败', 'error')
  } finally {
    syncing.value = false
  }
}

// 编辑文档
const editDocument = (doc: KnowledgeItem) => {
  editingDocId.value = doc.id
  editForm.value = {
    title: doc.title,
    source_type: doc.source_type,
    source_url: doc.source_url || '',
    metadata: doc.metadata || {}
  }
  editMetadataText.value = doc.metadata ? JSON.stringify(doc.metadata, null, 2) : ''
  showEditDialog.value = true
}

const closeEditDialog = () => {
  showEditDialog.value = false
  editingDocId.value = ''
  editMetadataText.value = ''
}

const saveEdit = async () => {
  if (!editForm.value.title) {
    showMessage('标题不能为空', 'error')
    return
  }

  let metadata: Record<string, any> | undefined
  if (editMetadataText.value.trim()) {
    try {
      metadata = JSON.parse(editMetadataText.value)
    } catch (e) {
      showMessage('元数据格式错误', 'error')
      return
    }
  }

  updating.value = true
  try {
    await knowledgeApi.updateDocument(editingDocId.value, {
      title: editForm.value.title,
      source_type: editForm.value.source_type,
      source_url: editForm.value.source_url,
      metadata
    })
    showMessage('更新成功')
    closeEditDialog()
    loadKnowledgeList()
  } catch (error: any) {
    showMessage(error.message || '更新失败', 'error')
  } finally {
    updating.value = false
  }
}

// 删除文档
const deleteDocument = async (id: string) => {
  if (!confirm('确定要删除这个文档吗?')) {
    return
  }

  try {
    const userStore = useUserStore()
    console.log("文档id为：{} userId 为 {}", id, userStore.userId)
    await knowledgeApi.deleteDocument(id, userStore.userId)
    showMessage('删除成功')
    loadKnowledgeList()
  } catch (error: any) {
    showMessage(error.message || '删除失败', 'error')
  }
}

const getSourceTypeName = (type: string) => {
  const map: Record<string, string> = {
    manual: '手动添加',
    feishu: '飞书文档',
    upload: '上传文件',
    web: '网页',
    other: '其他'
  }
  return map[type] || type
}

const formatDate = (dateStr: string) => {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  return date.toLocaleString('zh-CN')
}

const goBack = () => {
  router.push('/chat')
}
</script>

<style scoped>
.knowledge-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
  /* 强制白色背景，禁用暗黑模式 */
  background-color: #ffffff !important;
  color: #303133 !important;
}

.header {
  margin-bottom: 20px;
}

.header-content {
  display: flex;
  align-items: center;
  gap: 16px;
}

.back-btn {
  padding: 8px 16px;
  background: #f5f7fa;
  color: #606266;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.3s;
}

.back-btn:hover {
  background: #ecf5ff;
  border-color: #409eff;
  color: #409eff;
}

.header h1 {
  margin: 0;
  color: #333;
  font-size: 28px;
}

/* Tab 样式 */
.tabs {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
  border-bottom: 2px solid #e0e0e0;
}

.tab-btn {
  padding: 12px 24px;
  background: none;
  border: none;
  border-bottom: 3px solid transparent;
  cursor: pointer;
  font-size: 15px;
  color: #666;
  transition: all 0.3s;
  margin-bottom: -2px;
}

.tab-btn:hover {
  color: #409eff;
}

.tab-btn.active {
  color: #409eff;
  border-bottom-color: #409eff;
  font-weight: 500;
}

.tab-content {
  background: white;
  padding: 24px;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
}

/* 文档列表 */
.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.filter-group select {
  padding: 8px 12px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  font-size: 14px;
  cursor: pointer;
}

.refresh-btn {
  padding: 8px 16px;
  background: #409eff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background 0.3s;
}

.refresh-btn:hover {
  background: #66b1ff;
}

.loading {
  text-align: center;
  padding: 40px;
  color: #909399;
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: #909399;
}

.document-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.document-item {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 16px;
  border: 1px solid #ebeef5;
  border-radius: 6px;
  transition: all 0.3s;
}

.document-item:hover {
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  border-color: #409eff;
}

.doc-info {
  flex: 1;
}

.doc-info h3 {
  margin: 0 0 8px 0;
  font-size: 16px;
  color: #303133;
}

.doc-meta {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 8px;
}

.source-tag {
  padding: 2px 8px;
  background: #ecf5ff;
  color: #409eff;
  border-radius: 3px;
  font-size: 12px;
}

.time {
  color: #909399;
  font-size: 12px;
}

.metadata-preview {
  margin-top: 8px;
  padding: 8px;
  background: #f5f7fa;
  border-radius: 4px;
  font-size: 12px;
}

.metadata-preview strong {
  color: #606266;
  margin-right: 8px;
}

.meta-item {
  display: inline-block;
  margin-right: 12px;
  color: #606266;
}

.doc-actions {
  display: flex;
  gap: 8px;
  margin-left: 16px;
}

.edit-btn,
.delete-btn {
  padding: 6px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.3s;
}

.edit-btn {
  background: #409eff;
  color: white;
}

.edit-btn:hover {
  background: #66b1ff;
}

.delete-btn {
  background: #f56c6c;
  color: white;
}

.delete-btn:hover {
  background: #f78989;
}

/* 分页 */
.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 16px;
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid #ebeef5;
}

.pagination button {
  padding: 6px 16px;
  background: #409eff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}

.pagination button:disabled {
  background: #dcdfe6;
  cursor: not-allowed;
}

.pagination span {
  color: #606266;
  font-size: 14px;
}

/* 新增文档 Tab */
.add-tabs {
  display: flex;
  gap: 10px;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid #ebeef5;
}

.add-tab-btn {
  padding: 10px 20px;
  background: #f5f7fa;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  color: #606266;
  transition: all 0.3s;
}

.add-tab-btn:hover {
  background: #ecf5ff;
  border-color: #409eff;
  color: #409eff;
}

.add-tab-btn.active {
  background: #409eff;
  border-color: #409eff;
  color: white;
}

/* 表单样式 */
.add-form {
  max-width: 800px;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  font-weight: 500;
  color: #606266;
  font-size: 14px;
}

.form-group input,
.form-group textarea,
.form-group select {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  font-size: 14px;
  transition: border-color 0.3s;
  font-family: inherit;
}

.form-group input:focus,
.form-group textarea:focus,
.form-group select:focus {
  outline: none;
  border-color: #409eff;
}

.form-group small {
  display: block;
  margin-top: 4px;
  color: #909399;
  font-size: 12px;
}

.form-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.submit-btn {
  padding: 12px 32px;
  background: #67c23a;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background 0.3s;
}

.submit-btn:hover:not(:disabled) {
  background: #85ce61;
}

.submit-btn:disabled {
  background: #b3e19d;
  cursor: not-allowed;
}

/* 上传区域 */
.upload-container {
  max-width: 600px;
}

.upload-area {
  border: 2px dashed #dcdfe6;
  border-radius: 8px;
  padding: 40px;
  text-align: center;
  transition: all 0.3s;
  cursor: pointer;
}

.upload-area:hover {
  border-color: #409eff;
  background: #f5f7fa;
}

.upload-btn {
  padding: 10px 24px;
  background: #409eff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  margin-bottom: 12px;
}

.upload-btn:hover {
  background: #66b1ff;
}

.upload-area p {
  margin: 8px 0;
  color: #606266;
}

.upload-area small {
  color: #909399;
  font-size: 12px;
}

.file-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px;
  background: #f5f7fa;
  border-radius: 4px;
  margin: 16px 0;
}

.clear-btn {
  padding: 6px 12px;
  background: #f56c6c;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.clear-btn:hover {
  background: #f78989;
}

/* 飞书同步 */
.feishu-container {
  max-width: 600px;
}

.feishu-tips {
  padding: 16px;
  background: #ecf5ff;
  border-left: 4px solid #409eff;
  border-radius: 4px;
  margin: 20px 0;
}

.feishu-tips h4 {
  margin: 0 0 12px 0;
  color: #409eff;
}

.feishu-tips ul {
  margin: 0;
  padding-left: 20px;
  color: #606266;
}

.feishu-tips li {
  margin-bottom: 6px;
}

/* 对话框 */
.dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.dialog {
  background: white;
  border-radius: 8px;
  width: 90%;
  max-width: 600px;
  max-height: 80vh;
  overflow-y: auto;
}

.dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid #ebeef5;
}

.dialog-header h3 {
  margin: 0;
  font-size: 18px;
  color: #303133;
}

.close-btn {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: #909399;
  padding: 0;
  width: 30px;
  height: 30px;
  line-height: 1;
}

.close-btn:hover {
  color: #606266;
}

.dialog-body {
  padding: 20px;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding: 16px 20px;
  border-top: 1px solid #ebeef5;
}

.cancel-btn {
  padding: 8px 20px;
  background: #f5f7fa;
  color: #606266;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.cancel-btn:hover {
  background: #ebeef5;
}

.save-btn {
  padding: 8px 20px;
  background: #409eff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.save-btn:hover:not(:disabled) {
  background: #66b1ff;
}

.save-btn:disabled {
  background: #a0cfff;
  cursor: not-allowed;
}

/* 消息提示 */
.message {
  position: fixed;
  top: 20px;
  right: 20px;
  padding: 12px 20px;
  border-radius: 4px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
  z-index: 2000;
  animation: slideIn 0.3s ease-out;
}

.message.success {
  background-color: #f0f9ff;
  color: #67c23a;
  border: 1px solid #c2e7b0;
}

.message.error {
  background-color: #fef0f0;
  color: #f56c6c;
  border: 1px solid #fbc4c4;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

@media (max-width: 768px) {
  .form-row {
    grid-template-columns: 1fr;
  }
  
  .document-item {
    flex-direction: column;
  }
  
  .doc-actions {
    margin-left: 0;
    margin-top: 12px;
    width: 100%;
    justify-content: flex-end;
  }
}
</style>
