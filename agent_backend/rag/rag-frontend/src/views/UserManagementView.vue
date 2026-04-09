<template>
  <div class="user-management">
    <!-- 主内容区域 -->
    <div class="content-wrapper">
      <div class="content-header">
        <h1>用户管理</h1>
        <button @click="showCreateModal = true" class="create-btn">+ 创建用户</button>
      </div>

      <!-- 用户列表 -->
      <div class="user-list">
        <table v-if="users.length > 0">
          <thead>
            <tr>
              <th>用户名</th>
              <th>角色</th>
              <th>权限</th>
              <th>创建时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="user in users" :key="user.user_id">
              <td>{{ user.username }}</td>
              <td>
                <select 
                  v-model="user.role" 
                  @change="updateUser(user)"
                  :disabled="user.user_id === userStore.userId"
                >
                  <option value="admin">管理员</option>
                  <option value="user">普通用户</option>
                </select>
              </td>
              <td>
                <div class="permissions">
                  <span 
                    v-for="perm in user.permissions" 
                    :key="perm" 
                    class="permission-tag"
                  >
                    {{ getPermissionText(perm) }}
                  </span>
                </div>
              </td>
              <td>{{ formatDate(user.created_at) }}</td>
              <td>
                <div class="action-buttons">
                  <button 
                    @click="editUser(user)" 
                    class="edit-btn"
                  >
                    编辑
                  </button>
                  <button 
                    @click="deleteUser(user)" 
                    class="delete-btn"
                    :disabled="user.user_id === userStore.userId"
                  >
                    删除
                  </button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
        <div v-else class="empty-state">
          <p>暂无用户数据</p>
        </div>
      </div>
    </div>

    <!-- 创建用户模态框 -->
    <div v-if="showCreateModal" class="modal-overlay" @click.self="closeModal">
      <div class="modal">
        <div class="modal-header">
          <h3>创建新用户</h3>
          <button @click="closeModal" class="close-btn">×</button>
        </div>
        <div class="modal-body">
          <div class="form-group">
            <label>用户名</label>
            <input v-model="newUser.username" type="text" placeholder="请输入用户名" />
          </div>
          <div class="form-group">
            <label>密码</label>
            <input v-model="newUser.password" type="password" placeholder="请输入密码" />
          </div>
          <div class="form-group">
            <label>角色</label>
            <select v-model="newUser.role">
              <option value="user">普通用户</option>
              <option value="admin">管理员</option>
            </select>
          </div>
        </div>
        <div class="modal-footer">
          <button @click="closeModal" class="cancel-btn">取消</button>
          <button @click="createUser" class="submit-btn" :disabled="!canSubmit">创建</button>
        </div>
      </div>
    </div>

    <!-- 编辑用户模态框 -->
    <div v-if="showEditModal" class="modal-overlay" @click.self="closeEditModal">
      <div class="modal">
        <div class="modal-header">
          <h3>编辑用户</h3>
          <button @click="closeEditModal" class="close-btn">×</button>
        </div>
        <div class="modal-body">
          <div class="form-group">
            <label>用户名</label>
            <input v-model="editUserForm.username" type="text" placeholder="请输入用户名" />
          </div>
          <div class="form-group">
            <label>新密码（留空则不修改）</label>
            <input v-model="editUserForm.password" type="password" placeholder="输入新密码" />
          </div>
          <div class="form-group">
            <label>角色</label>
            <select v-model="editUserForm.role">
              <option value="user">普通用户</option>
              <option value="admin">管理员</option>
            </select>
          </div>
        </div>
        <div class="modal-footer">
          <button @click="closeEditModal" class="cancel-btn">取消</button>
          <button @click="saveEditUser" class="submit-btn" :disabled="!canEditSubmit">保存</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useUserStore } from '@/stores/user'
import { userApi } from '@/api'

const router = useRouter()
const userStore = useUserStore()

const users = ref<any[]>([])
const showCreateModal = ref(false)
const showEditModal = ref(false)
const newUser = ref({
  user_id: '',
  username: '',
  password: '',
  role: 'user',
  current_user_id: userStore.userId
})
const editUserForm = ref({
  user_id: '',
  username: '',
  password: '',
  role: 'user'
})

const canSubmit = computed(() => {
  return newUser.value.username && newUser.value.password
})

const canEditSubmit = computed(() => {
  return editUserForm.value.username
})

onMounted(async () => {
  // 检查权限
  if (userStore.role !== 'admin') {
    alert('无权访问此页面')
    router.push('/chat')
    return
  }
  
  await loadUsers()
})

async function loadUsers() {
  try {
    const response = await userApi.getUserList()
    users.value = response.users || []
  } catch (err) {
    console.error('加载用户列表失败:', err)
    alert('加载用户列表失败')
  }
}

async function createUser() {
  try {
    const createData = {
      user_id: newUser.value.username,
      username: newUser.value.username,
      password: newUser.value.password,
      role: newUser.value.role,
      current_user_id: userStore.userId
    }
    await userApi.createUser(createData)
    alert('创建成功')
    closeModal()
    await loadUsers()
  } catch (err: any) {
    console.error('创建用户失败:', err)
    alert(err.message || '创建用户失败')
  }
}

function editUser(user: any) {
  editUserForm.value = {
    user_id: user.user_id,
    username: user.username,
    password: '',
    role: user.role
  }
  showEditModal.value = true
}

async function saveEditUser() {
  try {
    const updateData: any = {
      username: editUserForm.value.username,
      role: editUserForm.value.role
    }
    // 只有填写了密码才更新
    if (editUserForm.value.password) {
      updateData.password = editUserForm.value.password
    }
    await userApi.updateUser(editUserForm.value.user_id, updateData)
    closeEditModal()
    await loadUsers()
    alert('更新成功')
  } catch (err: any) {
    console.error('更新用户失败:', err)
    alert(err.message || '更新失败')
  }
}

function closeModal() {
  showCreateModal.value = false
  newUser.value = { user_id: '', username: '', password: '', role: 'user', current_user_id: userStore.userId }
}

function closeEditModal() {
  showEditModal.value = false
  editUserForm.value = { user_id: '', username: '', password: '', role: 'user' }
}

async function updateUser(user: any) {
  try {
    await userApi.updateUser(user.user_id, { role: user.role })
    alert('更新成功')
  } catch (err: any) {
    console.error('更新用户失败:', err)
    alert(err.message || '更新失败')
    await loadUsers() // 恢复原值
  }
}

async function deleteUser(user: any) {
  if (!confirm(`确定要删除用户 "${user.username}" 吗？`)) {
    return
  }
  
  try {
    await userApi.deleteUser(user.user_id)
    alert('删除成功')
    await loadUsers()
  } catch (err: any) {
    console.error('删除用户失败:', err)
    alert(err.message || '删除失败')
  }
}

function formatDate(dateStr: string) {
  if (!dateStr) return '-'
  const date = new Date(dateStr)
  return date.toLocaleString('zh-CN')
}

function getPermissionText(perm: string) {
  const map: Record<string, string> = {
    read: '读取',
    write: '写入',
    delete: '删除',
    manage_users: '用户管理'
  }
  return map[perm] || perm
}

function handleLogout() {
  userStore.clearUserInfo()
  router.push('/login')
}
</script>

<style scoped>
.user-management {
  padding: 30px;
  height: 100%;
  width: 100%;
}

.content-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
}

.content-header h1 {
  margin: 0;
  font-size: 24px;
  color: #333;
}

.create-btn {
  padding: 10px 20px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 14px;
}

.create-btn:hover {
  background: #5568d3;
}

.user-list {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

table {
  width: 100%;
  border-collapse: collapse;
}

thead {
  background: #f5f5f5;
}

th {
  padding: 15px;
  text-align: left;
  font-weight: 600;
  color: #333;
  border-bottom: 2px solid #e0e0e0;
}

td {
  padding: 15px;
  border-bottom: 1px solid #e0e0e0;
  color: #555;
}

tr:hover {
  background: #f9f9f9;
}

select {
  padding: 6px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  cursor: pointer;
}

select:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.permissions {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.permission-tag {
  display: inline-block;
  padding: 2px 8px;
  background: #e3f2fd;
  color: #1976d2;
  border-radius: 3px;
  font-size: 11px;
  white-space: nowrap;
}

.action-buttons {
  display: flex;
  gap: 8px;
}

.edit-btn {
  padding: 6px 12px;
  background: #4caf50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.edit-btn:hover {
  background: #45a049;
}

.delete-btn {
  padding: 6px 12px;
  background: #f44336;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.delete-btn:hover:not(:disabled) {
  background: #d32f2f;
}

.delete-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.empty-state {
  padding: 60px;
  text-align: center;
  color: #999;
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
  max-width: 500px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid #e0e0e0;
}

.modal-header h3 {
  margin: 0;
  font-size: 18px;
  color: #333;
}

.close-btn {
  background: none;
  border: none;
  font-size: 28px;
  color: #999;
  cursor: pointer;
  line-height: 1;
}

.close-btn:hover {
  color: #333;
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
  color: #333;
}

.form-group input,
.form-group select {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  box-sizing: border-box;
}

.form-group input:focus,
.form-group select:focus {
  outline: none;
  border-color: #667eea;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 20px;
  border-top: 1px solid #e0e0e0;
}

.cancel-btn {
  padding: 10px 20px;
  background: #f5f5f5;
  color: #333;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.cancel-btn:hover {
  background: #e0e0e0;
}

.submit-btn {
  padding: 10px 20px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.submit-btn:hover:not(:disabled) {
  background: #5568d3;
}

.submit-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
