// 系统配置
export const APP_CONFIG = {
  // 系统名称
  SYSTEM_NAME: '知识库 RAG 系统',
  
  // API 基础地址
  API_BASE_URL: 'http://localhost:8000/api',
}

// 页面标题
export const PAGE_TITLES = {
  CHAT: '智能问答',
  KNOWLEDGE: '知识库管理',
  USERS: '用户管理',
}

// 欢迎消息
export const WELCOME_MESSAGES = {
  CHAT_TITLE: '欢迎使用知识库 RAG 系统',
  CHAT_SUBTITLE: '您可以问我任何关于产品、服务的问题',
}

// 预设问题
export const PRESET_QUESTIONS = [
  '**如何配置？',
  '**如何配置？',
  '**如何配置？',
]

// 按钮文本
export const BUTTON_TEXTS = {
  NEW_SESSION: '+ 新建会话',
  SEND: '发送',
  CREATE_USER: '+ 创建用户',
  EDIT: '编辑',
  DELETE: '删除',
  SAVE: '保存',
  CANCEL: '取消',
  LOGOUT: '退出',
}

// 提示信息
export const MESSAGES = {
  LOADING: 'AI 正在思考中...',
  AT_LEAST_ONE_SESSION: '至少保留一个会话',
  CONFIRM_DELETE_SESSION: '确定要删除当前会话吗？',
  CONFIRM_DELETE_USER: (username: string) => `确定要删除用户 "${username}" 吗？`,
  SESSION_DELETED: '删除成功',
  SESSION_DELETE_FAILED: '删除会话失败',
  USER_CREATED: '创建成功',
  USER_CREATE_FAILED: '创建用户失败',
  USER_UPDATED: '更新成功',
  USER_UPDATE_FAILED: '更新失败',
  USER_DELETED: '删除成功',
  USER_DELETE_FAILED: '删除失败',
  LOAD_SESSIONS_FAILED: '加载会话失败',
  LOAD_MESSAGES_FAILED: '加载消息失败',
  LOAD_USERS_FAILED: '加载用户列表失败',
  SEND_MESSAGE_FAILED: '抱歉，发送消息失败，请稍后重试。',
  UNAUTHORIZED: '未授权，请重新登录',
  NO_PERMISSION: '无权访问此页面',
  EMPTY_USERS: '暂无用户数据',
}

// 角色文本
export const ROLE_TEXTS = {
  ADMIN: '管理员',
  USER: '用户',
}

// 权限文本
export const PERMISSION_TEXTS: Record<string, string> = {
  read: '读取',
  write: '写入',
  delete: '删除',
  manage_users: '用户管理',
}
