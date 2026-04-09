// 系统配置
export const APP_CONFIG = {
  // 系统名称
  SYSTEM_NAME: import.meta.env.VITE_APP_NAME || '知识库 RAG 系统',
  
  // API 基础地址
  API_BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api',
}

// 页面标题
export const PAGE_TITLES = {
  CHAT: import.meta.env.VITE_PAGE_TITLE_CHAT || '智能问答',
  KNOWLEDGE: import.meta.env.VITE_PAGE_TITLE_KNOWLEDGE || '知识库管理',
  USERS: import.meta.env.VITE_PAGE_TITLE_USERS || '用户管理',
}

// 欢迎消息
export const WELCOME_MESSAGES = {
  CHAT_TITLE: import.meta.env.VITE_WELCOME_TITLE || '欢迎使用知识库 RAG 系统',
  CHAT_SUBTITLE: import.meta.env.VITE_WELCOME_SUBTITLE || '您可以问我任何关于产品、服务的问题',
}

// 预设问题
export const PRESET_QUESTIONS = (() => {
  try {
    const preset = import.meta.env.VITE_PRESET_QUESTIONS
    return preset ? JSON.parse(preset) : ['**如何配置？', '**如何配置？', '**如何配置？']
  } catch (e) {
    console.error('解析预设问题失败:', e)
    return ['**如何配置？', '**如何配置？', '**如何配置？']
  }
})()

// 按钮文本
export const BUTTON_TEXTS = {
  NEW_SESSION: import.meta.env.VITE_BTN_NEW_SESSION || '+ 新建会话',
  SEND: import.meta.env.VITE_BTN_SEND || '发送',
  CREATE_USER: import.meta.env.VITE_BTN_CREATE_USER || '+ 创建用户',
  EDIT: import.meta.env.VITE_BTN_EDIT || '编辑',
  DELETE: import.meta.env.VITE_BTN_DELETE || '删除',
  SAVE: import.meta.env.VITE_BTN_SAVE || '保存',
  CANCEL: import.meta.env.VITE_BTN_CANCEL || '取消',
  LOGOUT: import.meta.env.VITE_BTN_LOGOUT || '退出',
}

// 提示信息
export const MESSAGES = {
  LOADING: import.meta.env.VITE_MSG_LOADING || 'AI 正在思考中...',
  AT_LEAST_ONE_SESSION: import.meta.env.VITE_MSG_AT_LEAST_ONE_SESSION || '至少保留一个会话',
  CONFIRM_DELETE_SESSION: import.meta.env.VITE_MSG_CONFIRM_DELETE_SESSION || '确定要删除当前会话吗？',
  CONFIRM_DELETE_USER: (username: string) => `确定要删除用户 "${username}" 吗？`,
  SESSION_DELETED: import.meta.env.VITE_MSG_SESSION_DELETED || '删除成功',
  SESSION_DELETE_FAILED: import.meta.env.VITE_MSG_SESSION_DELETE_FAILED || '删除会话失败',
  USER_CREATED: import.meta.env.VITE_MSG_USER_CREATED || '创建成功',
  USER_CREATE_FAILED: import.meta.env.VITE_MSG_USER_CREATE_FAILED || '创建用户失败',
  USER_UPDATED: import.meta.env.VITE_MSG_USER_UPDATED || '更新成功',
  USER_UPDATE_FAILED: import.meta.env.VITE_MSG_USER_UPDATE_FAILED || '更新失败',
  USER_DELETED: import.meta.env.VITE_MSG_USER_DELETED || '删除成功',
  USER_DELETE_FAILED: import.meta.env.VITE_MSG_USER_DELETE_FAILED || '删除失败',
  LOAD_SESSIONS_FAILED: import.meta.env.VITE_MSG_LOAD_SESSIONS_FAILED || '加载会话失败',
  LOAD_MESSAGES_FAILED: import.meta.env.VITE_MSG_LOAD_MESSAGES_FAILED || '加载消息失败',
  LOAD_USERS_FAILED: import.meta.env.VITE_MSG_LOAD_USERS_FAILED || '加载用户列表失败',
  SEND_MESSAGE_FAILED: import.meta.env.VITE_MSG_SEND_MESSAGE_FAILED || '抱歉，发送消息失败，请稍后重试。',
  UNAUTHORIZED: import.meta.env.VITE_MSG_UNAUTHORIZED || '未授权，请重新登录',
  NO_PERMISSION: import.meta.env.VITE_MSG_NO_PERMISSION || '无权访问此页面',
  EMPTY_USERS: import.meta.env.VITE_MSG_EMPTY_USERS || '暂无用户数据',
}

// 角色文本
export const ROLE_TEXTS = {
  ADMIN: import.meta.env.VITE_ROLE_ADMIN || '管理员',
  USER: import.meta.env.VITE_ROLE_USER || '用户',
}

// 权限文本
export const PERMISSION_TEXTS: Record<string, string> = {
  read: import.meta.env.VITE_PERM_READ || '读取',
  write: import.meta.env.VITE_PERM_WRITE || '写入',
  delete: import.meta.env.VITE_PERM_DELETE || '删除',
  manage_users: import.meta.env.VITE_PERM_MANAGE_USERS || '用户管理',
}
