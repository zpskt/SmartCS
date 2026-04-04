import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useUserStore = defineStore('user', () => {
  // 从 localStorage 恢复用户信息
  const userId = ref<string>(localStorage.getItem('userId') || '')
  const username = ref<string>(localStorage.getItem('username') || '')
  const role = ref<string>('')
  const token = ref<string>(localStorage.getItem('token') || '')
  const isLoggedIn = ref<boolean>(!!localStorage.getItem('token'))

  function setUserInfo(id: string, name: string, userRole: string, userToken: string) {
    userId.value = id
    username.value = name
    role.value = userRole
    token.value = userToken
    isLoggedIn.value = true
    
    localStorage.setItem('token', userToken)
    localStorage.setItem('userId', id)
    localStorage.setItem('username', name)
  }

  function clearUserInfo() {
    userId.value = ''
    username.value = ''
    role.value = ''
    token.value = ''
    isLoggedIn.value = false
    
    localStorage.removeItem('token')
    localStorage.removeItem('userId')
    localStorage.removeItem('username')
  }

  return {
    userId,
    username,
    role,
    token,
    isLoggedIn,
    setUserInfo,
    clearUserInfo,
  }
})
