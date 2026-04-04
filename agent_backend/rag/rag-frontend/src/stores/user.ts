import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useUserStore = defineStore('user', () => {
  const userId = ref<string>('')
  const username = ref<string>('')
  const role = ref<string>('')
  const token = ref<string>('')
  const isLoggedIn = ref<boolean>(false)

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
