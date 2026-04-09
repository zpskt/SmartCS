<script setup lang="ts">
import { RouterView, useRoute } from 'vue-router'
import { computed } from 'vue'
import { useUserStore } from '@/stores/user'
import { APP_CONFIG, ROLE_TEXTS, BUTTON_TEXTS } from '@/config'

const route = useRoute()
const userStore = useUserStore()

// 判断是否显示全局布局（登录页不显示）
const showLayout = computed(() => route.path !== '/login')

function handleLogout() {
  userStore.clearUserInfo()
  window.location.href = '/login'
}
</script>

<template>
  <div v-if="showLayout" class="app-layout">
    <!-- 顶部栏 -->
    <header class="top-bar">
      <div class="logo">{{ APP_CONFIG.SYSTEM_NAME }}</div>
      <div class="user-section">
        <span class="username">{{ userStore.username }}</span>
        <span class="role-badge">{{ userStore.role === 'admin' ? ROLE_TEXTS.ADMIN : ROLE_TEXTS.USER }}</span>
        <button @click="handleLogout" class="logout-btn">{{ BUTTON_TEXTS.LOGOUT }}</button>
      </div>
    </header>

    <div class="main-container">
      <!-- 左侧导航 -->
      <aside class="sidebar">
        <nav class="nav-menu">
          <router-link to="/chat" class="nav-item">
            <span class="icon">💬</span>
            <span class="text">智能问答</span>
          </router-link>
          <router-link to="/knowledge" class="nav-item">
            <span class="icon">📚</span>
            <span class="text">知识库管理</span>
          </router-link>
          <router-link 
            v-if="userStore.role === 'admin'" 
            to="/users" 
            class="nav-item"
          >
            <span class="icon">👥</span>
            <span class="text">用户管理</span>
          </router-link>
        </nav>
      </aside>

      <!-- 主内容区 -->
      <main class="content">
        <RouterView />
      </main>
    </div>
  </div>
  <RouterView v-else />
</template>

<style scoped>
.app-layout {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  width: 100%;
}

.top-bar {
  height: 64px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 40px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  z-index: 100;
  flex-shrink: 0;
}

.logo {
  font-size: 22px;
  font-weight: 600;
  letter-spacing: 0.5px;
}

.user-section {
  display: flex;
  align-items: center;
  gap: 20px;
}

.username {
  font-size: 15px;
  font-weight: 500;
}

.role-badge {
  padding: 5px 14px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 14px;
  font-size: 13px;
  font-weight: 500;
}

.logout-btn {
  padding: 8px 20px;
  background: rgba(255, 255, 255, 0.9);
  color: #667eea;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s;
}

.logout-btn:hover {
  background: white;
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.main-container {
  display: flex;
  flex: 1;
  overflow: hidden;
  width: 100%;
}

.sidebar {
  width: 240px;
  background: white;
  border-right: 1px solid #e8e8e8;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.05);
  flex-shrink: 0;
}

.nav-menu {
  padding: 24px 0;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 16px 28px;
  margin: 6px 16px;
  text-decoration: none;
  color: #555;
  border-radius: 8px;
  transition: all 0.2s;
  font-size: 15px;
}

.nav-item:hover {
  background: #f5f5f5;
  color: #667eea;
}

.nav-item.router-link-active {
  background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
  color: #667eea;
  font-weight: 600;
  box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
}

.icon {
  font-size: 20px;
}

.text {
  flex: 1;
}

.content {
  flex: 1;
  overflow-y: auto;
  background: #f8f9fa;
  padding: 0;
  width: 100%;
}
</style>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body {
  width: 100%;
  height: 100%;
  overflow-x: hidden;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}

#app {
  width: 100%;
  height: 100vh;
}
</style>
