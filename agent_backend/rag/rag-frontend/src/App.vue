<script setup lang="ts">
import { RouterView, useRoute } from 'vue-router'
import { computed } from 'vue'
import { useUserStore } from '@/stores/user'

const route = useRoute()
const userStore = useUserStore()

// 判断是否显示全局布局（登录页不显示）
const showLayout = computed(() => route.path !== '/login')
</script>

<template>
  <div v-if="showLayout" class="app-layout">
    <!-- 顶部栏 -->
    <header class="top-bar">
      <div class="logo">企业知识库 RAG 系统</div>
      <div class="user-section">
        <span class="username">{{ userStore.username }}</span>
        <span class="role-badge">{{ userStore.role === 'admin' ? '管理员' : '用户' }}</span>
        <button @click="handleLogout" class="logout-btn">退出</button>
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
  height: 100vh;
}

.top-bar {
  height: 60px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 30px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  z-index: 100;
}

.logo {
  font-size: 20px;
  font-weight: 600;
  letter-spacing: 0.5px;
}

.user-section {
  display: flex;
  align-items: center;
  gap: 15px;
}

.username {
  font-size: 14px;
}

.role-badge {
  padding: 4px 12px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}

.logout-btn {
  padding: 6px 16px;
  background: rgba(255, 255, 255, 0.9);
  color: #667eea;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
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
}

.sidebar {
  width: 220px;
  background: white;
  border-right: 1px solid #e8e8e8;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.05);
}

.nav-menu {
  padding: 20px 0;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 24px;
  margin: 4px 12px;
  text-decoration: none;
  color: #555;
  border-radius: 8px;
  transition: all 0.2s;
  font-size: 14px;
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
  font-size: 18px;
}

.text {
  flex: 1;
}

.content {
  flex: 1;
  overflow-y: auto;
  background: #f8f9fa;
}
</style>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}

#app {
  width: 100%;
  height: 100vh;
}
</style>
