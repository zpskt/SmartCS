# RAG 前端项目

企业知识库 RAG 系统的 Vue 3 前端应用。

## 技术栈

- Vue 3 (Composition API)
- TypeScript
- Vue Router
- Pinia (状态管理)
- Axios (HTTP 客户端)

## 安装依赖

```bash
npm install
```

## 开发运行

```bash
npm run dev
```

默认访问地址: http://localhost:5173

## 构建生产版本

```bash
npm run build
```

## 配置后端 API

在 `src/utils/api.ts` 文件中修改 `API_BASE_URL`:

```typescript
const API_BASE_URL = 'http://localhost:8000/api'
```

## 功能特性

- ✅ 用户登录认证
- ✅ 会话管理（创建、切换）
- ✅ 智能对话（RAG 问答）
- ✅ 显示参考来源
- ✅ 路由守卫保护
- ✅ 响应式布局

## 项目结构

```
rag-frontend/
├── src/
│   ├── api/          # API 接口定义
│   ├── components/   # 公共组件
│   ├── router/       # 路由配置
│   ├── stores/       # Pinia 状态管理
│   ├── utils/        # 工具函数
│   ├── views/        # 页面视图
│   ├── App.vue       # 根组件
│   └── main.ts       # 入口文件
├── package.json
└── vite.config.ts
```

## 注意事项

1. 确保后端服务已启动（默认端口 8000）
2. 首次使用需要注册或登录账号
3. 支持多会话管理，每个会话独立保存历史记录
