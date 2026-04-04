#!/bin/bash

# RAG 前端启动脚本

echo "======================================"
echo "企业知识库 RAG 系统 - 前端服务"
echo "======================================"
echo ""

# 检查依赖
if [ ! -d "node_modules" ]; then
    echo "📦 首次运行，正在安装依赖..."
    npm install
fi

# 启动开发服务器
echo "🚀 正在启动前端开发服务器..."
echo ""
npm run dev
