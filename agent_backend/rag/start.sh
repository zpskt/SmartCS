#!/bin/bash

# 企业知识库 RAG 系统 - 快速启动脚本

echo "======================================"
echo "企业知识库 RAG 系统"
echo "======================================"
echo ""

# 检查 Python 环境
echo "🔍 检查 Python 环境..."
python_version=$(python3 --version 2>&1)
if [ $? -eq 0 ]; then
    echo "✓ Python 环境：$python_version"
else
    echo "✗ 错误：未找到 Python3，请先安装 Python3"
    exit 1
fi

# 检查虚拟环境
if [ -d "venv" ]; then
    echo "✓ 检测到虚拟环境"
    source venv/bin/activate
else
    echo "ℹ️  未检测到虚拟环境，使用全局 Python 环境"
fi

# 检查依赖
echo ""
echo "📦 检查依赖包..."
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "✓ 依赖包安装/更新完成"
    else
        echo "✗ 警告：依赖包安装失败，请手动执行：pip install -r requirements.txt"
    fi
else
    echo "✗ 错误：未找到 requirements.txt 文件"
    exit 1
fi

# 检查环境配置
echo ""
echo "⚙️  检查环境配置..."
if [ -f ".env" ]; then
    echo "✓ 检测到 .env 配置文件"
elif [ -f ".env.example" ]; then
    echo "⚠️  未检测到 .env 文件，将使用默认配置"
    echo "💡 建议复制 .env.example 为 .env 并修改配置"
else
    echo "⚠️  未检测到环境配置文件"
fi

# 创建数据目录
echo ""
echo "📁 创建数据目录..."
mkdir -p data/chroma_db
mkdir -p data/history
echo "✓ 数据目录已创建"

# 显示菜单
echo ""
echo "======================================"
echo "请选择启动方式："
echo "======================================"
echo "1. 运行示例脚本 (演示完整功能)"
echo "2. 运行测试 (单元测试)"
echo "3. 启动 API 服务器 (FastAPI)"
echo "4. Python 交互模式"
echo "0. 退出"
echo ""

read -p "请输入选项 [0-4]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 运行示例脚本..."
        python example_usage.py
        ;;
    2)
        echo ""
        echo "🧪 运行单元测试..."
        python test_app.py
        ;;
    3)
        echo ""
        echo "🌐 启动 API 服务器..."
        echo "访问地址：http://localhost:8000"
        echo "API 文档：http://localhost:8000/docs"
        echo ""
        python api_server.py
        ;;
    4)
        echo ""
        echo "💻 进入 Python 交互模式..."
        python -c "from rag.app import get_enterprise_rag_system; rag = get_enterprise_rag_system(); print('RAG 系统已加载，可以使用 rag 变量访问系统功能')"
        ;;
    0)
        echo "👋 退出"
        exit 0
        ;;
    *)
        echo "✗ 无效的选项，请输入 0-4"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "操作完成"
echo "======================================"
