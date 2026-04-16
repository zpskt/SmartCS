# 意图识别与表单交互功能指南

## 功能概述

实现了**意图识别 + 表单交互**的智能化交互模式，当用户输入包含特定工具意图时，系统会自动弹出对应表单，用户填写后执行工具。

### 传统方式 vs 新方式

**传统方式（LLM自动解析参数）：**
```
用户: "帮我生成一个门体配置的SQL，型号M001，成品编码PC001..."
→ LLM尝试从文本中提取参数 → 可能提取错误 → 调用工具
```

**新方式（表单交互）：**
```
用户: "我要生成门体配置SQL"
→ 意图识别 → 弹出表单 → 用户填写 → 确认执行 → 返回结果
```

## 架构设计

### 后端组件

#### 1. 工具元数据管理 (`services/tool_metadata.py`)
定义每个工具的表单Schema：
```python
TOOL_METADATA_REGISTRY = {
    "generate_insert_sql": {
        "tool_name": "generate_insert_sql",
        "display_name": "生成门体配置SQL",
        "description": "根据设备信息生成INSERT SQL语句",
        "fields": [
            {
                "name": "device_model",
                "label": "设备型号",
                "type": "text",
                "required": True,
                "placeholder": "例如: M001"
            },
            // ...更多字段
        ]
    }
}
```

#### 2. 意图识别服务 (`services/intent_recognizer.py`)
基于关键词匹配识别用户意图：
```python
recognizer = IntentRecognizer()
result = recognizer.recognize("我要生成门体配置SQL")
# 返回: {
#   "intent_type": "tool_form",
#   "tool_name": "generate_insert_sql",
#   "form_schema": {...},
#   "confidence": 0.85
# }
```

#### 3. API接口 (`api_server.py`)

**意图识别接口：**
```http
POST /api/intent/recognize
{
  "message": "我要生成门体配置SQL"
}

Response:
{
  "intent_type": "tool_form",
  "tool_name": "generate_insert_sql",
  "form_schema": {...},
  "confidence": 0.85,
  "message": "检测到您需要「生成门体配置SQL」，请填写以下信息："
}
```

**工具执行接口：**
```http
POST /api/tools/execute
{
  "tool_name": "generate_insert_sql",
  "parameters": {
    "device_model": "M001",
    "product_code": "PC001",
    "is_special": false,
    "status_role": "门体1 door1status",
    "alarm_role": "门体1 door1alarm"
  }
}

Response:
{
  "success": true,
  "tool_name": "generate_insert_sql",
  "result": "生成的 SQL 语句:\n\n```sql\ninsert into door...\n```"
}
```

### 前端组件

#### 1. API封装 (`src/api/index.ts`)
```typescript
export const intentApi = {
  recognizeIntent(data: IntentRecognitionRequest): Promise<IntentRecognitionResponse>
}

export const toolApi = {
  executeTool(data: ExecuteToolRequest): Promise<ExecuteToolResponse>
}
```

#### 2. 聊天视图集成 (`src/views/ChatView.vue`)

**流程：**
1. 用户输入消息
2. 调用意图识别API
3. 如果识别为`tool_form`，显示表单弹窗
4. 用户填写表单并提交
5. 调用工具执行API
6. 在聊天中显示结果

## 如何添加新工具

### 步骤1: 注册工具元数据

编辑 `services/tool_metadata.py`：

```python
TOOL_METADATA_REGISTRY = {
    # ...现有工具
    
    "create_branch": {
        "tool_name": "create_branch",
        "display_name": "创建代码分支",
        "description": "在代码仓库中创建新的分支",
        "category": "git",
        "icon": "🌿",
        "fields": [
            {
                "name": "branch_name",
                "label": "分支名称",
                "type": "text",
                "placeholder": "例如: feature/new-feature",
                "required": True,
                "help_text": "请输入分支名称"
            },
            {
                "name": "base_branch",
                "label": "基于哪个分支",
                "type": "text",
                "placeholder": "例如: main",
                "required": True,
                "default": "main"
            }
        ]
    }
}
```

### 步骤2: 添加工具函数

在对应的工具文件中实现工具函数（如已有则跳过）：

```python
@tool
def create_branch(branch_name: str, base_branch: str) -> str:
    """创建代码分支"""
    # 实现逻辑
    return f"已创建分支 {branch_name} 基于 {base_branch}"
```

### 步骤3: 注册到工具映射表

编辑 `api_server.py` 中的 `_execute_tool_by_name` 函数：

```python
async def _execute_tool_by_name(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    from tools.model_adapter import generate_insert_sql
    from tools.git_tools import create_branch  # 导入新工具
    
    tool_functions = {
        "generate_insert_sql": generate_insert_sql,
        "create_branch": create_branch,  # 添加到映射表
    }
    
    # ...其余逻辑
```

### 步骤4: 添加关键词映射

编辑 `services/intent_recognizer.py`：

```python
self.keyword_mapping = {
    # ...现有映射
    
    "创建分支": "create_branch",
    "新建分支": "create_branch",
    "branch": "create_branch",
}
```

完成！现在用户输入"我要创建分支"时，会自动弹出分支创建表单。

## 支持的表单字段类型

- `text`: 单行文本输入
- `textarea`: 多行文本输入
- `select`: 下拉选择
- `number`: 数字输入
- `boolean`: 布尔值（可通过select实现）

## 意图识别优化建议

### 1. 增加关键词
在 `IntentRecognizer.__init__` 中添加更多关键词映射：
```python
self.keyword_mapping = {
    "生成sql": "generate_insert_sql",
    "创建sql": "generate_insert_sql",
    "门体配置": "generate_insert_sql",
    # 添加同义词、近义词
}
```

### 2. 调整置信度阈值
修改 `recognize` 方法中的阈值：
```python
if best_match and best_match["confidence"] > 0.5:  # 调整此值
```

### 3. 使用机器学习模型（进阶）
可以替换为基于BERT等模型的意图分类器，提高识别准确率。

## 测试示例

### 测试意图识别
```bash
curl -X POST http://localhost:8000/api/intent/recognize \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"message": "我要生成门体配置SQL"}'
```

### 测试工具执行
```bash
curl -X POST http://localhost:8000/api/tools/execute \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "tool_name": "generate_insert_sql",
    "parameters": {
      "device_model": "M001",
      "product_code": "PC001",
      "is_special": false,
      "status_role": "门体1 door1status",
      "alarm_role": "门体1 door1alarm"
    }
  }'
```

## 优势

✅ **准确性高**: 用户手动填写，避免LLM解析错误  
✅ **交互友好**: 表单引导用户输入，降低学习成本  
✅ **易于扩展**: 新增工具只需配置元数据  
✅ **灵活降级**: 识别失败时自动降级为普通聊天  

## 注意事项

1. 工具元数据中的字段名必须与工具函数的参数名一致
2. 必填字段验证在前端和后端都应该进行
3. 意图识别的关键词需要根据实际使用情况不断优化
4. 对于复杂工具，可以考虑分步表单或向导模式
