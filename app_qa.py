import time
from core.rag import RagService
from core.knowledge_base import KnowledgeBaseService
import streamlit as st
import config_data as config
import pdfplumber


# 设置页面配置
st.set_page_config(page_title="智能客服系统", layout="wide")

# 创建两个 tab 页
tab1, tab2 = st.tabs(["📁 知识库管理", "💬 智能对话"])

# Tab 1: 文件上传
with tab1:
    st.title("知识库管理")
    uploader_file = st.file_uploader(
        "上传文件到知识库",
        type=['txt', 'md', '.docx', '.doc','.pdf'],
        accept_multiple_files=False
    )
    
    if "kb_service" not in st.session_state:
        st.session_state.kb_service = KnowledgeBaseService()
    
    if uploader_file is not None:
        # 提取文件信息
        file_name = uploader_file.name
        file_type = uploader_file.type
        file_size = uploader_file.size
        st.subheader(f"文件名：{file_name}")
        st.subheader(f"文件类型：{file_type}")
        st.subheader(f"文件大小：{file_size:.2f}KB")

        # 根据文件类型选择处理方式
        text = ""
        if file_name.endswith('.pdf'):
            # 可以使用 PyPDF2 或 pdfplumber 解析 PDF
            with pdfplumber.open(uploader_file) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages])
        elif file_name.endswith(('.doc', '.docx')):
            st.warning("⚠️ Word 文件需要先解析，当前版本暂不支持")
            # 可以使用 python-docx 解析
            from docx import Document
            doc = Document(uploader_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            # txt、md 等文本文件
            text = uploader_file.getvalue().decode("utf-8")

        if text:
            with st.spinner(f"正在处理文件{file_name}..."):
                result = st.session_state.kb_service.upload_by_str(text, file_name)
                st.success(result)

        with st.spinner(f"正在处理文件{file_name}..."):
            result = st.session_state.kb_service.upload_by_str(text, file_name)
            st.success(result)

# Tab 2: 智能对话
with tab2:
    st.title("智能客服")
    st.divider()
    
    if "message" not in st.session_state:
        st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]
    
    if "rag" not in st.session_state:
        st.session_state["rag"] = RagService()
    
    for message in st.session_state["message"]:
        st.chat_message(message["role"]).write(message["content"])
    
    # 在页面最下方提供用户输入栏
    prompt = st.chat_input()
    
    if prompt:
        # 在页面输出用户的提问
        st.chat_message("user").write(prompt)
        st.session_state["message"].append({"role": "user", "content": prompt})
        
        ai_res_list = []
        with st.spinner("AI 思考中..."):
            res_stream = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)
            
            def capture(generator, cache_list):
                for chunk in generator:
                    cache_list.append(chunk)
                    yield chunk
            
            st.chat_message("assistant").write_stream(capture(res_stream, ai_res_list))
            st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})

# ["a", "b", "c"]   "".join(list)    -> abc
# ["a", "b", "c"]   ",".join(list)    -> a,b,c