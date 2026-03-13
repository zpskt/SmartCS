import streamlit as st
from knowledge_base import KnowledgeBaseService
# 网页标题
st.title("配置知识库更新服务")
uploader_file = st.file_uploader("上传文件",
                                 type=['txt', 'md','.docx', '.doc'],
                                 accept_multiple_files= False
                                 )
if "service" not in st.session_state:
    st.session_state.service = KnowledgeBaseService()

if uploader_file is not None:
    # 提取文件信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size
    st.subheader(f"文件名：{file_name}")
    st.subheader(f"文件类型：{file_type}")
    st.subheader(f"文件大小：{file_size:.2f}KB")

    text = uploader_file.getvalue().decode("utf-8")
    with st.spinner(f"正在处理文件{file_name}..."):
        result = st.session_state["service"].upload_by_str(text, file_name)
        st.write(result)
