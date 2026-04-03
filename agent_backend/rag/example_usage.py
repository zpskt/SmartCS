"""
企业知识库 RAG 系统 - 使用示例
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent  # 改为当前目录，即 rag 目录
sys.path.insert(0, str(project_root))

from app import get_enterprise_rag_system
from models.schemas import ChatRequest


def main():
    """主函数 - 演示完整的使用流程"""
    
    # 获取系统实例
    rag_system = get_enterprise_rag_system()
    
    print("=" * 60)
    print("企业知识库 RAG 系统 - 功能演示")
    print("=" * 60)
    
    # ========== 1. 用户登录 ==========
    print("\n【1】用户登录")
    login_result = rag_system.login("admin", "admin123")
    if login_result["success"]:
        user_id = login_result["user_id"]
        print(f"✓ 登录成功！用户：{login_result['username']}, 角色：{login_result['role']}")
    else:
        print(f"✗ 登录失败：{login_result['message']}")
        return
    
    # ========== 2. 添加知识到知识库 ==========
    print("\n【2】添加产品知识到知识库")
    knowledge_result = rag_system.add_knowledge(
        title="扫地机器人 X1 产品说明",
        content="""
        X1 是我们公司最新推出的智能扫地机器人，具有以下特点：
        
        1. 智能导航：采用激光雷达导航，能够精准建图和路径规划
        2. 强大吸力：5000Pa 大吸力，能够有效清除灰尘和毛发
        3. 长续航：5200mAh 电池，续航时间可达 180 分钟
        4. 自动回充：电量低于 20% 时自动返回充电座
        5. APP 控制：支持手机 APP 远程控制和定时清扫
        6. 静音设计：工作噪音低于 55 分贝
        
        适用场景：
        - 适合 80-200 平米的户型
        - 适用于木地板、瓷砖、短毛地毯等地面
        - 可以清扫床底、沙发底等低矮空间
        
        维护保养：
        - 每次使用后清理尘盒
        - 每周清洗滤网
        - 每月清理主刷和边刷
        - 定期擦拭传感器
        """,
        source_type="manual",
        created_by=user_id,
        source_url=None
    )
    
    if knowledge_result["success"]:
        print(f"✓ 知识添加成功！文档 ID: {knowledge_result['doc_id']}")
    else:
        print(f"✗ 知识添加失败：{knowledge_result['message']}")
    
    # ========== 3. 创建会话 ==========
    print("\n【3】创建对话会话")
    session_result = rag_system.create_session(
        user_id=user_id,
        title="产品咨询会话"
    )
    if session_result["success"]:
        session_id = session_result["session_id"]
        print(f"✓ 会话创建成功！会话 ID: {session_id}")
    else:
        print(f"✗ 会话创建失败：{session_result.get('message', '未知错误')}")
        return
    
    # ========== 4. 进行对话 ==========
    print("\n【4】开始对话")
    
    # 第一轮对话
    print("\n用户提问：X1 扫地机器人的续航时间是多少？")
    request1 = ChatRequest(
        session_id=session_id,
        message="X1 扫地机器人的续航时间是多少？"
    )
    response1 = rag_system.chat(request1)
    print(f"AI 回答：{response1.content}")
    if response1.sources:
        print(f"参考来源：{response1.sources[0]['title']}")
    
    # 第二轮对话（带上下文）
    print("\n用户提问：它适合多大的房子？")
    request2 = ChatRequest(
        session_id=session_id,
        message="它适合多大的房子？"
    )
    response2 = rag_system.chat(request2)
    print(f"AI 回答：{response2.content}")
    if response2.sources:
        print(f"参考来源：{response2.sources[0]['title']}")
    
    # 第三轮对话（继续上下文）
    print("\n用户提问：需要怎么保养？")
    request3 = ChatRequest(
        session_id=session_id,
        message="需要怎么保养？"
    )
    response3 = rag_system.chat(request3)
    print(f"AI 回答：{response3.content}")
    if response3.sources:
        print(f"参考来源：{response3.sources[0]['title']}")
    
    # ========== 5. 搜索知识 ==========
    print("\n【5】搜索相关知识")
    search_results = rag_system.search_knowledge(
        query="扫地机器人 续航 电池",
        limit=3
    )
    print(f"找到 {len(search_results)} 条相关知识:")
    for i, result in enumerate(search_results, 1):
        print(f"{i}. {result.get('metadata', {}).get('title', '无标题')}")
        print(f"   内容片段：{result['content'][:100]}...")
    
    # ========== 6. 获取会话列表 ==========
    print("\n【6】获取用户的会话列表")
    session_list = rag_system.get_session_list(user_id)
    print(f"用户共有 {len(session_list)} 个会话:")
    for session in session_list:
        print(f"- {session['title']} (创建于：{session['created_at']})")
    
    # ========== 7. 导出会话 ==========
    print("\n【7】导出会话记录")
    export_result = rag_system.export_session(
        session_id=session_id,
        format="json"
    )
    if export_result["success"]:
        print("✓ 会话导出成功!")
        print(f"导出格式：{export_result['format']}")
        print(f"导出内容预览：{export_result['content'][:200]}...")
    
    # ========== 8. 添加长期记忆 ==========
    print("\n【8】保存重要信息到长期记忆")
    memory_result = rag_system.add_long_term_memory(
        session_id=session_id,
        content="用户对 X1 扫地机器人的续航时间和适用面积比较关心",
        category="user_preference",
        importance=0.8
    )
    if memory_result["success"]:
        print("✓ 记忆保存成功!")
    
    # ========== 9. 搜索记忆 ==========
    print("\n【9】搜索历史记忆")
    memories = rag_system.search_memories(
        session_id=session_id,
        query="续航"
    )
    print(f"找到 {len(memories)} 条相关记忆:")
    for memory in memories:
        print(f"- {memory['content']} (重要度：{memory['importance']})")
    
    # ========== 10. 清空会话 ==========
    print("\n【10】清空会话消息")
    clear_result = rag_system.clear_session_messages(session_id)
    if clear_result["success"]:
        print("✓ 会话消息已清空")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
