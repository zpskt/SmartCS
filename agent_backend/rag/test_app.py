"""
单元测试文件
测试 RAG 系统的各项功能
"""
import unittest
from app import get_enterprise_rag_system
from models.schemas import ChatRequest


class TestEnterpriseRAGSystem(unittest.TestCase):
    """企业 RAG 系统测试类"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.rag_system = get_enterprise_rag_system()
        self.test_user_id = "test_user"
        self.test_session_id = None
    
    def tearDown(self):
        """测试后的清理工作"""
        # 清理测试会话
        if self.test_session_id:
            self.rag_system.delete_session(self.test_session_id)
    
    def test_01_login(self):
        """测试用户登录"""
        result = self.rag_system.login("admin", "admin123")
        self.assertTrue(result["success"])
        self.assertEqual(result["username"], "admin")
        print("✓ 登录测试通过")
    
    def test_02_add_knowledge(self):
        """测试添加知识"""
        result = self.rag_system.add_knowledge(
            title="测试文档",
            content="这是一份测试文档，用于验证知识库功能。",
            source_type="manual",
            created_by=self.test_user_id
        )
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["doc_id"])
        print("✓ 添加知识测试通过")
    
    def test_03_create_session(self):
        """测试创建会话"""
        result = self.rag_system.create_session(
            user_id=self.test_user_id,
            title="测试会话"
        )
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["session_id"])
        self.test_session_id = result["session_id"]
        print("✓ 创建会话测试通过")
    
    def test_04_chat(self):
        """测试对话功能"""
        # 先创建会话
        session_result = self.rag_system.create_session(
            user_id=self.test_user_id,
            title="对话测试会话"
        )
        session_id = session_result["session_id"]
        
        try:
            # 发送消息
            request = ChatRequest(
                session_id=session_id,
                message="你好"
            )
            response = self.rag_system.chat(request)
            
            self.assertIsNotNone(response.content)
            self.assertEqual(response.session_id, session_id)
            print("✓ 对话测试通过")
        finally:
            # 清理测试会话
            self.rag_system.delete_session(session_id)
    
    def test_05_search_knowledge(self):
        """测试搜索知识"""
        # 先添加一些知识
        self.rag_system.add_knowledge(
            title="搜索测试文档",
            content="这个文档包含搜索相关的关键词，用于测试搜索功能。",
            source_type="manual",
            created_by=self.test_user_id
        )
        
        # 执行搜索
        results = self.rag_system.search_knowledge(
            query="搜索",
            limit=5
        )
        
        self.assertIsInstance(results, list)
        print(f"✓ 搜索测试通过，找到 {len(results)} 条结果")
    
    def test_06_export_session(self):
        """测试导出会话"""
        # 创建会话并添加一些消息
        session_result = self.rag_system.create_session(
            user_id=self.test_user_id,
            title="导出测试会话"
        )
        session_id = session_result["session_id"]
        
        try:
            # 导出 JSON 格式
            export_result = self.rag_system.export_session(
                session_id=session_id,
                format="json"
            )
            
            self.assertTrue(export_result["success"])
            self.assertIn("content", export_result)
            print("✓ 导出会话测试通过")
        finally:
            self.rag_system.delete_session(session_id)
    
    def test_07_memory_management(self):
        """测试记忆管理"""
        # 创建会话
        session_result = self.rag_system.create_session(
            user_id=self.test_user_id,
            title="记忆测试会话"
        )
        session_id = session_result["session_id"]
        
        try:
            # 添加长期记忆
            memory_result = self.rag_system.add_long_term_memory(
                session_id=session_id,
                content="测试记忆内容",
                category="test",
                importance=0.9
            )
            self.assertTrue(memory_result["success"])
            
            # 搜索记忆
            memories = self.rag_system.search_memories(
                session_id=session_id,
                query="测试"
            )
            self.assertIsInstance(memories, list)
            print(f"✓ 记忆管理测试通过，找到 {len(memories)} 条记忆")
        finally:
            self.rag_system.delete_session(session_id)
    
    def test_08_permission_check(self):
        """测试权限检查"""
        # admin 用户应该有 write 权限
        has_write = self.rag_system.check_permission("admin", "write")
        self.assertTrue(has_write)
        
        # 测试不存在的用户应该没有权限
        has_write_fake = self.rag_system.check_permission("fake_user", "write")
        self.assertFalse(has_write_fake)
        
        print("✓ 权限检查测试通过")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEnterpriseRAGSystem)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print(f"测试完成！")
    print(f"成功：{result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败：{len(result.failures)}")
    print(f"错误：{len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
