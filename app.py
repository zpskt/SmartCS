"""
智能客服系统主应用入口
"""
from customer_service.dialog_management.dialog_manager import DialogManager
from customer_service.intent_recognition.intent_recognition import IntentRecognizer
from customer_service.sentiment_analysis.sentiment_analysis import SentimentAnalyzer
from customer_service.knowledge_base.knowledge_base import KnowledgeBase

def main():
    """
    主函数 - 初始化并运行智能客服系统
    """
    print("正在启动智能客服系统...")
    
    # 初始化各模块
    knowledge_base = KnowledgeBase()
    intent_recognizer = IntentRecognizer()
    sentiment_analyzer = SentimentAnalyzer()
    dialog_manager = DialogManager()
    
    print("智能客服系统启动成功！")
    print("支持的功能：")
    print("- 多轮对话管理")
    print("- 意图识别")
    print("- 情感分析")
    print("- 知识库查询")
    print("\n输入 'quit' 或 'exit' 退出系统")
    
    # 简单的交互循环示例
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ['quit', 'exit']:
            print("智能客服系统已退出。")
            break
            
        # 这里应该是完整的处理流程
        # 1. 意图识别
        intent = intent_recognizer.recognize(user_input)
        
        # 2. 情感分析
        sentiment = sentiment_analyzer.analyze(user_input)
        
        # 3. 查询知识库
        knowledge = knowledge_base.query(user_input)
        
        # 4. 对话管理
        response = dialog_manager.generate_response(user_input, intent, sentiment, knowledge)
        
        print(f"客服: {response}")

if __name__ == "__main__":
    main()