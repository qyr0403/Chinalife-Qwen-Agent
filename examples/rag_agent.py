import os
from typing import List, Dict

from qwen_agent.agents import Assistant

from qwen_agent.gui import WebUI


def get_files_from_folder(folder_path: str) -> List [str]:
    supporter_extensions = ['.pdf', '.docx', '.pptx', '.txt', '.csv', '.tsv', '.xlsx', '.xls', '.html']
    files = []

    # 转换为绝对路径
    folder_path = os.path.abspath(folder_path)

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder {folder_path} does not exist.")
    
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext.lower() in supporter_extensions:
                file_path = os.path.join(root, filename)
                file_path = os.path.abspath(file_path)
                files.append(file_path)
    return files

def create_folder_rag_agent(folder_path: str):
    files = get_files_from_folder(folder_path)
    if not files:
        raise ValueError(f"No supported files found in folder {folder_path}.")

    llm_cfg = {
        # Use the model service provided by DashScope:
        # 'model': 'qwen-max-latest',
        # 'model_type': 'qwen_dashscope',
        # 'api_key': 'YOUR_DASHSCOPE_API_KEY',
        # It will use the `DASHSCOPE_API_KEY' environment variable if 'api_key' is not set here.

        # Use a model service compatible with the OpenAI API, such as vLLM or Ollama:
        'model': 'deepseek-chat',
        'model_server': 'https://api.deepseek.com',  # base_url, also known as api_base
        'api_key': 'sk-3c0a4836554f4261b7b964de3c5d02e3',

        # (Optional) LLM hyperparameters for generation:
        'generate_cfg': {
            'top_p': 0.8
        }
    }
    bot = Assistant(
        llm=llm_cfg,
        name='RAG智能助手',
        system_message='你是一个能够基于文件内容回答问题的助手。请根据提供的文件内容来回答用户问题，在根据文件内容回答问题时标注信息来源的文档和位置。当用户有涉及数据分析和表示的需求时，你可以根据用户的要求和参考信息中的有关数据使用Python语言和matplotlib库生成代码，并执行它来进行呈现。',
        function_list=[
            {
                'name': 'retrieval',
            },
            'multimodal_doc_parser', 'code_interpreter'
        ],
        files=files,
    )
    return bot

def test_folder_rag_agent():
    folder_path = "C:\\Qwen-Agent\\documents" # 可以根据需要修改为其他文件夹路径
    try:
        bot = create_folder_rag_agent(folder_path)

        messages = [{
            'role': 'user',
            'content': '帮我分析下量子计算机发展的路线，并画一张折线图。'
        }]
        
        response = []
        for response in bot.run(messages):
            # import pdb; pdb.set_trace()
            print('bot response:', response)
    except FileNotFoundError as e:
        print(f"Error: {e}")

def app_gui():
    folder_path = "C:\\Qwen-Agent\\documents"  # 可以根据需要修改为其他文件夹路径
    try:
        bot = create_folder_rag_agent(folder_path)

        chatbot_config = {
            'prompt.suggestions': [
                {'text': '请介绍文档的主要内容'},
                {'text': '从文档中提取关键信息'}
            ]
        }
        WebUI(bot, chatbot_config=chatbot_config).run()
    except FileNotFoundError as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    # test_folder_rag_agent()
    app_gui()



