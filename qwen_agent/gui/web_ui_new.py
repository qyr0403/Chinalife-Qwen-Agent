# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import os
import base64
import os
import subprocess
import sys
import time
import uuid
import pprint
import re
from typing import List, Optional, Union

from qwen_agent import Agent, MultiAgentHub
from qwen_agent.agents.user_agent import PENDING_USER_INPUT
from qwen_agent.gui.gradio_utils import format_cover_html
from qwen_agent.gui.utils import convert_fncall_to_text, convert_history_to_chatbot, get_avatar_image
from qwen_agent.llm.schema import AUDIO, CONTENT, FILE, IMAGE, NAME, ROLE, USER, VIDEO, Message
from qwen_agent.log import logger
from qwen_agent.utils.utils import print_traceback

import base64
import os
import uuid

import gradio as gr
import modelscope_studio.components.antd as antd
import modelscope_studio.components.antdx as antdx
import modelscope_studio.components.base as ms
import modelscope_studio.components.pro as pro
from modelscope_studio.components.pro.chatbot import (ChatbotActionConfig,
                                                      ChatbotBotConfig,
                                                      ChatbotUserConfig,
                                                      ChatbotWelcomeConfig)
from modelscope_studio.components.pro.multimodal_input import \
  MultimodalInputUploadConfig
from openai import OpenAI

from qwen_agent.agents import Assistant

# =========== Configuration

# API KEY
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

folder_path = r"D:\PROJECT\Chinalife-Qwen-Agent\examples\resource"  # 可以根据需要修改为其他文件夹路径
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
    # 'model': 'kimi-k2-0711-preview',
    # 'model_server': 'https://api.moonshot.cn/v1',  # base_url, also known as api_base
    # 'api_key': 'sk-HwfDyinhIU1l4lBlcNTNq87jOESwyfTGNBz7Ysp1qakcKOWw',
    'model': 'deepseek-chat',
    'model_server': 'https://api.deepseek.com',  # base_url, also known as api_base
    'api_key': 'sk-3c0a4836554f4261b7b964de3c5d02e3',

    # (Optional) LLM hyperparameters for generation:
    'generate_cfg': {
        'top_p': 0.8
    },
}
bot = Assistant(
    llm=llm_cfg,
    name='RAG智能助手',
    system_message='你是一个能够基于文件内容回答问题的助手。请根据提供的文件内容来回答用户问题，在根据文件内容回答问题时标注信息来源的文档和位置。当用户有涉及数据分析和表示的需求时，你可以根据用户的要求和参考信息中的有关数据使用Python语言和matplotlib库生成代码，并执行它来进行呈现。',
    function_list=[
            {
                'name': 'retrieval',
            },
            'code_interpreter'
        ],
    files=files,
)

save_history = False

# =========== Configuration

DEFAULT_PROMPTS = [{
    "label":
    "📅 帮我分析",
    "children": [{
        "description":
        "请讲解一下格力公司最近的增长点"
    }, {
        "description":
        "请解读本季度财报"
    }, {
        "description":
        "帮我分析下量子计算机发展的路线，并画一张折线图。"
    }]
}, {
    "label":
    "🖋 帮我可视化",
    "children": [{
        "description":
        "帮我可视化附件文档数据"
    }, {
        "description":
        "帮我可视化盈利波动"
    }, {
        "description":
        "帮我可视化数据"
    }]
}]

DEFAULT_SUGGESTIONS = [{
    "label":
    "📅 帮我分析",
    "value":
    "📅 帮我分析",
    "children": [{
        "label": "讲解格力公司最近的增长点",
        "value": "请讲解一下格力公司最近的增长点"
    }, {
        "label": "解读本季度财报",
        "value": "请解读本季度财报"
    }, {
        "label": "分析量子计算机发展路线",
        "value": "帮我分析下量子计算机发展的路线，并画一张折线图。"
    }]
}, {
    "label":
    "🖋 帮我可视化",
    "value":
    "🖋 帮我可视化",
    "children": [{
        "label": "可视化附件文档数据",
        "value": "帮我可视化附件文档数据"
    }, {
        "label": "可视化盈利波动",
        "value": "帮我可视化盈利波动"
    }, {
        "label": "可视化数据",
        "value": "帮我可视化数据"
    }]
}]

DEFAULT_LOCALE = 'en_US'

DEFAULT_THEME = {
    "token": {
        "colorPrimary": "#6A57FF",
    }
}


def user_config(disabled_actions=None):
    return ChatbotUserConfig(actions=[
        "copy", "edit",
        ChatbotActionConfig(
            action="delete",
            popconfirm=dict(title="删除",
                            description="你确定要删除吗?",
                            okButtonProps=dict(danger=True)))
    ],
                             disabled_actions=disabled_actions)


def bot_config(disabled_actions=None):
    return ChatbotBotConfig(
        actions=[
            "copy", "like", "dislike", "edit",
            ChatbotActionConfig(
                action="retry",
                popconfirm=dict(
                    title="重新生成",
                    description=
                    "重新生成将删除原有的后续信息",
                    okButtonProps=dict(danger=True))),
            ChatbotActionConfig(action="delete",
                                popconfirm=dict(
                                    title="删除",
                                    description=
                                    "你确定要删除吗?",
                                    okButtonProps=dict(danger=True)))
        ],
        avatar=
        r"D:\PROJECT\Chinalife-Qwen-Agent\qwen_agent\gui\assets\fmt.png",
        disabled_actions=disabled_actions)


class Gradio_Events:

    @staticmethod
    def submit(state_value):
        # Define your code here
        # The best way is to use the image url.
        def image_to_base64(image_path):
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(
                    image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"

        def format_history(history):
            # messages = [{
            #     "role": "system",
            #     "content": "You are a helpful and harmless assistant.",
            # }]
            messages=[]
            for item in history:
                if item["role"] == "user":
                    content=[{
                            'text': item["content"][1]["content"]
                        }]
                    for file in item["content"][0]["content"]:
                        if os.path.exists(file):
                            content.append({
                                'file': file
                            })
                    messages.append({
                        "role":
                        "user",
                        "content": content.copy(),
                    })
                elif item["role"] == "assistant":
                    messages.append({
                        "role": "assistant",
                        "content": item["content"]
                    })
            return messages

        history = state_value["conversations_history"][
            state_value["conversation_id"]]
        history_messages = format_history(history)

        history.append({
            "role": "assistant",
            "content": "",
            "loading": True,
            "status": "pending"
        })

        yield {
            chatbot: gr.update(value=history),
            state: gr.update(value=state_value),
        }
        try:
            responses=[]
            for responses in bot.run(history_messages):
                if not responses:
                    continue
                if responses[-1][CONTENT] == PENDING_USER_INPUT:
                    logger.info('Interrupted. Waiting for user input!')
                    break
                display_responses = convert_fncall_to_text(responses)
                if not display_responses:
                    continue
                if display_responses[-1][CONTENT] is None:
                    continue
                for chunk in display_responses:
                    if chunk[CONTENT]!=None:
                        history[-1]["content"] = chunk[CONTENT]
                        history[-1]["loading"] = False
                        yield {
                            chatbot: gr.update(value=history),
                            state: gr.update(value=state_value)
                        }
            history[-1]["status"] = "done"
            yield {
                chatbot: gr.update(value=history),
                state: gr.update(value=state_value),
            }
        except Exception as e:
            history[-1]["loading"] = False
            history[-1]["status"] = "done"
            history[-1]["content"] = "Failed to respond, please try again."
            yield {
                chatbot: gr.update(value=history),
                state: gr.update(value=state_value)
            }
            raise e

    @staticmethod
    def add_user_message(input_value, state_value):
        if not state_value["conversation_id"]:
            random_id = str(uuid.uuid4())
            history = []
            state_value["conversation_id"] = random_id
            state_value["conversations_history"][random_id] = history
            state_value["conversations"].append({
                "label": input_value["text"],
                "key": random_id
            })

        history = state_value["conversations_history"][
            state_value["conversation_id"]]
        # print(input_value,state_value)
        history.append({
            "role":
            "user",
            "content": [{
                "type": "file",
                "content": [f for f in input_value["files"]]
            }, {
                "type": "text",
                "content": input_value["text"]
            }]
        })
        '''
        {'text': '解释下', 
        'files': ['C:\\Users\\guihaoyue\\AppData\\Local\\Temp\\gradio\\bf9d9427507315fd5238f87e774db6975a00da91eeafa5c7ebfad59926a0f139\\2.大模型预训练 与主流大模型结构.pdf']
        }
        {'conversations_history': 
        {'09b287a2-ffc2-45cf-ae28-f307cf8955ec': 
        [{'role': 'user', 'content': [{'type': 'file', 'content': []}, {'type': 'text', 'content': '你好'}]}, 
        {'role': 'assistant', 'content': '你好！很高兴见到你，有什么我可以帮忙的吗？', 'loading': False, 'status': 'done'}]}, 
        'conversations': [{'label': '你好', 'key': '09b287a2-ffc2-45cf-ae28-f307cf8955ec'}], 
        'conversation_id': '09b287a2-ffc2-45cf-ae28-f307cf8955ec'}
        '''
        return gr.update(value=state_value)

    @staticmethod
    def preprocess_submit(clear_input=True):

        def preprocess_submit_handler(state_value):
            history = state_value["conversations_history"][
                state_value["conversation_id"]]
            return {
                **({
                    input:
                    gr.update(value=None, loading=True) if clear_input else gr.update(loading=True),
                } if clear_input else {}),
                conversations:
                gr.update(active_key=state_value["conversation_id"],
                          items=list(
                              map(
                                  lambda item: {
                                      **item,
                                      "disabled":
                                      True if item["key"] != state_value[
                                          "conversation_id"] else False,
                                  }, state_value["conversations"]))),
                add_conversation_btn:
                gr.update(disabled=True),
                clear_btn:
                gr.update(disabled=True),
                conversation_delete_menu_item:
                gr.update(disabled=True),
                chatbot:
                gr.update(value=history,
                          bot_config=bot_config(
                              disabled_actions=['edit', 'retry', 'delete']),
                          user_config=user_config(
                              disabled_actions=['edit', 'delete'])),
                state:
                gr.update(value=state_value),
            }

        return preprocess_submit_handler

    @staticmethod
    def postprocess_submit(state_value):
        history = state_value["conversations_history"][
            state_value["conversation_id"]]
        return {
            input:
            gr.update(loading=False),
            conversation_delete_menu_item:
            gr.update(disabled=False),
            clear_btn:
            gr.update(disabled=False),
            conversations:
            gr.update(items=state_value["conversations"]),
            add_conversation_btn:
            gr.update(disabled=False),
            chatbot:
            gr.update(value=history,
                      bot_config=bot_config(),
                      user_config=user_config()),
            state:
            gr.update(value=state_value),
        }

    @staticmethod
    def cancel(state_value):
        history = state_value["conversations_history"][
            state_value["conversation_id"]]
        history[-1]["loading"] = False
        history[-1]["status"] = "done"
        history[-1]["footer"] = "Chat completion paused"
        return Gradio_Events.postprocess_submit(state_value)

    @staticmethod
    def delete_message(state_value, e: gr.EventData):
        index = e._data["payload"][0]["index"]
        history = state_value["conversations_history"][
            state_value["conversation_id"]]
        history = history[:index] + history[index + 1:]

        state_value["conversations_history"][
            state_value["conversation_id"]] = history

        return gr.update(value=state_value)

    @staticmethod
    def edit_message(state_value, chatbot_value, e: gr.EventData):
        index = e._data["payload"][0]["index"]
        history = state_value["conversations_history"][
            state_value["conversation_id"]]
        history[index]["content"] = chatbot_value[index]["content"]
        return gr.update(value=state_value)

    @staticmethod
    def regenerate_message(state_value, e: gr.EventData):
        index = e._data["payload"][0]["index"]
        history = state_value["conversations_history"][
            state_value["conversation_id"]]
        history = history[:index]
        state_value["conversations_history"][
            state_value["conversation_id"]] = history
        # custom code
        return gr.update(value=history), gr.update(value=state_value)

    @staticmethod
    def select_suggestion(input_value, e: gr.EventData):
        input_value["text"] = input_value["text"][:-1] + e._data["payload"][0]
        return gr.update(value=input_value)

    @staticmethod
    def apply_prompt(input_value, e: gr.EventData):
        input_value["text"] = e._data["payload"][0]["value"]["description"]
        return gr.update(value=input_value)

    @staticmethod
    def new_chat(state_value):
        if not state_value["conversation_id"]:
            return gr.skip()
        state_value["conversation_id"] = ""
        return gr.update(active_key=state_value["conversation_id"]), gr.update(
            value=None), gr.update(value=state_value)

    @staticmethod
    def select_conversation(state_value, e: gr.EventData):
        active_key = e._data["payload"][0]
        if state_value["conversation_id"] == active_key or (
                active_key not in state_value["conversations_history"]):
            return gr.skip()
        state_value["conversation_id"] = active_key
        return gr.update(active_key=active_key), gr.update(
            value=state_value["conversations_history"][active_key]), gr.update(
                value=state_value)

    @staticmethod
    def click_conversation_menu(state_value, e: gr.EventData):
        conversation_id = e._data["payload"][0]["key"]
        operation = e._data["payload"][1]["key"]
        if operation == "delete":
            del state_value["conversations_history"][conversation_id]

            state_value["conversations"] = [
                item for item in state_value["conversations"]
                if item["key"] != conversation_id
            ]

            if state_value["conversation_id"] == conversation_id:
                state_value["conversation_id"] = ""
                return gr.update(
                    items=state_value["conversations"],
                    active_key=state_value["conversation_id"]), gr.update(
                        value=None), gr.update(value=state_value)
            else:
                return gr.update(
                    items=state_value["conversations"]), gr.skip(), gr.update(
                        value=state_value)
        return gr.skip()

    @staticmethod
    def clear_conversation_history(state_value):
        if not state_value["conversation_id"]:
            return gr.skip()
        state_value["conversations_history"][
            state_value["conversation_id"]] = []
        return gr.update(value=None), gr.update(value=state_value)

    @staticmethod
    def update_browser_state(state_value):

        return gr.update(value=dict(
            conversations=state_value["conversations"],
            conversations_history=state_value["conversations_history"]))

    @staticmethod
    def apply_browser_state(browser_state_value, state_value):
        state_value["conversations"] = browser_state_value["conversations"]
        state_value["conversations_history"] = browser_state_value[
            "conversations_history"]
        return gr.update(
            items=browser_state_value["conversations"]), gr.update(
                value=state_value)


css = """
#chatbot {
  height: calc(100vh - 32px - 21px - 16px);
}

#chatbot .chatbot-conversations {
  height: 100%;
  background-color: var(--ms-gr-ant-color-bg-layout);
}

#chatbot .chatbot-conversations .chatbot-conversations-list {
  padding-left: 0;
  padding-right: 0;
  flex: 1;
  height: 0;
  overflow: auto;
}

#chatbot .chatbot-chat {
  padding: 32px;
  height: 100%;
}

@media (max-width: 768px) {
  #chatbot .chatbot-chat {
      padding: 0;
  }
}

#chatbot .chatbot-chat .chatbot-chat-messages {
  flex: 1;
}
"""


def logo():
    with antd.Typography.Title(level=1,
                               elem_style=dict(fontSize=24,
                                               padding=8,
                                               margin=0)):
        with antd.Flex(align="center", gap="small", justify="center"):
            antd.Image(
                r"D:\PROJECT\Chinalife-Qwen-Agent\qwen_agent\gui\assets\logo_china_life.jpeg",
                preview=False,
                alt="logo",
                width=24,
                height=24)
            ms.Span("中国人寿")


with gr.Blocks(css=css, fill_width=True) as demo:
    state = gr.State({
        "conversations_history": {},
        "conversations": [],
        "conversation_id": "",
    })

    with ms.Application(), antdx.XProvider(
            theme=DEFAULT_THEME, locale=DEFAULT_LOCALE), ms.AutoLoading():
        with antd.Row(gutter=[20, 20], wrap=False, elem_id="chatbot"):
            # Left Column
            with antd.Col(md=dict(flex="0 0 260px", span=24, order=0),
                          span=0,
                          order=1,
                          elem_style=dict(width=0),
                          elem_classes="chatbot-conversations"):
                with antd.Flex(vertical=True,
                               gap="small",
                               elem_style=dict(height="100%")):
                    # Logo
                    logo()

                    # New Conversation Button
                    with antd.Button(value=None,
                                     color="primary",
                                     variant="filled",
                                     block=True) as add_conversation_btn:
                        ms.Text("新对话")
                        with ms.Slot("icon"):
                            antd.Icon("PlusOutlined")

                    # Conversations List
                    with antdx.Conversations(
                            elem_classes="chatbot-conversations-list",
                    ) as conversations:
                        with ms.Slot('menu.items'):
                            with antd.Menu.Item(
                                    label="Delete", key="delete", danger=True
                            ) as conversation_delete_menu_item:
                                with ms.Slot("icon"):
                                    antd.Icon("DeleteOutlined")
            # Right Column
            with antd.Col(flex=1, elem_style=dict(height="100%")):
                with antd.Flex(vertical=True, elem_classes="chatbot-chat"):
                    # Chatbot
                    chatbot = pro.Chatbot(
                        elem_classes="chatbot-chat-messages",
                        welcome_config=ChatbotWelcomeConfig(
                            variant="borderless",
                            icon=
                            r"D:\PROJECT\Chinalife-Qwen-Agent\qwen_agent\gui\assets\fmt.png",
                            title=f"你好，我是CLAMC-Agent",
                            description=
                            "我是大语言模型驱动的智能投研助手，你可以上传文件开始进行使用~",
                            prompts=dict(
                                title="今天能为你做些什么呢?",
                                styles={
                                    "list": {
                                        "width": '100%',
                                    },
                                    "item": {
                                        "flex": 1,
                                    },
                                },
                                items=DEFAULT_PROMPTS),
                        ),
                        user_config=user_config(),
                        bot_config=bot_config())
                    # Input
                    with antdx.Suggestion(
                            items=DEFAULT_SUGGESTIONS,
                            # onKeyDown Handler in Javascript
                            should_trigger="""(e, { onTrigger, onKeyDown }) => {
                      switch(e.key) {
                        case '/':
                          onTrigger()
                          break
                        case 'ArrowRight':
                        case 'ArrowLeft':
                        case 'ArrowUp':
                        case 'ArrowDown':
                          break;
                        default:
                          onTrigger(false)
                      }
                      onKeyDown(e)
                    }""") as suggestion:
                        with ms.Slot("children"):
                            with pro.MultimodalInput(
                                    placeholder="输入 / 获取建议",
                                    upload_config=MultimodalInputUploadConfig(
                                        upload_button_tooltip=
                                        "上传附件",
                                        max_count=6,
                                        # accept="image/*",
                                        multiple=True)) as input:
                                with ms.Slot("prefix"):
                                    # Clear Button
                                    with antd.Tooltip(
                                            title="清除对话历史"
                                    ):
                                        with antd.Button(
                                                value=None,
                                                type="text") as clear_btn:
                                            with ms.Slot("icon"):
                                                antd.Icon("ClearOutlined")

    # Events Handler
    if save_history:
        browser_state = gr.BrowserState(
            {
                "conversations_history": {},
                "conversations": [],
            },
            storage_key="ms_chatbot_storage")
        state.change(fn=Gradio_Events.update_browser_state,
                     inputs=[state],
                     outputs=[browser_state])

        demo.load(fn=Gradio_Events.apply_browser_state,
                  inputs=[browser_state, state],
                  outputs=[conversations, state])

    add_conversation_btn.click(fn=Gradio_Events.new_chat,
                               inputs=[state],
                               outputs=[conversations, chatbot, state])
    conversations.active_change(fn=Gradio_Events.select_conversation,
                                inputs=[state],
                                outputs=[conversations, chatbot, state])
    conversations.menu_click(fn=Gradio_Events.click_conversation_menu,
                             inputs=[state],
                             outputs=[conversations, chatbot, state])
    chatbot.welcome_prompt_select(fn=Gradio_Events.apply_prompt,
                                  inputs=[input],
                                  outputs=[input])

    clear_btn.click(fn=Gradio_Events.clear_conversation_history,
                    inputs=[state],
                    outputs=[chatbot, state])

    suggestion.select(fn=Gradio_Events.select_suggestion,
                      inputs=[input],
                      outputs=[input])
    chatbot.delete(fn=Gradio_Events.delete_message,
                   inputs=[state],
                   outputs=[state])
    chatbot.edit(fn=Gradio_Events.edit_message,
                 inputs=[state, chatbot],
                 outputs=[state])

    regenerating_event = chatbot.retry(
        fn=Gradio_Events.regenerate_message,
        inputs=[state],
        outputs=[chatbot, state
                 ]).then(fn=Gradio_Events.preprocess_submit(clear_input=False),
                         inputs=[state],
                         outputs=[
                             input, clear_btn, conversation_delete_menu_item,
                             add_conversation_btn, conversations, chatbot,
                             state
                         ]).then(fn=Gradio_Events.submit,
                                 inputs=[state],
                                 outputs=[chatbot, state])

    submit_event = input.submit(
        fn=Gradio_Events.add_user_message,
        inputs=[input, state],
        outputs=[state
                 ]).then(fn=Gradio_Events.preprocess_submit(clear_input=True),
                         inputs=[state],
                         outputs=[
                             input, clear_btn, conversation_delete_menu_item,
                             add_conversation_btn, conversations, chatbot,
                             state
                         ]).then(fn=Gradio_Events.submit,
                                 inputs=[state],
                                 outputs=[chatbot, state])
    regenerating_event.then(fn=Gradio_Events.postprocess_submit,
                            inputs=[state],
                            outputs=[
                                input, conversation_delete_menu_item,
                                clear_btn, conversations, add_conversation_btn,
                                chatbot, state
                            ])
    submit_event.then(fn=Gradio_Events.postprocess_submit,
                      inputs=[state],
                      outputs=[
                          input, conversation_delete_menu_item, clear_btn,
                          conversations, add_conversation_btn, chatbot, state
                      ])
    input.cancel(fn=Gradio_Events.cancel,
                 inputs=[state],
                 outputs=[
                     input, conversation_delete_menu_item, clear_btn,
                     conversations, add_conversation_btn, chatbot, state
                 ],
                 cancels=[submit_event, regenerating_event],
                 queue=False)


static_server_process = None

def start_static_server():
    """启动静态文件服务器以服务 code_interpreter 生成的图片"""
    global static_server_process
    try:
        # 1. 获取 code_interpreter 的实际工作目录

        work_dir = r'D:\PROJECT\Chinalife-Qwen-Agent\workspace\tools\code_interpreter'
        os.makedirs(work_dir, exist_ok=True)  # 确保目录存在
        port = 8000  # 选择一个 Gradio 不会使用的端口
        print(f"[INFO] Starting static file server for directory: {work_dir} on port {port}")

        # 2. 构建命令
        # 使用 sys.executable 确保使用当前 Python 环境
        cmd = [sys.executable, '-m', 'http.server', str(port), '--directory', work_dir]
        print(f"[DEBUG] Static server command: {' '.join(cmd)}")

        # 3. 启动子进程
        # cwd=work_dir 设置工作目录,
        # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL 隐藏输出
        # 如果需要调试，可以注释掉 DEVNULL 部分
        static_server_process = subprocess.Popen(
            cmd,
            cwd=work_dir,
            stdout=subprocess.DEVNULL,  # subprocess.PIPE for debugging
            stderr=subprocess.DEVNULL,  # subprocess.PIPE for debugging
        )

        # 4. 简单等待服务器启动
        time.sleep(1)
        if static_server_process.poll() is not None:  # 如果进程已经结束
             print(f"[ERROR] Static server failed to start. Return code: {static_server_process.returncode}")
             # 可以尝试读取 stderr 来获取错误信息 (如果没用 DEVNULL)
             # stderr_output = static_server_process.stderr.read() if static_server_process.stderr else b''
             # print(f"[ERROR] Static server error output: {stderr_output.decode()}")
             static_server_process = None
             return

        print(f"[INFO] Static file server started (PID: {static_server_process.pid}).")

        # 5. 设置环境变量供 code_interpreter 使用
        static_url = f"http://localhost:{port}"
        os.environ['M6_CODE_INTERPRETER_STATIC_URL'] = static_url
        print(f"[INFO] Set M6_CODE_INTERPRETER_STATIC_URL={static_url}")

    except Exception as e:
        print(f"[ERROR] Failed to start static file server: {e}")
        import traceback
        traceback.print_exc()
        static_server_process = None


def stop_static_server():
    """停止静态文件服务器"""
    global static_server_process
    if static_server_process and static_server_process.poll() is None:  # 检查是否仍在运行
        print("[INFO] Stopping static file server...")
        try:
            static_server_process.terminate()
            static_server_process.wait(timeout=5)  # 等待最多5秒
            print("[INFO] Static file server stopped.")
        except subprocess.TimeoutExpired:
            print("[WARNING] Static server did not terminate in time, killing it...")
            static_server_process.kill()
            static_server_process.wait()
            print("[INFO] Static file server killed.")
        except Exception as e:
            print(f"[ERROR] Error stopping static server: {e}")
    elif static_server_process:
        print(f"[INFO] Static file server (PID: {static_server_process.pid}) had already exited.")


if __name__ == "__main__":
    start_static_server()

    atexit.register(stop_static_server)

    try:
        demo.queue().launch(share=False)
    finally:
        stop_static_server()

