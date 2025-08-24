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

import json
import os
from typing import List, Tuple
import requests
from qwen_agent.tools.base import register_tool
from qwen_agent.tools.doc_parser import Record
from qwen_agent.tools.search_tools.base_search import BaseSearch
from langchain_core.embeddings import Embeddings

# class clamc_embedding(Embeddings):
# 自定义 Embedding 类，用于调用您的 embedding 服务
class ClamcEmbeddings(Embeddings):
    def __init__(self, 
                 url: str = "http://10.4.31.223:8077/v1/embeddings", 
                 headers: dict = None):
        self.url = url
        if headers is None:
            self.headers = {
                "Authorization": "Bearer clamc2024@GPT%72",
                "Content-Type": "application/json"
            }
        else:
            self.headers = headers

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 准备当前批次的数据
            data = {
                "encoding_format": "float",
                "input": batch_texts
            }
            
            # 发送请求
            response = requests.post(self.url, headers=self.headers, json=data)
            response.raise_for_status()  # 检查 HTTP 错误
            
            # 解析结果
            result = response.json()
            batch_embeddings = [item["embedding"] for item in result["data"]]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        # 单个查询嵌入
        data = {
            "model": "text-embedding-v4",
            "encoding_format": "float",
            "input": text
        }
        response = requests.post(self.url, headers=self.headers, json=data)
        response.raise_for_status()
        result = response.json()
        # 假设返回格式: {"data": [{"embedding": [...]}]}
        return result["data"][0]["embedding"]

def rerank_query(query: str, 
                 texts: List[str], 
                 url: str = "http://10.4.31.223:8078/rerank", 
                 headers: dict = None) -> List[float]:
    if headers is None:
        headers = {
            "Authorization": "Bearer clamc2024@GPT%72",
            "Content-Type": "application/json"
        }
    batch_size = 10
    all_scores = [0.0] * len(texts)
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # 准备当前批次的数据
        data = {
            "query": query,
            "texts": batch_texts
        }
        
        # 发送请求
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        # 解析结果
        result = response.json()
        
        # 根据返回结果中的索引填充分数（注意索引是相对于当前批次的）
        for item in result:
            original_index = i + item['index']  # 计算原始索引
            score = item['score']
            if 0 <= original_index < len(texts):
                all_scores[original_index] = score
    
    return all_scores

@register_tool('vector_search')
class VectorSearch(BaseSearch):
    # TODO: Optimize the accuracy of the embedding retriever.

    def sort_by_scores(self, query: str, docs: List[Record], **kwargs) -> List[Tuple[str, int, float]]:
        # TODO: More types of embedding can be configured
        try:
            from langchain.schema import Document
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Please install langchain by: `pip install langchain`')
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
            from langchain_community.vectorstores import FAISS
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                'Please install langchain_community by: `pip install langchain_community`, '
                'and install faiss by: `pip install faiss-cpu` or `pip install faiss-gpu` (for CUDA supported GPU)')
        # Extract raw query
        try:
            query_json = json.loads(query)
            # This assumes that the user's input will not contain json str with the 'text' attribute
            if 'text' in query_json:
                query = query_json['text']
        except json.decoder.JSONDecodeError:
            pass

        # Plain all chunks from all docs
        all_chunks = []
        for doc in docs:
            for chk in doc.raw:
                all_chunks.append(Document(page_content=chk.content, metadata=chk.metadata))

        # embeddings = ClamcEmbeddings(
        #     url="https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
        #     headers={
        #         "Authorization": f"Bearer sk-5e15f18b5bbd46fdb8a7a710af9a1f83",
        #         "Content-Type": "application/json"
        #     },
        # )
        embeddings = ClamcEmbeddings()
        # embeddings = DashScopeEmbeddings(model='text-embedding-v1',
                                        #  dashscope_api_key=os.getenv('DASHSCOPE_API_KEY', ''))
        all_chunks = all_chunks[:10]
        db = FAISS.from_documents(all_chunks, embeddings)
        chunk_and_score = db.similarity_search_with_score(query, k=50)
        rerank_score = rerank_query(query, texts=[chk[0].page_content for chk in chunk_and_score])
        chunk_and_score = [(chk[0], rerank_score[i]) for i, chk in enumerate(chunk_and_score)]
        chunk_and_score.sort(key=lambda x: x[1])
        return [(chk.metadata['source'], chk.metadata['chunk_id'], score) for chk, score in chunk_and_score]
