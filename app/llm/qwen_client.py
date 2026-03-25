import dashscope
from dashscope import Generation
from typing import AsyncGenerator, Optional
import asyncio
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class QwenClient:
    """通义千问客户端（支持流式和重试）"""
    
    def __init__(self):
        self.settings = get_settings()
        dashscope.api_key = self.settings.dashscope_api_key
        self.model = self.settings.qwen_model
        self.max_retries = self.settings.max_retries
        self.retry_delay = self.settings.retry_delay
    
    async def generate_with_retry(self, prompt: str, **kwargs) -> str:
        """带重试的生成"""
        for attempt in range(self.max_retries):
            try:
                response = Generation.call(
                    model=self.model,
                    prompt=prompt,
                    **kwargs
                )
                
                if response.status_code == 200:
                    return response.output.text
                else:
                    logger.warning(f"API调用失败: {response.code}")
                    
            except Exception as e:
                logger.warning(f"第{attempt+1}次尝试失败: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
        
        raise Exception("达到最大重试次数")
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """流式生成"""
        for attempt in range(self.max_retries):
            try:
                responses = Generation.call(
                    model=self.model,
                    prompt=prompt,
                    stream=True,
                    **kwargs
                )
                
                for response in responses:
                    if response.status_code == 200:
                        yield response.output.text
                    else:
                        logger.error(f"流式调用失败: {response.code}")
                        break
                        
            except Exception as e:
                logger.warning(f"流式第{attempt+1}次尝试失败: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
        
    def generate_structured(self, prompt: str, output_schema: dict) -> dict:
        """结构化输出"""
        system_prompt = f"""你是一个专业的技术文档助手。请严格按照以下JSON Schema格式返回答案：

{output_schema}

重要：只返回JSON对象，不要包含任何其他文本。"""

        response = Generation.call(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            result_format='message'
        )
        
        if response.status_code == 200:
            import json
            content = response.output.choices[0].message.content
            return json.loads(content)
        else:
            raise Exception(f"API调用失败: {response.code}")