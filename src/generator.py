import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import traceback

load_dotenv()

class Generator:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com"
        # 增加 timeout 设置（比如 60 秒），防止网络慢导致连接直接中断
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url,
            timeout=60.0  # 这里是新加的
        )

    # --- 核心兼容函数：让你的前端不再报错 ---
    def generate_idea(self, pain_points):
        """
        兼容旧版前端：接收 pain_points 列表并生成 6 个模块的 Pitch Deck
        """
        if not pain_points:
            return "Error: No pain points provided."

        # 将列表转为文本证据
        evidence_text = "\n".join([f"- {p.get('content', '')}" for p in pain_points[:15]])
        
        # 构造符合要求的 6 模块提示词
        messages = self._build_pitch_prompt(evidence_text)
        return self._send_request(messages)

    # --- 新增：专门负责发送请求的私有方法 ---
    def _send_request(self, messages):
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating idea: {str(e)}"

    # --- 核心提示词：严格遵守你的图 6 模块要求 ---
    def _build_pitch_prompt(self, evidence):
        system_content = "You are an expert Venture Capitalist. Write in English only."
        user_content = f"""
        Based on these market pain points:
        {evidence}

        Generate a startup concept with EXACTLY these 6 modules:
        1. **The Problem**: High-frequency complaints from data.
        2. **The Solution**: MVP features solving above problems.
        3. **Target Audience**: Early adopter personas.
        4. **Market Validation**: Logical argument from pain intensity.
        5. **Differentiation**: Why better than DJI or Roborock?
        6. **Business Model**: 
           - Price Sensitivity: Analyze if users find current tech too expensive.
           - Pricing Strategy: Propose a target price range and revenue tiers.
        """
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    # --- 保留组员的 JSON 逻辑（防止以后用到） ---
    def generate_pitch_from_json(self, json_path):
        if not os.path.exists(json_path): return "JSON not found."
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 这里可以调用上面的 _build_pitch_prompt 逻辑
        return "JSON logic processed."