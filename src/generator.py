import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import traceback

# Load env variables
load_dotenv()


class Generator:
    def __init__(self):
        """
        Initialize the OpenAI client using DeepSeek configuration.
        """
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com"

        if not self.api_key:
            print("WARNING: DEEPSEEK_API_KEY not found in .env")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    # -------------------- JSON loader --------------------
    def load_analyzer_json(self, json_path: str) -> dict:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------- pain_points from aspect --------------------
    def build_pain_points_from_aspects(self, data: dict, max_points: int = 30):
        """
        pain_points = key_findings[].aspect (去重)
        返回格式与旧代码兼容: [{'content': str, 'likes': int}, ...]
        """
        pain_points = []
        seen = set()

        for k in (data.get("key_findings", []) or []):
            aspect = (k.get("aspect") or "").strip()
            if not aspect:
                continue
            if aspect in seen:
                continue
            pain_points.append({"content": aspect, "likes": 0})
            seen.add(aspect)

        return pain_points[:max_points]

    # -------------------- evidence text (use as much JSON as possible) --------------------
    def build_evidence_text(
        self,
        data: dict,
        max_findings: int = 10,
        max_comments_per_finding: int = 3,
        max_threads: int = 10
    ) -> str:
        """
        把 analyzer.json 尽可能多的信息，压缩成可喂给 LLM 的 evidence 文本块
        """
        parts = []

        # meta
        report_period = data.get("report_period")
        total_notes = data.get("total_notes")
        total_comments = data.get("total_comments")
        meta_line = []
        if report_period:
            meta_line.append(f"report_period={report_period}")
        if total_notes is not None:
            meta_line.append(f"total_notes={total_notes}")
        if total_comments is not None:
            meta_line.append(f"total_comments={total_comments}")
        if meta_line:
            parts.append("=== Meta ===")
            parts.append("- " + " | ".join(meta_line))

        # key_findings
        kfs = data.get("key_findings", []) or []
        if kfs:
            parts.append("\n=== Key Findings ===")
            for i, k in enumerate(kfs[:max_findings], start=1):
                aspect = (k.get("aspect") or "").strip()
                summary = (k.get("summary") or "").strip()
                score = k.get("sentiment_score", None)

                header = f"[KF-{i}]"
                if aspect:
                    header += f" aspect={aspect}"
                if score is not None:
                    header += f" sentiment_score={score}"

                parts.append(header)
                if summary:
                    parts.append(f"- summary: {summary}")

                reps = k.get("top_representative_comments", []) or []
                reps = [c for c in reps if c and c != "..."][:max_comments_per_finding]
                if reps:
                    parts.append("- representative_comments:")
                    for c in reps:
                        parts.append(f"  - {c.strip()}")

        # hot_threads
        hts = data.get("hot_threads", []) or []
        if hts:
            parts.append("\n=== Hot Threads ===")
            for i, t in enumerate(hts[:max_threads], start=1):
                note_id = (t.get("note_id") or "").strip()
                topic = (t.get("topic") or "").strip()
                level = (t.get("controversy_level") or "").strip()
                summ = (t.get("thread_summary") or "").strip()

                header = f"[HT-{i}]"
                if topic:
                    header += f" topic={topic}"
                if level:
                    header += f" controversy_level={level}"
                if note_id:
                    header += f" note_id={note_id}"

                parts.append(header)
                if summ:
                    parts.append(f"- thread_summary: {summ}")

        return "\n".join(parts) if parts else "No evidence found in analyzer JSON."

    # -------------------- prompt builder: MUST output 6 modules --------------------
    def _build_prompt(self, pain_points, evidence_text: str):
        """
        pain_points: 从 aspect 来（简短）
        evidence_text: JSON 里尽可能多的信息（summary/评论/热帖/统计）
        """

        pain_text = "\n".join([f"- {p.get('content', '')}" for p in pain_points[:15]])

        system_content = (
            "You are an expert Venture Capitalist and Product Manager. "
            "Your goal is to identify market opportunities and propose a startup concept. "
            "Write in English only."
        )

        user_content = f"""
You are analyzing the Robot Vacuum market.

Pain Points (extracted from JSON key_findings[].aspect):
{pain_text}

Full Evidence (use this heavily; do not ignore it):
=== EVIDENCE START ===
{evidence_text}
=== EVIDENCE END ===

Task:
Based ONLY on the evidence above, generate an investor-ready pitch with EXACTLY these 6 modules (same titles and numbering).
If evidence is insufficient for a module, write "Insufficient evidence in input" for that part rather than inventing facts.

Output format (STRICT, no extra sections, no tables):
        1. **Startup Name**: (Catchy and relevant)
        2. **One-Liner Pitch**: (A compelling value proposition)
        3. **The Problem**: (Summarize the key pain points you are solving based on the evidence)
        4. **The Solution**: (Describe the specific features that solve the complaints above)
        5. **Target Audience**: (Who is most frustrated currently?)
        6. **differentiation**: (Why is this better than current market leaders like DJI/Roborock?)
"Write in English only."

"""

        # ✅ 修复你原文件这里少逗号的语法错误 :contentReference[oaicite:2]{index=2}
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def generate_pitch_from_json(self, json_path: str):
        """
        一步到位：读取 JSON -> aspect pain_points + full evidence -> 生成 6 模块输出
        """
        data = self.load_analyzer_json(json_path)
        pain_points = self.build_pain_points_from_aspects(data)
        evidence_text = self.build_evidence_text(data)

        messages = self._build_prompt(pain_points, evidence_text)

        try:
            print("Sending request to DeepSeek API...")
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            traceback.print_exc()
            return f"Error generating pitch deck: {repr(e)}"


if __name__ == "__main__":
    # 默认同目录 analyzer.json；你也可以改成绝对路径
    json_path = "analyzer.json"

    gen = Generator()

    # 没 key：只打印 evidence 和 prompt 预览，方便你检查“有没有用上全部 JSON”
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Skipping actual API call: No Key found in .env")
        print("Set DEEPSEEK_API_KEY to run the real test.\n")
        data = gen.load_analyzer_json(json_path)
        pain_points = gen.build_pain_points_from_aspects(data)
        evidence_text = gen.build_evidence_text(data)
        msgs = gen._build_prompt(pain_points, evidence_text)
        print("---- Evidence Preview ----")
        print(evidence_text[:1500], "...\n")
        print("---- Prompt Preview ----")
        print(msgs[1]["content"][:1500], "...\n")
    else:
        result = gen.generate_pitch_from_json(json_path)
        print(result)
