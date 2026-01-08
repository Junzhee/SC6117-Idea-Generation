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

    # -------------------- prompt builder --------------------
    def _build_prompt(self, pain_points, evidence_text: str):
        pain_text = "\n".join([f"- {p.get('content', '')}" for p in pain_points[:20]])

        system_content = (
            "You are a world-class Product Strategist and Venture Capitalist (like a YC partner). "
            "You are analyzing raw market feedback to build a unicorn startup concept. "
            "Your tone is professional, insightful, and evidence-based. "
            "Use Markdown formatting heavily (bullet points, bold text) to make the output readable."
        )

        user_content = f"""
### Context
You are analyzing the **Robot Vacuum (扫地机器人)** market based on real user feedback.

### Key Pain Points (from Analysis)
{pain_text}

### Deep Dive Evidence (Raw Comments & Threads)
{evidence_text}

### Mission
Based **ONLY** on the evidence above, generate a structured Investment Pitch.
Do not hallucinate features that users don't need. Solve the specific complaints found in the text.

### Output Format Rules (STRICT)
You must output exactly 6 sections with the specific headers below. Do not generate tables.
Inside each section, use **Markdown Lists**, **Bold key phrases**, and clear structure.
You can be flexible, if you think the things is too noncense, you can provide your own reasonable ideas.

1. **Startup Name**: 
   - Provide a catchy, modern name (English).
   - *Format*: Just the name and a 1-sentence explanation of the name.

2. **One-Liner Pitch**: 
   - A single, punchy sentence (< 20 words) describing the value prop.
   - *Format*: Blockquote this sentence (use >).

3. **The Problem**: 
   - Don't just list generic issues. Synthesize the "Deep Frustration".
   - **Must cite specific user complaints** from the evidence (e.g., "Users frequently complain about tangled hair...").
   - *Format*: Use bullet points. Bold the core pain point (e.g., "**Tangled Hair**: Users are tired of...").
   - Generate more than 5 points.

4. **The Solution**: 
   - Propose concrete product features that directly map to the problems above.
   - Be specific (e.g., instead of "Better AI", say "Vision-based obstacle recognition for cables").
   - *Format*: Use a bulleted list of features. Use Emoji for each feature (e.g., 🧶 **Anti-Tangle Roller**: ...).

5. **Target Audience**: 
   - Define the specific persona who is complaining the most (e.g., Pet owners, Parents, Tech enthusiasts).
   - *Format*: Short description + 3-5 key characteristics in a list.

6. **Differentiation**: 
   - Why will this win against giants like DJI/Roborock?
   - Focus on the "Unmet Needs" found in the evidence.
   - *Format*: A short comparison list.

"Write in English only. Make the content look professional and visually structured."
"""

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def generate_pitch_from_json(self, json_path: str):
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
    json_path = "data/analyzer.json"

    gen = Generator()

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
