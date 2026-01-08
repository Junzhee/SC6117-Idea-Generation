import os
from openai import OpenAI
from dotenv import load_dotenv

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

    def _build_prompt(self, pain_points):
        """
        Constructs the prompt messages for the LLM.
        Edit this function to change how the AI behaves.
        
        Args:
            pain_points (list): List of dictionaries containing user feedback.
            
        Returns:
            list: A list of message dictionaries for the OpenAI API.
        """
        # 1. Format the evidence from raw data
        # We take the top 10 points to avoid hitting token limits
        evidence_text = "\n".join(
            [f"- User Complaint: '{p['content']}' (Likes: {p['likes']})" 
             for p in pain_points[:10]]
        )

        # 2. Define the System Persona
        system_content = (
            "You are an expert Venture Capitalist and Product Manager. "
            "Your goal is to identify market opportunities from user complaints."
        )

        # 3. Define the specific instruction (The Prompt)
        user_content = f"""
        I have analyzed social media data for the Robot Vacuum market and found the following VALIDATED user pain points:

        {evidence_text}

        Based on these specific pain points, act as a founder and propose a NEW Startup Product Concept.
        
        Please structure your response strictly as follows:
        
        1. **Startup Name**: (Catchy and relevant)
        2. **One-Liner Pitch**: (A compelling value proposition)
        3. **The Problem**: (Summarize the key pain points you are solving based on the evidence)
        4. **The Solution**: (Describe the specific features that solve the complaints above)
        5. **Target Audience**: (Who is most frustrated currently?)
        6. **differentiation**: (Why is this better than current market leaders like DJI/Roborock?)
        """

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def generate_idea(self, pain_points):
        """
        Orchestrates the API call to generate a startup idea.
        """
        if not pain_points:
            return "Error: No pain points provided. Please analyze data first."

        # Get the constructed prompt messages
        messages = self._build_prompt(pain_points)

        try:
            print("Sending request to DeepSeek API...")
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False,
                temperature=0.7 # Creativity level
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating idea: {str(e)}"

if __name__ == "__main__":
    # Test Code
    
    # Check for API Key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Skipping actual API call: No Key found in .env")
        print("Set DEEPSEEK_API_KEY to run the real test.")
    else:
        gen = Generator()
        
        # Mock Data similar to what analyzer.py outputs
        dummy_points = [
            {'content': 'The water tank leaks everywhere!', 'likes': 120},
            {'content': 'It gets stuck on thick carpets constantly.', 'likes': 85},
            {'content': 'The app disconnects too often.', 'likes': 45}
        ]
        
        # Test Prompt Construction
        print("\n[1] Testing Prompt Construction...")
        msgs = gen._build_prompt(dummy_points)
        print(f"System Prompt: {msgs[0]['content'][:50]}...")
        print(f"User Prompt Length: {len(msgs[1]['content'])} chars")
        
        # Test API Call (Optional cost)
        # print("\n[2] Testing Live Generation (Uncomment to run)...")
        # result = gen.generate_idea(dummy_points)
        # print("\nGenerated Output:\n")
        # print(result)
        
        print("\nGenerator module test finished.")