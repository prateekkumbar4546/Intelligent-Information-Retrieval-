from langchain.agents import Tool, AgentExecutor
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import OpenAIFunctionsAgent
import os
import logging
import base64
import requests

logger = logging.getLogger(__name__)

class AgentSystem:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=100)
        self.setup_tools()
        self.setup_agent()
        self.setup_sarvam_tts()

    def setup_tools(self):
        self.tools = [
            Tool(
                name="Sound_Query",
                func=self.sound_query,
                description="Use this for questions about Sound, Sound waves, or Ultrasound from NCERT textbooks."
            ),
            Tool(
                name="Calculate",
                func=self.calculate,
                description="Useful for performing mathematical calculations. Input should be a mathematical expression."
            ),
            Tool(
                name="General_Chat",
                func=self.general_chat,
                description="Useful for general conversation, greetings, or when other tools are not applicable."
            )
        ]

    def sound_query(self, query):
        logger.info(f"Sound Query tool called with query: {query}")
        response = self.rag_system.query(query)
        logger.info(f"Sound Query response: {response}")
        return response

    def calculate(self, expression):
        logger.info(f"Calculate tool called with expression: {expression}")
        try:
            result = str(eval(expression))
            logger.info(f"Calculation result: {result}")
            return result
        except:
            logger.error(f"Invalid expression: {expression}")
            return "Invalid expression. Please provide a valid mathematical expression."

    def general_chat(self, query):
        logger.info(f"General Chat tool called with query: {query}")
        response = self.llm.predict(f"""You are a friendly AI assistant. Respond to this query appropriately: {query}""")
        logger.info(f"General Chat response: {response}")
        return response

    def setup_agent(self):
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message="""You are an AI assistant specialized in Sound, Sound waves, and Ultrasound, with knowledge from NCERT textbooks. 
            Your primary function is to provide accurate information on these topics and help with related calculations.

            1. For queries about Sound, Sound waves, or Ultrasound, use the "Sound_Query" tool.
            2. For mathematical calculations related to Sound physics, use the "Calculate" tool.
            3. For general questions or non-Sound topics, use the "General_Chat" tool.

            Always prioritize using the Sound_Query tool for any Sound-related questions, as it contains specific NCERT textbook information."""
        )
        
        self.agent = OpenAIFunctionsAgent(llm=self.llm, tools=self.tools, prompt=prompt)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent, tools=self.tools, verbose=True
        )

    def setup_sarvam_tts(self):
        self.sarvam_api_key = os.getenv("SARVAM_API_KEY")
        if not self.sarvam_api_key:
            logger.error("SARVAM_API_KEY is not set in the environment variables")
            raise ValueError("SARVAM_API_KEY is not set")
        
        # Updated API endpoint
        self.sarvam_tts_url = "https://api.sarvam.ai/text-to-speech"
        self.sarvam_headers = {
    "Content-Type": "application/json",
    "API-Subscription-Key": self.sarvam_api_key
}

    def text_to_speech(self, text):
        payload = {
            "inputs": [
        text
    ],    
    "target_language_code": "en-IN",  
    "speaker": "meera", 
    "pitch": 0,  
    "pace": 1.0,
    "loudness": 1.0, 
    "speech_sample_rate": 8000, 
    "enable_preprocessing": True,
    "model": "bulbul:v1"
    }
        
        try:
            response = requests.post(self.sarvam_tts_url, json=payload, headers=self.sarvam_headers)
            response.raise_for_status()
            # print("Response Headersssssss", response.headers) 
            response_json = response.json()
            # print("Response JSON",response_json)  
            audio_content = response_json['audios']
            print(f"Length of audio content: {len(audio_content)}")  # Debug log
            # audio_base64 = base64.b64encode(audio_content).decode('utf-8')
            # print(f"Base64 length: {len(audio_base64)}")
            return audio_content
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in TTS API call: {str(e)}")
            if response.status_code == 404:
                logger.error("API endpoint not found. Please check the Sarvam AI documentation for the correct endpoint.")
            elif response.status_code == 401:
                logger.error("Unauthorized. Please check your Sarvam AI API key.")
            return None

    def run(self, query):
        logger.info(f"Agent system received query: {query}")
        response = self.agent_executor.run(query)
        logger.info(f"Agent executor response: {response}")
        return response