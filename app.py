from phi.agent import Agent
from phi.tools.youtube_tools import YouTubeTools
from phi.llm.groq import Groq
from phi.assistant import Assistant
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize Assistant with the specified Groq model
assistant = Assistant(
    llm=Groq(model="mixtral-8x7b-32768")  # You can change the model if needed
)

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    print("Groq API key loaded successfully.")
else:
    print("Error: Groq API key is not loaded.")

# Use Assistant to print the response for the query
assistant.print_response("Give a list of videos on Machine Learning", markdown=True)
