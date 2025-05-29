from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo



websearch_agent = Agent(
    name = "Web search Agent",
    role = "Search the web for the information"
    model = Groq
)