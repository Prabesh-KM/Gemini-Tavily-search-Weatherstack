import os
import requests
from dotenv import load_dotenv

from langchain import hub
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()  # reads GOOGLE_API_KEY, TAVILY_API_KEY, WEATHERSTACK_API_KEY from .env


@tool
def get_weather_data(city: str) -> str:
    """Get the current weather for a city."""
    url = "http://api.weatherstack.com/current"
    params = {"access_key": os.getenv("WEATHERSTACK_API_KEY"), "query": city}
    data = requests.get(url, params=params, timeout=10).json()

    if "current" not in data:
        return f"Could not fetch weather for {city}"

    w = data["current"]
    return f"{city}: {w['temperature']}°C, {w['weather_descriptions'][0]}, humidity {w['humidity']}%"


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tools = [TavilySearchResults(max_results=2), get_weather_data]
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

response = executor.invoke({"input": "Find the capital of Nepal and then find its current weather."})
print("\nFINAL OUTPUT:\n", response["output"])