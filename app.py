import os
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain import hub
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


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


@st.cache_resource
def get_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    tools = [TavilySearchResults(max_results=2), get_weather_data]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


st.title("🤖 AI Agent Dashboard")
st.caption("Gemini + Tavily search + Weatherstack")

question = st.text_area(
    "Ask the agent",
    value="Find the capital of Nepal and then find its current weather.",
)

if st.button("Run agent", type="primary"):
    if not question.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Agent is thinking..."):
            try:
                result = get_agent().invoke({"input": question})
            except Exception as e:
                st.error(f"Agent failed: {e}")
                st.stop()

        st.subheader("Answer")
        st.success(result["output"])

        with st.expander("Agent steps"):
            for action, observation in result["intermediate_steps"]:
                st.markdown(f"**Tool:** `{action.tool}`")
                st.code(str(action.tool_input))
                st.text(str(observation)[:1000])