
# This code initializes a Wikipedia search agent using Langchain and OpenAI's language model.
# It defines a function to run the agent with a given query and prints the response.
# The agent uses the WikipediaAPIWrapper to search for information on Wikipedia.
# The agent is set to use a zero-shot react description approach, which means it can generate responses without prior training on specific tasks.
# The agent is initialized with a temperature of 0, which means it will generate more deterministic responses.
# The code is designed to be run as a standalone script, and it will print the response to the console.
# The agent is capable of searching Wikipedia for information and providing answers to user queries.

from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import WikipediaAPIWrapper

from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Check if the API key is set
if os.environ["OPENAI_API_KEY"] is None:
    raise ValueError("OpenAI API key is not set. Please set it in the .env file.")

# Wikipedia Tool
wiki_tool = Tool(
    name="Wikipedia Search",
    func=WikipediaAPIWrapper().run,
    description="A tool to search Wikipedia for information. Use this tool when you need to find information on a specific topic. The input should be a search query.",
)

# LLM Model
llm = OpenAI(temperature=0)


tools = [wiki_tool]
# Initialize the agent with the Wikipedia tool and the language model
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True
)

def run_agent(query):
    """
    Run the agent with the given query.
    """
    response = agent.run(query)
    return response

if __name__ == "__main__":
    query = "Tell me the history of AI - Artificial Intelligence."
    response = run_agent(query)
    print(response)

# The code is structured to allow for easy modification and extension, such as adding more tools or changing the language model.