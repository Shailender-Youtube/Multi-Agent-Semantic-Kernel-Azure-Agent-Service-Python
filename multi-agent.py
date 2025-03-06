import asyncio
import os
import aiohttp
from dotenv import load_dotenv

from azure.identity.aio import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from semantic_kernel.agents.azure_ai import AzureAIAgent

# Load configuration from .env file
load_dotenv()

# ----- Agent IDs Configuration -----
# Document Search Agent (Agent 1)
AGENT1_ID = "xxxxx"

# Web Search Agent (Agent 2)
AGENT2_ID = "xxxxxx"

# Summary Agent (Agent 3)
AGENT3_ID = "xxxxxx"

# ----- SERP API and Azure Project Configuration -----
SERP_API_KEY = os.environ.get("SERP_API_KEY")
SERP_ENDPOINT = os.environ.get("SERP_ENDPOINT", "https://serpapi.com/search.json")
PROJECT_CONN_STR = os.environ.get("AZURE_AI_PROJECT_CONNECTION_STRING")


async def search_web(query: str) -> str:
    """
    Perform a web search using the SERP API and return a combined snippet summary.
    """
    params = {
        "api_key": SERP_API_KEY,
        "q": query,
        "location": "Australia",
        "hl": "en"
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(SERP_ENDPOINT, params=params) as response:
            result = await response.json()
            snippets = []
            if "organic_results" in result:
                for item in result["organic_results"]:
                    snippet = item.get("snippet")
                    if snippet:
                        snippets.append(snippet)
            return "\n".join(snippets) if snippets else "No results found."


async def main():
    # Prompt the user for a query regarding planning/building permits
    user_query = input("Enter your query for planning and building permit info: ")

    # Initialize credentials and the Azure AI Projects client
    credential = DefaultAzureCredential()
    try:
        project_client = AIProjectClient.from_connection_string(
            conn_str=PROJECT_CONN_STR,
            credential=credential
        )
        print("✅ Project client initialized.")
    except Exception as e:
        print(f"❌ Error initializing project client: {e}")
        return

    # Use the async context for proper cleanup of the credential and client
    async with credential, AzureAIAgent.create_client(credential=credential) as client:
        # Retrieve pre-created agent definitions for each role
        document_agent_def = await client.agents.get_agent(assistant_id=AGENT1_ID)
        web_agent_def = await client.agents.get_agent(assistant_id=AGENT2_ID)
        summary_agent_def = await client.agents.get_agent(assistant_id=AGENT3_ID)

        # Instantiate Semantic Kernel agent objects
        document_agent = AzureAIAgent(client=client, definition=document_agent_def)
        web_agent = AzureAIAgent(client=client, definition=web_agent_def)
        summary_agent = AzureAIAgent(client=client, definition=summary_agent_def)

        # ----- Agent 1: Document Search Agent -----
        # Create a thread for Agent 1 and send the user's query as a chat message.
        thread1 = await client.agents.create_thread()
        await document_agent.add_chat_message(
            thread_id=thread1.id,
            message=user_query
        )
        response_doc = await document_agent.get_response(thread_id=thread1.id)
        print("\n[Document Search Agent Response]")
        print(response_doc)

        # ----- Agent 2: Web Search Agent -----
        # Perform a SERP API search with the user's query.
        web_results = await search_web(user_query)
        thread2 = await client.agents.create_thread()
        await web_agent.add_chat_message(
            thread_id=thread2.id,
            message=f"Web search results for '{user_query}':\n{web_results}"
        )
        response_web = await web_agent.get_response(thread_id=thread2.id)
        print("\n[Web Search Agent Response]")
        print(response_web)

        # ----- Agent 3: Summary Agent -----
        # Combine responses from Agent 1 and Agent 2 and send to Agent 3 for summarization.
        combined_info = (
            f"Document Agent info: {response_doc}\n\n"
            f"Web Agent info: {response_web}"
        )
        thread3 = await client.agents.create_thread()
        await summary_agent.add_chat_message(
            thread_id=thread3.id,
            message=combined_info
        )
        response_summary = await summary_agent.get_response(thread_id=thread3.id)
        print("\n[Summary Agent Response]")
        print(response_summary)

        # ----- Optional Cleanup: Delete the conversation threads -----
        for thread in [thread1, thread2, thread3]:
            try:
                await client.agents.delete_thread(thread.id)
            except Exception as e:
                print(f"❌ Error deleting thread {thread.id}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
