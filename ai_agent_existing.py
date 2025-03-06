import asyncio
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents.azure_ai import AzureAIAgent


async def main() -> None:
    # Use asynchronous context managers to authenticate and create the client.
    async with (
        DefaultAzureCredential() as creds,
        AzureAIAgent.create_client(credential=creds) as client,
    ):
        # 1. Retrieve the agent definition based on the assistant ID.
        #    Replace "asst_MwpyijFo7T4MEzwvS8Pb5F98" with your actual assistant ID.
        agent_definition = await client.agents.get_agent(
            assistant_id="xxxxxxx",
        )

        # 2. Create a Semantic Kernel agent using the retrieved definition.
        agent = AzureAIAgent(client=client, definition=agent_definition)

        # 3. Create a new conversation thread.
        thread = await client.agents.create_thread()

        try:
            # 4. Define the single query and add it as a chat message.
            user_query = "Can Azure App Services be integrated with Vnet?"
            await agent.add_chat_message(thread_id=thread.id, message=user_query)
            print(f"# User: {user_query}")

            # 5. Retrieve and print the agent's response.
            response = await agent.get_response(thread_id=thread.id)
            print(f"# Agent: {response}")
        finally:
            # 6. Cleanup: Delete the conversation thread.
            await client.agents.delete_thread(thread.id)
            # Note: The agent is not deleted so it can be reused later.

if __name__ == "__main__":
    asyncio.run(main())
