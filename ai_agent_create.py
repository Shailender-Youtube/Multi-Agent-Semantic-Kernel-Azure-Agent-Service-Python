import asyncio
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents.azure_ai import AzureAIAgent, AzureAIAgentSettings


async def main() -> None:
    # Create default agent settings (ensuring model_deployment_name is set via your environment).
    ai_agent_settings = AzureAIAgentSettings.create()

    async with (
        DefaultAzureCredential() as creds,
        AzureAIAgent.create_client(credential=creds) as client,
    ):
        # 1. Create an agent on the Azure AI agent service.
        #    This agent is named "TechSupportAdvisor" with instructions for providing technical support.
        agent_definition = await client.agents.create_agent(
            model=ai_agent_settings.model_deployment_name,
            name="TechSupportAdvisor",
            instructions="You are a helpful assistant that provides technical support regarding Azure services. Keep the answers short and concise",
        )

        # 2. Create a Semantic Kernel agent using the retrieved agent definition.
        agent = AzureAIAgent(client=client, definition=agent_definition)

        # 3. Start a new conversation thread on the Azure AI agent service.
        thread = await client.agents.create_thread()

        try:
            # 4. Send a single query to the agent.
            user_query = "Can Azure App Services be integrated with Vnet?"
            await agent.add_chat_message(thread_id=thread.id, message=user_query)
            print(f"# User: {user_query}")

            # 5. Retrieve and print the agent's response.
            response = await agent.get_response(thread_id=thread.id)
            print(f"# TechSupportAdvisor: {response}")
        finally:
            # 6. Cleanup: Delete the conversation thread and the created agent.
            await client.agents.delete_thread(thread.id)
            await client.agents.delete_agent(agent.id)

if __name__ == "__main__":
    asyncio.run(main())
