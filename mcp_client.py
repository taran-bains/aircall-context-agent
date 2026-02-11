"""MCP Client to test the server."""
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    """Test MCP server with sample queries."""
    print("🔌 Connecting to MCP server...")
    
    # Server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            print("✅ Connected to server\n")
            
            # List available tools
            tools = await session.list_tools()
            print("📋 Available tools:")
            for tool in tools.tools:
                print(f"   - {tool.name}: {tool.description[:50]}...")
            
            print("\n" + "="*60)
            
            # Test search_calls
            print("\n🔍 Testing search_calls tool:")
            query = "What billing issues were reported?"
            print(f"   Query: {query}")
            
            result = await session.call_tool(
                "search_calls",
                {"query": query}
            )
            
            print(f"\n   Result:")
            for content in result.content:
                print(f"   {content.text}")
            
            print("\n" + "="*60)
            
            # Test get_call_stats
            print("\n📊 Testing get_call_stats tool:")
            result = await session.call_tool(
                "get_call_stats",
                {"stat_type": "by_agent"}
            )
            
            print(f"\n   Result:")
            for content in result.content:
                print(f"   {content.text}")


if __name__ == "__main__":
    print("🤖 MCP Client Test\n")
    asyncio.run(main())
