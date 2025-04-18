# Update imports to include OpenAI
from langgraph.graph import StateGraph
from tools.youtube_tool import get_youtube_transcript
from tools.chromadb_tool import store_embeddings
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from typing import Dict, Any, TypedDict
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from agents.qa_agent import create_qa_agent

# Load environment variables from .env file
load_dotenv()

# Set environment variables for LangChain and OpenAI
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Set environment variables
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["YOUTUBE_API_KEY"] = YOUTUBE_API_KEY

# Define the state type for the graph using TypedDict instead of Dict
class GraphState(TypedDict, total=False):
    """
    State for the video processing graph.
    
    Attributes:
        urls (list): List of YouTube URLs to process
        all_chunks (list): All text chunks from video transcripts
        vector_store (Any): Vector store containing embeddings
        agent (Any): QA agent for answering questions
        conversation_history (list): Store conversation history
    """
    urls: list
    all_chunks: list
    vector_store: Any
    agent: Any
    conversation_history: list

def get_user_query():
    """
    Function to prompt the user for input.
    
    Returns:
        str: User's query
    """
    print("\nPlease enter your query about the videos (type 'exit' to quit):")
    query = input("Query: ")
    return query

def process_videos_node(state: Dict) -> Dict:
    """
    Node to process multiple videos and combine their transcripts.
    
    Args:
        state (Dict): Current state containing URLs
        
    Returns:
        Dict: Updated state with transcript chunks
        
    Raises:
        ValueError: If no URLs are found in the state
    """
    print(f"State at start of process_videos_node: {state}")
    
    # Ensure that the 'urls' field is correctly passed
    urls = state.get("urls", [])
    if not urls:
        raise ValueError("No URLs found in state.")
    
    all_chunks = []
    
    for url in urls:
        try:
            print(f"Processing video: {url}")
            transcript = get_youtube_transcript(url)
            
            # Create chunks from transcript
            for i in range(0, len(transcript), 100):
                chunk_text = " ".join(transcript[i:i+100])
                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata={"source": url}
                ))
            
            print(f"Successfully processed video: {url}")
        except Exception as e:
            print(f"Error processing video {url}: {e}")
            print("Continuing with other videos...")
    
    # Create a new state dictionary instead of modifying the existing one
    return {"urls": urls, "all_chunks": all_chunks}

def store_embeddings_node(state: Dict) -> Dict:
    """
    Node to store embeddings in ChromaDB.
    
    Args:
        state (Dict): Current state containing transcript chunks
        
    Returns:
        Dict: Updated state with vector store
        
    Raises:
        ValueError: If no valid chunks are found
    """
    print(f"State at start of store_embeddings_node: {state}")
    
    all_chunks = state.get("all_chunks", [])
    if not all_chunks:
        raise ValueError("No valid chunks found from any videos")
    
    # Store embeddings in ChromaDB
    collection_name = "multiple_videos_collection"
    vector_store = store_embeddings(all_chunks, collection_name=collection_name)
    
    # Create a new state dictionary with all previous keys plus the new one
    return {**state, "vector_store": vector_store}

def create_agent_node(state: Dict) -> Dict:
    """
    Node to create QA agent with vector store.
    
    Args:
        state (Dict): Current state containing vector store
        
    Returns:
        Dict: Updated state with QA agent
        
    Raises:
        ValueError: If no vector store is found in the state
    """
    print(f"State at start of create_agent_node: {state}")
    
    vector_store = state.get("vector_store")
    if not vector_store:
        raise ValueError("No vector store found in state.")
    
    # Create QA agent using the imported function
    agent_executor, conversation_history = create_qa_agent(vector_store)
    
    # Create a new state dictionary with all previous keys plus the new ones
    return {**state, "agent": agent_executor, "conversation_history": conversation_history}

def build_graph_and_agent(urls):
    """
    Build the graph and agent for the Streamlit app.
    
    Args:
        urls (list): List of YouTube URLs to process
        
    Returns:
        AgentExecutor: QA agent for answering questions
        
    Raises:
        Exception: If there's an error building the agent
    """
    try:
        # Create LangGraph builder with state schema
        builder = StateGraph(GraphState)
        
        # Add nodes to the graph
        builder.add_node("process_videos", process_videos_node)
        builder.add_node("store_embeddings", store_embeddings_node)
        builder.add_node("create_agent", create_agent_node)
        
        # Connect nodes in the graph
        builder.set_entry_point("process_videos")
        builder.add_edge("process_videos", "store_embeddings")
        builder.add_edge("store_embeddings", "create_agent")
        
        # Compile the graph
        graph = builder.compile()
        
        # Execute the graph with the initial state
        final_state = graph.invoke({"urls": urls})
        
        # Get the agent from the final state
        qa_agent = final_state["agent"]
        
        return qa_agent
        
    except Exception as e:
        raise Exception(f"Error building agent: {e}")

def main():
    """
    Main function to run the CLI version of the YouTube QA Bot.
    """
    print("\n=== YouTube QA Bot ===")
    print("1. Process videos and ask questions")
    print("2. Clear transcript cache")
    print("3. Clear vector databases")
    print("4. Clear all caches and databases")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ")
    
    if choice == "1":
        # Get list of YouTube URLs
        urls = []
        print("Enter YouTube URLs (one per line, type 'done' when finished):")
        
        while True:
            url = input()
            if url.lower() in ['done', 'exit', 'quit']:
                break
            urls.append(url)
        
        if not urls:
            print("No URLs provided. Exiting.")
            return
        
        try:
            # Create LangGraph builder with state schema
            builder = StateGraph(GraphState)
            
            # Add nodes to the graph
            builder.add_node("process_videos", process_videos_node)
            builder.add_node("store_embeddings", store_embeddings_node)
            builder.add_node("create_agent", create_agent_node)
            
            # Connect nodes in the graph
            builder.set_entry_point("process_videos")
            builder.add_edge("process_videos", "store_embeddings")
            builder.add_edge("store_embeddings", "create_agent")
            
            # Compile the graph
            graph = builder.compile()
            
            # Execute the graph with the initial state
            final_state = graph.invoke({"urls": urls})
            
            # Get the agent and conversation history from the final state
            qa_agent = final_state["agent"]
            conversation_history = final_state["conversation_history"]
            
            # Define system prompt to explain the task
            system_prompt = """
            Welcome to the Video QA Bot!
            You can ask questions related to the content of the videos you provided.
            To quit, just type 'exit'.
            """
            
            print(system_prompt)
            
            # Main conversation loop
            while True:
                # Get user query
                query = get_user_query()
                
                # If user types 'exit', break out of the loop
                if query.lower() == 'exit':
                    print("Goodbye!")
                    break
                
                # Update the conversation history with the new user query
                conversation_history.append({"role": "user", "content": query})
                
                # Get response from the QA agent, including the conversation history
                try:
                    response = qa_agent.invoke({
                        "input": query,
                        "conversation_history": conversation_history
                    })
                    
                    # Print the answer
                    if "output" in response:
                        print(f"\nAnswer: {response['output']}\n")
                    else:
                        print(f"\nAnswer: {response}\n")
                    
                    # Update the conversation history with the agent's response
                    conversation_history.append({"role": "assistant", "content": response['output']})
                except Exception as e:
                    print(f"\nError getting response: {e}\n")
                    print("Please try a different question.")
                
        except Exception as e:
            print(f"Error in processing: {e}")
            print("Please try different video URLs.")

if __name__ == "__main__":
    main()
