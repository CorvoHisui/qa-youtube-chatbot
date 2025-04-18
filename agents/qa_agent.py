from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import RetrievalQA

def create_qa_chain(vector_store):
    """
    Create a retrieval QA chain for answering questions based on video content.
    
    Args:
        vector_store: Vector store containing video transcript embeddings
        
    Returns:
        RetrievalQA: QA chain for retrieving and answering questions
    """
    # Use OpenAI for processing
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4-turbo-preview",
        max_tokens=1024
    )
    
    # Increase k to get more context from the videos
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})
    
    # Create a QA chain that retrieves relevant information
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )
    
    return qa_chain

def create_strict_qa_tool(qa_chain):
    """
    Create a tool that ensures answers come only from video content.
    
    Args:
        qa_chain: RetrievalQA chain for answering questions
        
    Returns:
        Tool: Tool for answering questions based only on video content
    """
    def strict_qa_tool(query):
        """
        Tool that ensures answers come only from video content.
        
        Args:
            query (str): User's question
            
        Returns:
            str: Answer based only on video content
        """
        try:
            result = qa_chain.invoke({"query": query})
            
            # Check if any documents were retrieved
            if not result.get("source_documents") or len(result.get("source_documents", [])) == 0:
                return "I don't have information about this in the video content."
            
            # Return only the result without source documents to the user
            return result["result"]
        except Exception as e:
            print(f"Error in QA tool: {e}")
            return "I couldn't find specific information about that in the video content."
    
    # Create a tool for the agent to use
    return Tool(
        name="video_transcript_qa",
        func=strict_qa_tool,
        description="ALWAYS use this tool to answer questions about the video content. This is the ONLY source of information you have."
    )

def create_qa_agent(vector_store):
    """
    Create a QA agent for answering questions about video content.
    
    Args:
        vector_store: Vector store containing video transcript embeddings
        
    Returns:
        AgentExecutor: Agent for answering questions
        list: Empty conversation history
    """
    # Create QA chain
    qa_chain = create_qa_chain(vector_store)
    
    # Create tool
    retrieval_tool = create_strict_qa_tool(qa_chain)
    tools = [retrieval_tool]
    
    # Use OpenAI for agent
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4-turbo-preview",
        max_tokens=1024
    )
    
    # Create a system prompt that enforces strict adherence to video content
    system_prompt = """You are a specialized assistant that ONLY answers questions based on the transcripts of provided YouTube videos.

CRITICAL RULES YOU MUST FOLLOW:
1. You have NO knowledge beyond what is in the video transcripts.
2. You can ONLY provide information that is EXPLICITLY mentioned in the video transcripts.
3. If the information is not in the transcripts, you MUST respond with EXACTLY: "I don't have that information in the video content."
4. You MUST use the video_transcript_qa tool for EVERY question without exception.
5. NEVER make up information or use general knowledge.
6. If asked about topics unrelated to the videos, respond with EXACTLY: "I can only answer questions about the content of the provided videos."
7. Do not reference external sources, websites, or any information not in the videos.
8. Do not offer opinions or interpretations beyond what is directly stated in the videos.

Your ONLY purpose is to retrieve and provide information from the video transcripts."""
    
    # Create the prompt template for the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="conversation_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the OpenAI functions agent
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate",
        return_intermediate_steps=False
    )
    
    # Return the agent and an empty conversation history
    return agent_executor, []