# YouTube QA Chatbot

A chatbot that answers questions about YouTube videos based on their transcripts.

## Features

- Process multiple YouTube videos at once
- Answer questions based solely on video content
- Web interface with Streamlit
- Transcript caching to avoid repeated API calls

## Setup

1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Create a `.env` file with the following variables:
   - OPENAI_API_KEY=your_openai_api_key
   - LANGCHAIN_API_KEY=your_langchain_api_key
   - LANGCHAIN_PROJECT=your_langchain_project
   - YOUTUBE_API_KEY=your_youtube_api_key


## Usage

### Web Interface

Run the Streamlit app: streamlit run app.py


## How It Works

1. The application extracts transcripts from YouTube videos
2. Transcripts are chunked and stored in a vector database (ChromaDB)
3. When a question is asked, the application:
   - Retrieves relevant chunks from the vector database
   - Uses an LLM to generate an answer based only on the retrieved content
   - Ensures answers are strictly based on video content

## Project Structure

- `main.py`: Command-line interface
- `app.py`: Streamlit web interface
- `agents/qa_agent.py`: QA agent implementation
- `tools/youtube_tool.py`: YouTube transcript extraction
- `tools/chromadb_tool.py`: Vector database operations
- `tools/utils.py`: Utility functions

## Limitations

- Only works with videos that have transcripts available
- Quality of answers depends on the quality of the transcripts
- Currently only supports English language videos
