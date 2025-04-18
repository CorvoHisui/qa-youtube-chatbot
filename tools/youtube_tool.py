import os
import json
from youtube_transcript_api import YouTubeTranscriptApi
from tools.utils import extract_video_id

# File to cache transcripts to avoid repeated API calls
CACHE_FILE = "transcript_cache.json"

# Load existing cache if available
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        transcript_cache = json.load(f)
else:
    transcript_cache = {}

def get_youtube_transcript(url):
    """
    Get the transcript for a YouTube video.
    Uses cache if available, otherwise fetches from YouTube.
    
    Args:
        url (str): YouTube URL
        
    Returns:
        list: List of transcript text segments
    """
    video_id = extract_video_id(url)
    if not video_id:
        print(f"Could not extract video ID from URL: {url}")
        return []
        
    if video_id in transcript_cache:
        print(f"[Cache] Transcript for {video_id} loaded from cache.")
        return transcript_cache[video_id]
    
    try:
        # Get transcript using the YouTubeTranscriptApi
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Extract just the text from each transcript segment
        transcript = [item['text'] for item in transcript_list]
        
        # Cache the transcript for future use
        transcript_cache[video_id] = transcript
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(transcript_cache, f, ensure_ascii=False, indent=2)
        
        return transcript
    except Exception as e:
        print(f"Error getting transcript: {e}")
        return []

def clear_transcript_cache():
    """
    Clear the transcript cache file.
    
    Returns:
        bool: True if cache was cleared, False if no cache was found
    """
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print("Transcript cache cleared.")
        # Reset the in-memory cache as well
        global transcript_cache
        transcript_cache = {}
        return True
    else:
        print("No transcript cache found.")
        return False
