from flask import Flask, render_template, request, jsonify, session
import os
import json
import uuid
import numpy as np
import faiss
import pickle
import time
import re
import base64
from typing import Dict, Any, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
import threading

# Import TTS module
from tts_module import ElevenLabsTTS


# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Global variables
VECTOR_DB_DIR = "vector_db"  # Default vector database directory
REFERENCE_FILE = None  # Default reference file (None)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyAC8xILnP67xEmC6tOuBYzXkye6Dqa1vbM"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") or "sk_73622fba70eec59749900a0e53a2a52c94fed6b087a11d33"
MODEL_NAME = "all-mpnet-base-v2"  # Default embedding model
TOP_K = 5  # Default number of chunks to retrieve
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Default ElevenLabs voice ID
ID_PREFIX = "Q"  # Default query ID prefix

# Initialize global chatbot instance
chatbot = None
loading_complete = False

class WebContextualChatbot:
    def __init__(self, vector_db_dir: str, model_name: str = "all-mpnet-base-v2", 
                 top_k: int = 5, reference_file: str = None, 
                 gemini_api_key: str = None, elevenlabs_api_key: str = None,
                 id_prefix: str = "Q"):
        """Initialize the Web Contextual RAG Chatbot
        
        Args:
            vector_db_dir: Path to the vector database directory
            model_name: Name of the sentence transformer model
            top_k: Number of chunks to retrieve
            reference_file: Path to reference QA data file (optional)
            gemini_api_key: Gemini API key
            elevenlabs_api_key: ElevenLabs API key
            id_prefix: Prefix for the query IDs (default: "Q")
        """
        self.top_k = top_k
        self.reference_data = None
        self.id_prefix = id_prefix
        self.query_counter = 0  # For sequential IDs
        self.query_history = []
        self.user_sessions = {}  # Track session data for multiple users
        
        # Set up Gemini API
        self.gemini_api_key = gemini_api_key
        if not self.gemini_api_key:
            raise ValueError("No Gemini API key provided")
        
        # Configure Gemini API
        genai.configure(api_key=self.gemini_api_key)
        
        # Initialize TTS module
        self.elevenlabs_api_key = elevenlabs_api_key
        self.tts_enabled = False
        try:
            self.tts = ElevenLabsTTS(api_key=self.elevenlabs_api_key)
            self.tts_enabled = True
            print("Text-to-Speech enabled with ElevenLabs")
        except Exception as e:
            print(f"Warning: Text-to-Speech could not be enabled: {e}")
            print("Running without voice output")
        
        # Check if vector DB directory exists
        if not os.path.exists(vector_db_dir):
            raise ValueError(f"Vector database directory {vector_db_dir} not found.")
        
        # Load chunks metadata
        chunks_meta_path = os.path.join(vector_db_dir, "chunks_meta.pkl")
        if not os.path.exists(chunks_meta_path):
            raise ValueError(f"Chunks metadata file not found in {vector_db_dir}.")
        
        with open(chunks_meta_path, "rb") as f:
            self.chunks_meta = pickle.load(f)
        
        # Load sections information
        sections_path = os.path.join(vector_db_dir, "sections.pkl")
        if os.path.exists(sections_path):
            with open(sections_path, "rb") as f:
                self.sections = pickle.load(f)
        else:
            self.sections = {}
        
        # Load FAISS index
        index_path = os.path.join(vector_db_dir, "embeddings.index")
        if not os.path.exists(index_path):
            raise ValueError(f"FAISS index file not found in {vector_db_dir}.")
        
        self.index = faiss.read_index(index_path)
        
        print(f"Loaded vector database from {vector_db_dir}:")
        print(f"- {len(self.chunks_meta)} chunks metadata")
        print(f"- {len(self.sections)} sections information")
        print(f"- FAISS index with {self.index.ntotal} vectors")
        
        # Initialize embedding model
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Load reference data if provided
        if reference_file and os.path.exists(reference_file):
            with open(reference_file, 'r', encoding='utf-8') as f:
                self.reference_data = json.load(f)
                print(f"Loaded {len(self.reference_data)} reference QA pairs")
                
                # Extract existing query IDs from reference data if available
                for item in self.reference_data:
                    if "query_id" in item:
                        try:
                            # If query_id is numeric, update counter to avoid duplicates
                            if item["query_id"].isdigit():
                                self.query_counter = max(self.query_counter, int(item["query_id"]))
                        except (ValueError, TypeError):
                            pass
                
                # Build reference index if we have reference data
                self.build_reference_index()
        
        # Initialize Gemini model
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        print("Initialized Gemini 1.5 Flash model")
    
    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        """Get session data for a user, creating it if it doesn't exist"""
        if session_id not in self.user_sessions:
            self.user_sessions[session_id] = {
                "chat": self.model.start_chat(history=[]),
                "last_question": None,
                "last_answer": None,
                "last_context": None,
                "query_history": []
            }
        return self.user_sessions[session_id]
    
    def generate_query_id(self) -> str:
        """Generate a unique query ID
        
        Returns:
            A unique query ID string
        """
        # Increment counter for sequential IDs
        self.query_counter += 1
        
        # Format: prefix + sequential number (e.g., Q1, Q2, Q3...)
        query_id = f"{self.id_prefix}{self.query_counter}"
        
        return query_id
    
    def build_reference_index(self):
        """Build FAISS index for reference QA data"""
        if not self.reference_data:
            self.ref_index = None
            return
            
        print("Building FAISS index for reference questions...")
        
        # Extract questions and encode them
        self.ref_questions = [qa.get("question", "") for qa in self.reference_data if qa.get("question")]
        
        # If no valid questions, skip
        if not self.ref_questions:
            self.ref_index = None
            return
        
        # Encode all questions
        ref_embeddings = []
        for question in self.ref_questions:
            embedding = self.embedding_model.encode(question)
            ref_embeddings.append(embedding)
        
        # Convert to numpy array
        self.ref_embeddings = np.array(ref_embeddings).astype('float32')
        
        # Create a FAISS index for reference questions
        embedding_dim = self.ref_embeddings.shape[1]
        self.ref_index = faiss.IndexFlatIP(embedding_dim)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(self.ref_embeddings)
        
        # Add vectors to the index
        self.ref_index.add(self.ref_embeddings)
        
        print(f"Reference FAISS index built with {len(self.ref_questions)} questions")
    
    def find_similar_question(self, query: str) -> Dict[str, Any]:
        """Find the most similar question in the reference data"""
        if not self.reference_data or not hasattr(self, 'ref_index') or self.ref_index is None:
            return None
        
        # Encode the query
        query_embedding = self.embedding_model.encode(query).astype('float32').reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search the reference FAISS index
        similarities, indices = self.ref_index.search(query_embedding, 1)
        
        # Threshold for similarity
        if similarities[0][0] > 0.8:
            idx = indices[0][0]
            if idx < len(self.reference_data):
                return self.reference_data[idx]
        
        return None
    
    def is_follow_up_question(self, query: str, session_data: Dict[str, Any]) -> bool:
        """Determine if the query is a follow-up to the previous question
        
        Args:
            query: The current query
            session_data: The user's session data
            
        Returns:
            True if the query is likely a follow-up question
        """
        if not session_data["last_question"] or not session_data["last_answer"]:
            return False
            
        # Check for pronouns that might indicate a follow-up
        pronoun_pattern = r'\b(it|this|that|these|those|they|them|he|she|his|her|its|their)\b'
        if re.search(pronoun_pattern, query.lower()):
            return True
            
        # Check for very short questions that are likely follow-ups
        if len(query.split()) <= 5:
            return True
            
        # Check if the query doesn't contain main subject/topic
        # First, try to extract main topics from last question using simple NLP
        # This is a simplified approach - for production, use a proper NLP library
        last_question_nouns = self._extract_nouns(session_data["last_question"])
        query_nouns = self._extract_nouns(query)
        
        # If the query doesn't contain any new nouns, it's likely a follow-up
        if not query_nouns or all(noun in last_question_nouns for noun in query_nouns):
            return True
            
        return False
    
    def _extract_nouns(self, text: str) -> List[str]:
        """Extract potential nouns from text using a simple approach"""
        # Remove common words and get potential nouns (words that aren't common verbs or prepositions)
        common_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'when', 'where', 'how', 'why',
                       'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                       'do', 'does', 'did', 'to', 'at', 'in', 'on', 'for', 'with', 'about', 'from'}
        
        words = text.lower().split()
        potential_nouns = [word for word in words if word not in common_words and len(word) > 2]
        return potential_nouns
    
    def expand_query_with_context(self, query: str, session_data: Dict[str, Any]) -> Tuple[str, bool]:
        """Expand a follow-up query with conversation context
        
        Args:
            query: The current query
            session_data: The user's session data
            
        Returns:
            Tuple of (expanded_query, is_follow_up)
        """
        # Check if this is likely a follow-up question
        if not self.is_follow_up_question(query, session_data):
            return query, False
            
        # Use Gemini to expand the query with context
        prompt = f"""You are helping to expand a follow-up question with context from the previous question and answer.

Previous question: {session_data["last_question"]}
Previous answer: {session_data["last_answer"]}

Follow-up question: {query}

Please rewrite the follow-up question as a standalone question that includes all necessary context.
Only return the rewritten question, without any explanations or additional text.
Make the rewritten question clear and specific enough to be answered without seeing the previous context.
"""
        
        try:
            response = self.model.generate_content(prompt)
            expanded_query = response.text.strip()
            
            # If expansion failed or made the query worse, return original
            if not expanded_query or len(expanded_query) < len(query):
                return query, True
                
            return expanded_query, True
        except Exception as e:
            print(f"Query expansion failed: {e}")
            return query, True
    
    def find_relevant_chunks(self, query: str) -> List[Dict[str, Any]]:
        """Find the most relevant document chunks for a query"""
        # Encode the query
        query_embedding = self.embedding_model.encode(query).astype('float32').reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search the FAISS index
        k = min(self.top_k * 2, len(self.chunks_meta))  # Get more than needed for filtering
        similarities, indices = self.index.search(query_embedding, k)
        
        # Get the chunks corresponding to the indices
        relevant_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks_meta):
                chunk = self.chunks_meta[idx].copy()
                chunk["similarity"] = float(similarities[0][i])
                relevant_chunks.append(chunk)
        
        # Apply filtering to improve diversity and relevance
        filtered_chunks = self.filter_chunks(relevant_chunks)
        
        return filtered_chunks[:self.top_k]
    
    def filter_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and rerank chunks for better diversity and relevance"""
        if not chunks:
            return []
        
        # First, sort by similarity
        sorted_chunks = sorted(chunks, key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Ensure we have at least one heading if available
        final_chunks = []
        seen_sections = set()
        heading_added = False
        
        # First pass: Add high-relevance items and ensure diversity
        for chunk in sorted_chunks:
            section = chunk.get("section", "Unknown")
            
            # Always include the most relevant chunk
            if not final_chunks:
                final_chunks.append(chunk)
                seen_sections.add(section)
                if chunk.get("is_heading", False):
                    heading_added = True
                continue
            
            # Try to add a heading if we don't have one yet
            if not heading_added and chunk.get("is_heading", False):
                final_chunks.append(chunk)
                seen_sections.add(section)
                heading_added = True
                continue
            
            # Add chunks from unseen sections first (for diversity)
            if section not in seen_sections and len(final_chunks) < self.top_k:
                final_chunks.append(chunk)
                seen_sections.add(section)
                continue
            
            # If we still need chunks and this one is highly relevant, add it
            if len(final_chunks) < self.top_k and chunk.get("similarity", 0) > 0.7:
                final_chunks.append(chunk)
        
        # If we still need more chunks, add the remaining most relevant ones
        remaining_needed = self.top_k - len(final_chunks)
        if remaining_needed > 0:
            # Find chunks not already included
            remaining_chunks = [c for c in sorted_chunks if c not in final_chunks]
            final_chunks.extend(remaining_chunks[:remaining_needed])
        
        # Sort final chunks by similarity again for consistency
        return sorted(final_chunks, key=lambda x: x.get("similarity", 0), reverse=True)
    
    def extract_sections_and_pages(self, relevant_chunks: List[Dict[str, Any]], similar_qa: Dict[str, Any] = None) -> Dict[str, List[str]]:
        """Extract sections and pages from relevant chunks or reference data"""
        if similar_qa and "references" in similar_qa:
            return {
                "sections": similar_qa["references"].get("sections", []),
                "pages": similar_qa["references"].get("pages", [])
            }
        
        # Extract sections and pages from chunks
        sections = []
        pages = []
        
        for chunk in relevant_chunks:
            # Extract section
            section = chunk.get("section", "")
            if section and section not in sections and section != "Unknown Section":
                sections.append(section)
            
            # Extract page
            page = chunk.get("page", None)
            if page and str(page) not in pages:
                pages.append(str(page))
        
        # If we have section names, look up their page ranges in the sections map
        if sections and self.sections:
            for section in sections:
                if section in self.sections:
                    for page in self.sections[section]:
                        if str(page) not in pages:
                            pages.append(str(page))
        
        # Sort and limit to reasonable number
        sections = sorted(sections[:5])
        pages = sorted(pages, key=lambda x: int(x) if x.isdigit() else 0)
        
        return {
            "sections": sections,
            "pages": pages
        }
    
    def format_context_for_llm(self, relevant_chunks: List[Dict[str, Any]], is_follow_up: bool = False, 
                              session_data: Dict[str, Any] = None) -> str:
        """Format relevant chunks into a context for the LLM"""
        context_parts = []
        
        # First, add headings
        for chunk in relevant_chunks:
            if chunk.get("is_heading", False):
                heading_text = chunk["text"].replace("[HEADING] ", "")
                context_parts.append(f"SECTION: {heading_text}")
        
        # Then add content with page numbers and metadata
        for i, chunk in enumerate(relevant_chunks):
            if not chunk.get("is_heading", False):
                page_info = f"Page {chunk['page']}"
                section_info = f"Section: {chunk.get('section', 'Unknown Section')}"
                similarity = chunk.get("similarity", 0)
                
                context_parts.append(
                    f"[Excerpt {i+1}] ({page_info}, {section_info}, Relevance: {similarity:.2f})\n{chunk['text']}\n"
                )
        
        # For follow-up questions, add previous context
        if is_follow_up and session_data and session_data["last_context"]:
            context_parts.append("\nRELEVANT PREVIOUS CONTEXT:")
            context_parts.append(f"Previous Question: {session_data['last_question']}")
            context_parts.append(f"Previous Answer: {session_data['last_answer']}")
            context_parts.append(session_data["last_context"])
        
        return "\n\n".join(context_parts)
    
    def generate_tts(self, text: str, voice_id: str = None) -> Optional[str]:
        """Generate TTS audio for a text and return base64 encoded audio data
        
        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID (if None, uses default)
            
        Returns:
            Base64 encoded audio data, or None if TTS is not enabled
        """
        if not self.tts_enabled:
            return None
        
        try:
            # Use specified voice ID or default
            voice_id = voice_id or VOICE_ID
            
            # Generate audio file
            audio_path = self.tts.text_to_speech(text, voice_id=voice_id)
            
            # Read audio file and encode as base64
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
                base64_audio = base64.b64encode(audio_data).decode("utf-8")
            
            return base64_audio
        except Exception as e:
            print(f"Error during TTS generation: {e}")
            return None
    
    def process_stt(self, audio_data: str) -> str:
        """Process speech-to-text from base64 audio data
        
        Args:
            audio_data: Base64 encoded audio data
            
        Returns:
            Transcribed text
        """
        try:
            # Decode base64 audio data
            binary_audio = base64.b64decode(audio_data)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(binary_audio)
            
            # Use Google Speech Recognition
            import speech_recognition as sr
            r = sr.Recognizer()
            
            # Convert webm to wav (if needed)
            import subprocess
            wav_filename = temp_filename + ".wav"
            subprocess.call(['ffmpeg', '-i', temp_filename, wav_filename], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
            
            # Transcribe audio
            with sr.AudioFile(wav_filename) as source:
                audio = r.record(source)
                text = r.recognize_google(audio)
            
            # Clean up temporary files
            os.remove(temp_filename)
            os.remove(wav_filename)
            
            return text
        except Exception as e:
            print(f"Error during STT processing: {e}")
            return ""
    
    def generate_response(self, query: str, session_id: str) -> Dict[str, Any]:
        """Generate structured response to the query with a unique ID, handling follow-up questions
        
        Args:
            query: The user's query
            session_id: The user's session ID
            
        Returns:
            Structured response with query ID, question, context, answer, and references
        """
        # Get session data
        session_data = self.get_session_data(session_id)
        
        # Generate a query ID
        query_id = self.generate_query_id()
        
        # Check if this is a follow-up question and expand it if needed
        expanded_query, is_follow_up = self.expand_query_with_context(query, session_data)
        
        # Use expanded query for processing but keep original for display
        processing_query = expanded_query
        display_query = query
        
        # Check if there's a similar question in the reference data
        similar_qa = self.find_similar_question(processing_query)
        
        # If we found a very similar question, use its ID if available
        reference_id = None
        if similar_qa:
            reference_id = similar_qa.get("query_id")
        
        # Use reference ID if available, otherwise use our generated ID
        final_id = reference_id or query_id
        
        # Find relevant document chunks
        relevant_chunks = self.find_relevant_chunks(processing_query)
        
        # Extract raw context for the response
        raw_context = "\n".join([chunk.get("text", "") for chunk in relevant_chunks])
        context = raw_context[:500] + "..." if len(raw_context) > 500 else raw_context
        
        # Format context for LLM consumption
        formatted_context = self.format_context_for_llm(relevant_chunks, is_follow_up, session_data)
        
        # Extract sections and pages
        references = self.extract_sections_and_pages(relevant_chunks, similar_qa)
        
        # Generate answer
        if similar_qa and "answer" in similar_qa and not is_follow_up:
            # Use reference answer if very similar question was found and not a follow-up
            answer = similar_qa["answer"]
        else:
            # Prepare the prompt for Gemini
            examples = ""
            if self.reference_data:
                # Include example answers to guide the format
                for i, qa in enumerate(self.reference_data[:3]):
                    if "answer" in qa:
                        examples += f"Example {i+1}: \"{qa['answer']}\"\n"
            
            prompt = f"""You are an expert educational assistant answering questions with high precision.
Answer the following query based ONLY on the context provided below.

{examples}
Your answer should be:
1. Brief and concise (one or two sentences)
2. Factual and directly extracted from the context
3. Formatted as a summary of key points
4. Without any introductory phrases like "According to the context..."

CONTEXT:
{formatted_context}

QUERY: {processing_query}

BRIEF ANSWER:"""
            
            # Generate response
            try:
                # Try with generation config (for newer library versions)
                generation_config = {"temperature": 0.2}
                response = session_data["chat"].send_message(prompt, generation_config=generation_config)
            except TypeError:
                # Fall back to default if temperature parameter not supported
                response = session_data["chat"].send_message(prompt)
                
            answer = response.text.strip()
        
        # Generate TTS for answer if enabled
        audio_data = self.generate_tts(answer) if self.tts_enabled else None
        
        # Create structured response with query ID
        structured_response = {
            "query_id": final_id,
            "question": display_query,  # Use original query for display
            "context": context,
            "answer": answer,
            "references": references,
            "timestamp": int(time.time()),
            "is_follow_up": is_follow_up,
            "audio_data": audio_data  # Include TTS audio data if available
        }
        
        # Update conversation context
        session_data["last_question"] = display_query
        session_data["last_answer"] = answer
        session_data["last_context"] = formatted_context
        
        # Add to query history
        session_data["query_history"].append(structured_response)
        
        return structured_response
    
    def clear_session(self, session_id: str):
        """Clear session data for a user"""
        if session_id in self.user_sessions:
            # Create a new chat instance to clear history
            self.user_sessions[session_id]["chat"] = self.model.start_chat(history=[])
            self.user_sessions[session_id]["last_question"] = None
            self.user_sessions[session_id]["last_answer"] = None
            self.user_sessions[session_id]["last_context"] = None
            # Keep the query history for reference

# Initialize chatbot in a separate thread to avoid blocking the app startup
def init_chatbot():
    global chatbot, loading_complete
    try:
        print("Initializing chatbot...")
        chatbot = WebContextualChatbot(
            vector_db_dir=VECTOR_DB_DIR,
            model_name=MODEL_NAME,
            top_k=TOP_K,
            reference_file=REFERENCE_FILE,
            gemini_api_key=GEMINI_API_KEY,
            elevenlabs_api_key=ELEVENLABS_API_KEY,
            id_prefix=ID_PREFIX
        )
        loading_complete = True
        print("Chatbot initialization complete!")
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        loading_complete = False

# Start initialization in a background thread
threading.Thread(target=init_chatbot).start()

# Flask routes
@app.route('/')
def index():
    """Render the main page"""
    # Create a unique session ID if not already present
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    return render_template('index.html', loading=not loading_complete)

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process a query and return the response"""
    if not loading_complete or not chatbot:
        return jsonify({'error': 'Chatbot is still initializing. Please wait.'}), 503
    
    # Get session ID
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    session_id = session['user_id']
    
    # Get query from request
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    query = data['query']
    
    # Process the query
    try:
        response = chatbot.generate_response(query, session_id)
        return jsonify(response)
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_context():
    """Clear the conversation context for the current session"""
    if not loading_complete or not chatbot:
        return jsonify({'error': 'Chatbot is still initializing. Please wait.'}), 503
    
    # Get session ID
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    session_id = session['user_id']
    
    # Clear the session
    try:
        chatbot.clear_session(session_id)
        return jsonify({'status': 'success', 'message': 'Conversation context cleared'})
    except Exception as e:
        print(f"Error clearing context: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get query history for the current session"""
    if not loading_complete or not chatbot:
        return jsonify({'error': 'Chatbot is still initializing. Please wait.'}), 503
    
    # Get session ID
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    session_id = session['user_id']
    
    # Get session data
    try:
        session_data = chatbot.get_session_data(session_id)
        return jsonify({'history': session_data['query_history']})
    except Exception as e:
        print(f"Error getting history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stt', methods=['POST'])
def process_stt():
    """Process speech-to-text from audio data"""
    if not loading_complete or not chatbot:
        return jsonify({'error': 'Chatbot is still initializing. Please wait.'}), 503
    
    # Get audio data from request
    data = request.get_json()
    if not data or 'audio' not in data:
        return jsonify({'error': 'No audio data provided'}), 400
    
    audio_data = data['audio']
    
    # Process the audio
    try:
        text = chatbot.process_stt(audio_data)
        return jsonify({'text': text})
    except Exception as e:
        print(f"Error processing speech-to-text: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/check_status', methods=['GET'])
def check_status():
    """Check if the chatbot is initialized"""
    return jsonify({'status': 'ready' if loading_complete else 'loading'})

if __name__ == "__main__":
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)