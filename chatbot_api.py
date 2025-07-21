import os
import sqlite3
import logging
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import signal
import sys
import threading
from flask import send_from_directory  # Add this line
import subprocess
import threading
import time
import atexit

# Core imports - same as debug script
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
COLLECTION_NAME = "kb_collection"
SQLITE_PATH = "./file_index.db"  # Same as sync.py
SYNC_BATCH_FILE = r"C:\Users\maria selciya\Desktop\sync_loop.bat"  # Update path as needed
SYNC_ENABLED = True  # Set to False to disable auto-sync

# Validate environment variables
def validate_environment():
    """Validate all required environment variables"""
    required_vars = ['QDRANT_URL', 'QDRANT_API_KEY', 'GOOGLE_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing environment variables: {missing_vars}")
    
    logger.info("✅ All environment variables validated")

class KnowledgeBaseManager:
    """Manager for knowledge base file operations - extracted from sync.py"""
    
    def __init__(self):
        self.conn = None
        self.cursor = None
    
    def init_database_connection(self):
        try:
            # Close existing connection if any
            if self.conn:
                self.conn.close()
            
            if not os.path.exists(SQLITE_PATH):
                logger.warning(f"⚠️ Database file not found: {SQLITE_PATH}")
                return False
        
            # Use check_same_thread=False for Flask threading
            self.conn = sqlite3.connect(SQLITE_PATH, timeout=30, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable dict-like access
            self.cursor = self.conn.cursor()
        
            # Test connection
            self.cursor.execute("SELECT COUNT(*) FROM file_index")
            logger.info(f"✅ Database connected successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False

    def ensure_connection(self):
        """Ensure database connection is active, reconnect if needed"""
        try:
            if not self.conn:
                return self.init_database_connection()
        
            # Test connection
            self.cursor.execute("SELECT 1")
            return True
        except (sqlite3.Error, AttributeError):
            logger.warning("🔄 Database connection lost, reconnecting...")
            return self.init_database_connection()
    
    def close_database_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def get_database_files(self):
        """Get all files from the database with detailed information"""
        try:
            if not self.ensure_connection():
                return {"error": "Database connection failed", "success": False}
            
            # Get all files with their metadata
            self.cursor.execute("""
                SELECT filename, filepath, file_type, file_size, mtime, hash, 
                       chunk_count, indexed_at 
                FROM file_index 
                ORDER BY indexed_at DESC
            """)
            files_data = self.cursor.fetchall()
            
            # Format the data
            files = []
            total_chunks = 0
            total_size = 0
            
            for row in files_data:
                filename, filepath, file_type, file_size, mtime, file_hash, chunk_count, indexed_at = row
                
                file_info = {
                    "filename": filename,
                    "filepath": filepath,
                    "file_type": file_type,
                    "file_size": file_size or 0,
                    "file_size_mb": round((file_size or 0) / (1024 * 1024), 2),
                    "modification_time": datetime.fromtimestamp(mtime).isoformat() if mtime else None,
                    "hash": file_hash,
                    "chunk_count": chunk_count or 0,
                    "indexed_at": indexed_at,
                    "status": "indexed"
                }
                
                files.append(file_info)
                total_chunks += chunk_count or 0
                total_size += file_size or 0
            
            return {
                "files": files,
                "summary": {
                    "total_files": len(files),
                    "total_chunks": total_chunks,
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "last_sync": files[0]["indexed_at"] if files else None
                },
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get database files: {e}")
            return {"error": str(e), "success": False}
    
    def get_file_statistics(self):
        """Get statistics about files in the knowledge base"""
        try:
            if not self.conn and not self.init_database_connection():
                return {"error": "Database connection failed", "success": False}
            
            # File type distribution
            self.cursor.execute("""
                SELECT file_type, COUNT(*) as count, SUM(chunk_count) as total_chunks,
                       SUM(file_size) as total_size
                FROM file_index 
                GROUP BY file_type 
                ORDER BY count DESC
            """)
            type_stats = self.cursor.fetchall()
            
            # Recent files (last 7 days)
            self.cursor.execute("""
                SELECT COUNT(*) 
                FROM file_index 
                WHERE datetime(indexed_at) > datetime('now', '-7 days')
            """)
            recent_files = self.cursor.fetchone()[0]
            
            # Largest files
            self.cursor.execute("""
                SELECT filename, file_size, chunk_count
                FROM file_index 
                ORDER BY file_size DESC 
                LIMIT 5
            """)
            largest_files = self.cursor.fetchall()
            
            # Format statistics
            file_types = []
            for file_type, count, chunks, size in type_stats:
                file_types.append({
                    "type": file_type,
                    "count": count,
                    "total_chunks": chunks or 0,
                    "total_size_mb": round((size or 0) / (1024 * 1024), 2)
                })
            
            largest = []
            for filename, size, chunks in largest_files:
                largest.append({
                    "filename": filename,
                    "size_mb": round((size or 0) / (1024 * 1024), 2),
                    "chunk_count": chunks or 0
                })
            
            return {
                "file_types": file_types,
                "recent_files_count": recent_files,
                "largest_files": largest,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get file statistics: {e}")
            return {"error": str(e), "success": False}
    
    def search_files(self, query):
        """Search files by name or type"""
        try:
            if not self.conn and not self.init_database_connection():
                return {"error": "Database connection failed", "success": False}
            
            # Search in filename and file_type
            search_query = f"%{query}%"
            self.cursor.execute("""
                SELECT filename, filepath, file_type, file_size, chunk_count, indexed_at
                FROM file_index 
                WHERE filename LIKE ? OR file_type LIKE ?
                ORDER BY indexed_at DESC
            """, (search_query, search_query))
            
            results = self.cursor.fetchall()
            
            files = []
            for row in results:
                filename, filepath, file_type, file_size, chunk_count, indexed_at = row
                files.append({
                    "filename": filename,
                    "filepath": filepath,
                    "file_type": file_type,
                    "file_size_mb": round((file_size or 0) / (1024 * 1024), 2),
                    "chunk_count": chunk_count or 0,
                    "indexed_at": indexed_at
                })
            
            return {
                "files": files,
                "count": len(files),
                "query": query,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ File search failed: {e}")
            return {"error": str(e), "success": False}

class SyncManager:
    """Manager for handling rclone sync process"""
    
    def __init__(self):
        self.sync_process = None
        self.sync_thread = None
        self.should_run = False
        
    def start_sync(self):
        """Start the rclone sync process"""
        try:
            if not SYNC_ENABLED:
                logger.info("📋 Sync is disabled in configuration")
                return False
                
            if not os.path.exists(SYNC_BATCH_FILE):
                logger.error(f"❌ Sync batch file not found: {SYNC_BATCH_FILE}")
                return False
            
            logger.info("🔄 Starting rclone sync process...")
            self.should_run = True
            
            # Start sync in a separate thread
            self.sync_thread = threading.Thread(target=self._run_sync, daemon=True)
            self.sync_thread.start()
            
            logger.info("✅ Rclone sync started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start sync: {e}")
            return False
    
    def _run_sync(self):
        """Run the sync process in a loop"""
        while self.should_run:
            try:
                logger.info("🔄 Running rclone sync...")
                
                # Run the batch file
                self.sync_process = subprocess.Popen(
                    SYNC_BATCH_FILE,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW  # Hide console window
                )
                
                # Wait for 30 seconds or until process completes
                try:
                    stdout, stderr = self.sync_process.communicate(timeout=30)
                    if stdout:
                        logger.debug(f"Sync output: {stdout.decode()}")
                    if stderr:
                        logger.warning(f"Sync warnings: {stderr.decode()}")
                except subprocess.TimeoutExpired:
                    # This is expected as the batch file runs in a loop
                    pass
                
                # Wait before next check
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"❌ Sync process error: {e}")
                time.sleep(30)  # Wait before retrying
    
    def stop_sync(self):
        """Stop the sync process"""
        try:
            logger.info("🛑 Stopping rclone sync process...")
            self.should_run = False
            
            if self.sync_process:
                self.sync_process.terminate()
                try:
                    self.sync_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.sync_process.kill()
                
            logger.info("✅ Rclone sync stopped")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to stop sync: {e}")
            return False

class ChatbotAPI:
    def __init__(self):
        self.client = None
        self.vectorstore = None
        self.embeddings = None
        self.llm = None
        self.initialized = False
        self.shutdown_requested = False
        self.kb_manager = KnowledgeBaseManager()  # Add KB manager
        self.sync_manager = SyncManager()  # Add this line

    def initialize(self):
        """Initialize all components - following debug.py pattern"""
        try:
            validate_environment()
            
            # Initialize Qdrant client
            logger.info("🔗 Connecting to Qdrant Cloud...")
            self.client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                timeout=30
            )

            # Start sync manager after successful initialization
            if SYNC_ENABLED:
                logger.info("🔄 Starting automatic sync...")
                sync_started = self.sync_manager.start_sync()
                if sync_started:
                    logger.info("✅ Automatic sync started")
                else:
                    logger.warning("⚠️ Failed to start automatic sync")
            
            self.initialized = True
            logger.info("🎉 ChatbotAPI initialized successfully!")
            return True
        
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"✅ Connected to Qdrant Cloud: {len(collections.collections)} collections found")
        
            # Initialize embeddings
            logger.info("🔤 Initializing Google embeddings...")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        
            # Initialize vector store
            logger.info("📚 Setting up vector store...")
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=COLLECTION_NAME,
                embedding=self.embeddings
            )
        
            # Initialize LLM
            logger.info("🤖 Initializing Gemini LLM...")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.1,
                convert_system_message_to_human=True
            )
        
            # Test vector store
            try:
                # Try to get collection info to verify everything is working
                collection_info = self.client.get_collection(COLLECTION_NAME)
                logger.info(f"✅ Vector store ready: {collection_info.points_count} points in collection")
            except Exception as e:
                logger.warning(f"⚠️ Collection test failed: {e}")
                # ... existing initialization code ...
        
            # Initialize KB manager database connection - REFRESH CONNECTION
            logger.info("🗄️ Initializing database connection...")
            if not self.kb_manager.init_database_connection():
                logger.warning("⚠️ Database connection failed, API will work with vector DB only")
            else:
                # Test database with a quick query
                test_result = self.kb_manager.get_database_files()
                if test_result.get('success'):
                    file_count = test_result.get('summary', {}).get('total_files', 0)
                    logger.info(f"✅ Database connected: {file_count} files found")
                else:
                    logger.warning("⚠️ Database test query failed")
        
            self.initialized = True
            logger.info("🎉 ChatbotAPI initialized successfully!")
            return True
        
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def shutdown(self):
        """Gracefully shutdown the chatbot"""
        try:
            logger.info("🔄 Shutting down ChatbotAPI...")
            self.initialized = False
            self.shutdown_requested = True
            
            # Close KB manager database connection
            self.kb_manager.close_database_connection()
            
            # Close Qdrant client connection
            if self.client:
                self.client.close()
                self.client = None
                logger.info("✅ Qdrant client closed")
            
            # Reset all components
            self.vectorstore = None
            self.embeddings = None
            self.llm = None
            
            logger.info("✅ ChatbotAPI shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
            return False
    
    def reload(self):
        """Reload the chatbot by shutting down and reinitializing"""
        try:
            logger.info("🔄 Reloading ChatbotAPI...")
            
            # Shutdown first
            self.shutdown()
            
            # Wait a moment for cleanup
            import time
            time.sleep(1)
            
            # Reinitialize
            self.shutdown_requested = False
            success = self.initialize()
            
            if success:
                logger.info("✅ ChatbotAPI reloaded successfully!")
            else:
                logger.error("❌ ChatbotAPI reload failed!")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Reload error: {e}")
            return False
    
    def get_files_in_database(self):
        """Get list of files stored in the vector database - FIXED VERSION"""
        try:
            if not self.initialized:
                raise Exception("ChatbotAPI not initialized")
            
            logger.info("📁 Retrieving files from Qdrant Cloud database...")
            
            # Use a simple scroll without filters first to get all points
            # Then filter by unique filenames in Python code
            points, _ = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=10000,  # Adjust based on your needs
                with_payload=True,
                with_vectors=False  # We don't need vectors, just metadata
            )
            
            # Extract unique filenames from metadata
            filenames = set()
            file_stats = {}
            
            for point in points:
                # Check if point has payload and filename
                if hasattr(point, 'payload') and point.payload and 'filename' in point.payload:
                    filename = point.payload['filename']
                    filenames.add(filename)
                    
                    # Count chunks per file
                    if filename in file_stats:
                        file_stats[filename]['chunks'] += 1
                    else:
                        file_stats[filename] = {
                            'chunks': 1,
                            'metadata': {k: v for k, v in point.payload.items() if k != 'filename'}
                        }
                # Handle case where payload might be structured differently
                elif hasattr(point, 'payload') and point.payload:
                    # Check for alternative metadata structure
                    metadata = point.payload.get('metadata', {})
                    if isinstance(metadata, dict) and 'filename' in metadata:
                        filename = metadata['filename']
                        filenames.add(filename)
                        
                        if filename in file_stats:
                            file_stats[filename]['chunks'] += 1
                        else:
                            file_stats[filename] = {
                                'chunks': 1,
                                'metadata': metadata
                            }
            
            # Create detailed file info
            files_info = []
            for filename in sorted(filenames):
                files_info.append({
                    'filename': filename,
                    'chunks_count': file_stats[filename]['chunks'],
                    'sample_metadata': file_stats[filename]['metadata']
                })
            
            logger.info(f"📄 Found {len(files_info)} unique files in Qdrant Cloud database")
            
            return {
                'files': files_info,
                'total_files': len(files_info),
                'total_chunks': len(points),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get files from Qdrant Cloud database: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def get_file_details(self, filename):
        """Get detailed information about a specific file - FIXED FOR QDRANT CLOUD"""
        try:
            if not self.initialized:
                raise Exception("ChatbotAPI not initialized")
            
            logger.info(f"📄 Getting details for file: {filename}")
            
            # Since filtering by filename is causing issues in Qdrant Cloud,
            # we'll get all points and filter in Python
            points, _ = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            # Filter points by filename in Python
            file_points = []
            for point in points:
                if hasattr(point, 'payload') and point.payload:
                    # Check direct filename
                    if point.payload.get('filename') == filename:
                        file_points.append(point)
                    # Check metadata.filename
                    elif point.payload.get('metadata', {}).get('filename') == filename:
                        file_points.append(point)
            
            if not file_points:
                return {
                    'error': f'File "{filename}" not found in database',
                    'success': False
                }
            
            # Analyze the chunks
            chunks_info = []
            total_content_length = 0
            metadata_sample = {}
            
            for i, point in enumerate(file_points):
                # Get content - check both 'text' and 'page_content' fields
                content = point.payload.get('text', '') or point.payload.get('page_content', '')
                
                chunk_info = {
                    'chunk_id': i + 1,
                    'point_id': str(point.id),
                    'content_preview': content[:200] + '...' if len(content) > 200 else content,
                    'content_length': len(content)
                }
                chunks_info.append(chunk_info)
                total_content_length += chunk_info['content_length']
                
                # Keep sample metadata from first chunk
                if i == 0:
                    metadata_sample = {k: v for k, v in point.payload.items() 
                                     if k not in ['text', 'page_content', 'filename']}
            
            return {
                'filename': filename,
                'chunks_count': len(file_points),
                'total_content_length': total_content_length,
                'average_chunk_size': total_content_length // len(file_points) if file_points else 0,
                'sample_metadata': metadata_sample,
                'chunks': chunks_info,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get file details: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def search_knowledge_base(self, query, k=5, filter_filename=None):
        """Search the knowledge base - FIXED FOR QDRANT CLOUD"""
        try:
            if not self.initialized:
                raise Exception("ChatbotAPI not initialized")
            
            # For now, we'll skip filename filtering due to index issues
            # and do post-filtering if needed
            search_kwargs = {"k": k * 2 if filter_filename else k}  # Get more if filtering
            
            logger.info(f"🔍 Searching for: '{query}' (k={k})")
            if filter_filename:
                logger.info(f"⚠️ Filename filter '{filter_filename}' will be applied after search due to index limitations")
            
            # Perform search without Qdrant filters
            results = self.vectorstore.similarity_search(query, **search_kwargs)
            
            # Apply filename filter in Python if specified
            if filter_filename:
                filtered_results = []
                for doc in results:
                    if hasattr(doc, 'metadata') and doc.metadata:
                        # Check if filename matches
                        doc_filename = doc.metadata.get('filename')
                        if doc_filename == filter_filename:
                            filtered_results.append(doc)
                        # Also check if filename is part of source or other metadata
                        elif filter_filename in str(doc.metadata):
                            filtered_results.append(doc)
                
                results = filtered_results[:k]  # Limit to requested k
                logger.info(f"📄 After filename filtering: {len(results)} chunks")
            
            logger.info(f"📄 Found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def generate_response(self, query, context_docs):
        """Generate response using Gemini LLM"""
        try:
            if not context_docs:
                return "I couldn't find any relevant information in the knowledge base to answer your question."
            
            # Prepare context from search results
            context_text = ""
            sources = set()
            
            for i, doc in enumerate(context_docs[:3], 1):  # Limit to top 3 for context
                context_text += f"Context {i}:\n{doc.page_content}\n\n"
                if hasattr(doc, 'metadata') and 'filename' in doc.metadata:
                    sources.add(doc.metadata['filename'])
            
            # Create prompt
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context from a knowledge base.

Instructions:
1. Answer the question using ONLY the information provided in the context
2. If the context doesn't contain enough information, say so clearly
3. Be concise but comprehensive
4. If you mention specific facts, try to indicate which context they came from
5. Be helpful and professional

Context:
{context}

Question: {question}

Answer:"""

            prompt = system_prompt.format(context=context_text.strip(), question=query)
            
            # Generate response
            logger.info("🤖 Generating LLM response...")
            response = self.llm.invoke(prompt)
            
            # Extract text content
            if hasattr(response, 'content'):
                answer_text = response.content
            else:
                answer_text = str(response)
            
            # Add sources information
            if sources:
                sources_text = ", ".join(sorted(sources))
                answer_text += f"\n\n📚 Sources: {sources_text}"
            
            logger.info("✅ Response generated successfully")
            return answer_text
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return f"Sorry, I encountered an error while generating a response: {str(e)}"
    
    def chat(self, query, filename_filter=None, k=5):
        """Main chat function"""
        try:
            if not self.initialized:
                if not self.initialize():
                    return {
                        "error": "Failed to initialize chatbot",
                        "response": "I'm experiencing technical difficulties. Please try again later."
                    }
            
            logger.info(f"💬 Processing query: '{query}'")
            
            # Search knowledge base
            context_docs = self.search_knowledge_base(query, k=k, filter_filename=filename_filter)
            
            # Generate response
            response = self.generate_response(query, context_docs)
            
            # Prepare metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "chunks_found": len(context_docs),
                "search_filter": filename_filter,
                "query": query
            }
            
            return {
                "response": response,
                "metadata": metadata,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Chat error: {e}")
            return {
                "error": str(e),
                "response": "I apologize, but I encountered an error while processing your question.",
                "success": False
            }

# Initialize Flask app
app = Flask(__name__)
CORS(app)
from flask import render_template  # already partially imported earlier

# Global chatbot instance
chatbot = ChatbotAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "initialized": chatbot.initialized,
        "shutdown_requested": chatbot.shutdown_requested,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/initialize', methods=['POST'])
def initialize_chatbot():
    """Initialize or reinitialize the chatbot"""
    try:
        success = chatbot.initialize()
        return jsonify({
            "success": success,
            "message": "Chatbot initialized successfully" if success else "Initialization failed",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Initialization failed"
        }), 500

@app.route('/reload', methods=['POST'])
def reload_chatbot():
    """Reload the chatbot (shutdown and reinitialize)"""
    try:
        success = chatbot.reload()
        return jsonify({
            "success": success,
            "message": "Chatbot reloaded successfully" if success else "Reload failed",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Reload endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Reload failed"
        }), 500

@app.route('/shutdown', methods=['POST'])
def shutdown_chatbot():
    """Shutdown the chatbot gracefully"""
    try:
        success = chatbot.shutdown()
        return jsonify({
            "success": success,
            "message": "Chatbot shutdown successfully" if success else "Shutdown failed",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Shutdown failed"
        }), 500

@app.route('/quit', methods=['POST'])
def quit_server():
    """Quit the entire server"""
    try:
        logger.info("🔴 Server shutdown requested...")
        
        # Shutdown chatbot first
        chatbot.shutdown()
        
        # Schedule server shutdown in a separate thread
        def shutdown_server():
            import time
            time.sleep(1)  # Give time for response to be sent
            logger.info("🔴 Shutting down Flask server...")
            os.kill(os.getpid(), signal.SIGTERM)
        
        thread = threading.Thread(target=shutdown_server)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Server shutting down...",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Quit error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Failed to shutdown server"
        }), 500

# Enhanced Knowledge Base Endpoints

@app.route('/knowledge-base/files', methods=['GET'])
def get_knowledge_base_files():
    """Get all files in the knowledge base with detailed information from SQLite database"""
    try:
        result = chatbot.kb_manager.get_database_files()
        
        if result.get('success', False):
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"❌ Get knowledge base files error: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/knowledge-base/statistics', methods=['GET'])
def get_knowledge_base_statistics():
    """Get statistics about the knowledge base"""
    try:
        result = chatbot.kb_manager.get_file_statistics()
        
        if result.get('success', False):
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"❌ Get knowledge base statistics error: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/knowledge-base/search', methods=['POST'])
def search_knowledge_base_files():
    """Search files in the knowledge base by name or type"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body",
                "success": False
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                "error": "Query cannot be empty",
                "success": False
            }), 400
        
        result = chatbot.kb_manager.search_files(query)
        
        if result.get('success', False):
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"❌ Search knowledge base files error: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500
    
@app.route('/knowledge-base/refresh', methods=['POST'])
def refresh_database_connection():
    """Refresh database connection to pick up latest changes"""
    try:
        logger.info("🔄 Refreshing database connection...")
        
        # Reinitialize database connection
        success = chatbot.kb_manager.init_database_connection()
        
        if success:
            # Test with a quick query
            test_result = chatbot.kb_manager.get_database_files()
            if test_result.get('success'):
                file_count = test_result.get('summary', {}).get('total_files', 0)
                return jsonify({
                    "success": True,
                    "message": f"Database connection refreshed successfully",
                    "files_count": file_count,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "Database connection established but query test failed",
                    "error": test_result.get('error')
                }), 500
        else:
            return jsonify({
                "success": False,
                "message": "Failed to refresh database connection"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Database refresh error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Database refresh failed"
        }), 500

@app.route('/knowledge-base/summary', methods=['GET'])
def get_knowledge_base_summary():
    """Get a comprehensive summary of the knowledge base"""
    try:
        # Get files from database
        files_result = chatbot.kb_manager.get_database_files()
        
        # Get statistics
        stats_result = chatbot.kb_manager.get_file_statistics()
        
        # Get vector database info
        vector_info = chatbot.get_files_in_database()
        
        if not all([files_result.get('success'), stats_result.get('success'), vector_info.get('success')]):
            return jsonify({
                "error": "Failed to get complete knowledge base information",
                "success": False
            }), 500
        
        # Combine all information
        summary = {
            "database_files": files_result.get('summary', {}),
            "vector_database": {
                "total_files": vector_info.get('total_files', 0),
                "total_chunks": vector_info.get('total_chunks', 0)
            },
            "file_statistics": stats_result,
            "sync_status": {
                "database_files": len(files_result.get('files', [])),
                "vector_files": vector_info.get('total_files', 0),
                "in_sync": len(files_result.get('files', [])) == vector_info.get('total_files', 0)
            },
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
        
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"❌ Get knowledge base summary error: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

# Main files endpoint (consolidated from both versions)
@app.route('/files', methods=['GET'])
def list_files():
    """List all files with comprehensive information including summary"""
    try:
        if not chatbot.kb_manager.ensure_connection():
            logger.warning("⚠️ Could not establish database connection")
        
        # Get files from SQLite database (includes summary)
        db_result = chatbot.kb_manager.get_database_files()
        
        # Get vector database info
        vector_result = chatbot.get_files_in_database()
        
        if not db_result.get('success', False):
            # Fallback to vector database only if SQLite fails
            logger.warning("⚠️ SQLite database unavailable, using vector database only")
            if vector_result.get('success', False):
                return jsonify(vector_result)
            else:
                return jsonify(vector_result), 500
        
        # Combine information from both sources
        files = db_result.get('files', [])
        db_summary = db_result.get('summary', {})
        
        # Create detailed summary text
        summary_lines = [
            "📋 CURRENT DATABASE FILES:",
            "=" * 60
        ]
        
        # Add file details
        for file_info in files:
            filename = file_info.get('filename', 'Unknown')
            file_type = file_info.get('file_type', 'unknown')
            chunk_count = file_info.get('chunk_count', 0)
            file_size_mb = file_info.get('file_size_mb', 0)
            
            summary_lines.append(f"✅ {filename} ({file_type}) - {chunk_count} chunks ({file_size_mb:.2f}MB)")
        
        # Add totals
        summary_lines.extend([
            f"📊 Total: {db_summary.get('total_files', len(files))} files",
            "=" * 60,
            "",
            "🎉 DATABASE STATUS:",
            f"✅ Total Files: {db_summary.get('total_files', 0)}",
            f"📄 Total Chunks: {db_summary.get('total_chunks', 0)}",
            f"💾 Total Size: {db_summary.get('total_size_mb', 0):.2f}MB"
        ])
        
        # Add sync status if we have vector info
        if vector_result.get('success', False):
            vector_files = vector_result.get('total_files', 0)
            db_files = db_summary.get('total_files', 0)
            in_sync = vector_files == db_files
            
            summary_lines.extend([
                f"🔄 Vector DB Files: {vector_files}",
                f"{'✅ IN SYNC' if in_sync else '⚠️ OUT OF SYNC'}"
            ])
        
        summary_text = "\n".join(summary_lines)
        
        # Return enhanced response with summary
        return jsonify({
            "success": True,
            "files": files,
            "total_files": db_summary.get('total_files', len(files)),
            "total_chunks": db_summary.get('total_chunks', 0),
            "total_size_mb": db_summary.get('total_size_mb', 0),
            "summary": summary_text,
            "database_info": db_summary,
            "vector_info": vector_result if vector_result.get('success') else None,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ List files error: {e}")
        return jsonify({
            "error": str(e),
            "success": False,
            "summary": f"❌ Error loading files: {str(e)}"
        }), 500

@app.route('/files/<filename>', methods=['GET'])
def get_file_details_endpoint(filename):
    """Get detailed information about a specific file"""
    try:
        result = chatbot.get_file_details(filename)
        
        status_code = 200 if result.get('success', False) else 404
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"❌ Get file details error: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body",
                "success": False
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                "error": "Query cannot be empty",
                "success": False
            }), 400
        
        # Optional parameters
        filename_filter = data.get('filename_filter')
        k = data.get('k', 5)
        
        # Process chat
        result = chatbot.chat(query, filename_filter=filename_filter, k=k)
        
        status_code = 200 if result.get('success', False) else 500
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        return jsonify({
            "error": str(e),
            "response": "Internal server error",
            "success": False
        }), 500

@app.route('/search', methods=['POST'])
def search_endpoint():
    """Search endpoint (returns raw search results)"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body"
            }), 400
        
        query = data['query'].strip()
        filename_filter = data.get('filename_filter')
        k = data.get('k', 5)
        
        if not chatbot.initialized:
            if not chatbot.initialize():
                return jsonify({
                    "error": "Failed to initialize chatbot"
                }), 500
        
        # Perform search
        results = chatbot.search_knowledge_base(query, k=k, filter_filename=filename_filter)
        
        # Format results
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            })
        
        return jsonify({
            "results": formatted_results,
            "count": len(formatted_results),
            "query": query,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"❌ Search endpoint error: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/test-database', methods=['GET'])
def test_database_integration():
    """Test endpoint to verify database integration"""
    try:
        # Test database connection
        db_success = chatbot.kb_manager.ensure_connection()
        
        # Test database query
        db_result = chatbot.kb_manager.get_database_files()
        
        # Test vector database
        vector_result = chatbot.get_files_in_database()
        
        return jsonify({
            "database_connection": db_success,
            "database_files": db_result.get('success', False),
            "database_file_count": db_result.get('summary', {}).get('total_files', 0),
            "vector_db_success": vector_result.get('success', False),
            "vector_file_count": vector_result.get('total_files', 0),
            "database_path": SQLITE_PATH,
            "database_exists": os.path.exists(SQLITE_PATH),
            "database_size": os.path.getsize(SQLITE_PATH) if os.path.exists(SQLITE_PATH) else 0,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/collections', methods=['GET'])
def list_collections():
    """List available Qdrant collections"""
    try:
        if not chatbot.client:
            chatbot.initialize()
        
        if not chatbot.client:
            return jsonify({
                "error": "Failed to connect to Qdrant"
            }), 500
        
        collections = chatbot.client.get_collections()
        collection_info = []
        
        for col in collections.collections:
            try:
                info = chatbot.client.get_collection(col.name)
                collection_info.append({
                    "name": col.name,
                    "points_count": info.points_count,
                    "vector_size": info.config.params.vectors.size
                })
            except Exception as e:
                collection_info.append({
                    "name": col.name,
                    "error": str(e)
                })
        
        return jsonify({
            "collections": collection_info,
            "success": True
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "Please check the API documentation for available endpoints"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "Something went wrong on the server"
    }), 500


@app.route('/app.js')
def serve_app_js():
    """Serve the app.js file from static folder"""
    return send_from_directory('static', 'app.js')

@app.route('/static/<path:filename>')
def serve_static_files(filename):
    """Serve static files (CSS, JS, images, etc.)"""
    return send_from_directory('static', 'static', filename)


@app.route('/', methods=['GET'])
def home():
    """Home route - serves HTML interface or JSON API info based on Accept header"""
    # Check if request accepts HTML (browser request)
    if request.headers.get('Accept', '').find('text/html') != -1:
        return render_template("intellirag.html")
    
    # Otherwise return JSON API information
    return jsonify({
        "message": "🔥 Welcome to your Enhanced Chatbot API!",
        "endpoints": {
            "core": ["/health", "/chat", "/search"],
            "management": ["/initialize", "/reload", "/shutdown", "/quit"],
            "files": ["/files", "/files/<filename>"],
            "knowledge_base": ["/knowledge-base/files", "/knowledge-base/statistics", "/knowledge-base/search", "/knowledge-base/summary"],
            "system": ["/collections"]
        },
        "status": {
            "initialized": chatbot.initialized,
            "timestamp": datetime.now().isoformat()
        }
    })

@app.route('/sync/status', methods=['GET'])
def get_sync_status():
    """Get sync process status"""
    try:
        is_running = chatbot.sync_manager.should_run if hasattr(chatbot, 'sync_manager') else False
        
        return jsonify({
            "sync_enabled": SYNC_ENABLED,
            "sync_running": is_running,
            "sync_batch_file": SYNC_BATCH_FILE,
            "batch_file_exists": os.path.exists(SYNC_BATCH_FILE),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/sync/start', methods=['POST'])
def start_sync():
    """Manually start sync process"""
    try:
        if not hasattr(chatbot, 'sync_manager'):
            return jsonify({
                "error": "Sync manager not available",
                "success": False
            }), 500
        
        success = chatbot.sync_manager.start_sync()
        
        return jsonify({
            "success": success,
            "message": "Sync started successfully" if success else "Failed to start sync",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/sync', methods=['POST'])
def sync_knowledge_base():
    """Sync knowledge base with filesystem using sync.py"""
    try:
        import subprocess
        import sys
        import os
        from pathlib import Path
        
        # Get the directory where your sync.py is located
        # Adjust this path to match your sync.py location
        sync_script_path = Path(__file__).parent / "sync.py"
        
        if not sync_script_path.exists():
            return jsonify({
                'success': False,
                'error': f'Sync script not found at {sync_script_path}'
            })
        
        # Get force_rebuild parameter
        data = request.get_json() or {}
        force_rebuild = data.get('force_rebuild', False)
        
        # Build command
        cmd = [sys.executable, str(sync_script_path)]
        if force_rebuild:
            cmd.append('--rebuild')
        
        # Run sync script
        logger.info(f"Running sync command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Check result
        if result.returncode == 0:
            # Parse output for summary info
            output_lines = result.stdout.strip().split('\n')
            summary_info = {}
            
            for line in output_lines:
                if 'Added/Updated:' in line:
                    summary_info['added'] = line.split(':')[1].strip()
                elif 'Removed:' in line:
                    summary_info['removed'] = line.split(':')[1].strip()
                elif 'Failed:' in line:
                    summary_info['failed'] = line.split(':')[1].strip()
            
            # Create message
            message = f"Added: {summary_info.get('added', '0')}, Removed: {summary_info.get('removed', '0')}"
            if summary_info.get('failed', '0') != '0':
                message += f", Failed: {summary_info.get('failed', '0')}"
            
            return jsonify({
                'success': True,
                'message': message,
                'output': result.stdout,
                'summary': summary_info
            })
        else:
            logger.error(f"Sync failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            
            return jsonify({
                'success': False,
                'error': f'Sync failed: {result.stderr or result.stdout or "Unknown error"}',
                'return_code': result.returncode
            })
            
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': 'Sync operation timed out (5 minutes)'
        })
    except Exception as e:
        logger.error(f"Sync endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': f'Sync error: {str(e)}'
        })
        logger.warning("⚠️ Failed to start automatic sync")    
        
@app.route('/sync/stop', methods=['POST'])
def stop_sync():
    """Manually stop sync process"""
    try:
        if not hasattr(chatbot, 'sync_manager'):
            return jsonify({
                "error": "Sync manager not available",
                "success": False
            }), 500
        
        success = chatbot.sync_manager.stop_sync()
        
        return jsonify({
            "success": success,
            "message": "Sync stopped successfully" if success else "Failed to stop sync",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    logger.info(f"🔴 Received signal {signum}, shutting down gracefully...")
    chatbot.shutdown()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)




if __name__ == '__main__':
    # Initialize on startup
    logger.info("🚀 Starting Enhanced Chatbot API...")
    
    def cleanup():
        logger.info("🧹 Performing cleanup...")
        if 'chatbot' in globals():
            chatbot.shutdown()
    
    atexit.register(cleanup)
    # Test initialization
    if chatbot.initialize():
        logger.info("✅ Chatbot initialized successfully on startup")
    else:
        logger.warning("⚠️ Chatbot initialization failed on startup, will retry on first request")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)