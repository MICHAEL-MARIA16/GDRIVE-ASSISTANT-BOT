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
        """Initialize SQLite database connection"""
        try:
            if not os.path.exists(SQLITE_PATH):
                logger.warning(f"⚠️ Database file not found: {SQLITE_PATH}")
                return False
                
            self.conn = sqlite3.connect(SQLITE_PATH, timeout=30)
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def close_database_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def get_database_files(self):
        """Get all files from the database with detailed information"""
        try:
            if not self.conn and not self.init_database_connection():
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

class ChatbotAPI:
    def __init__(self):
        self.client = None
        self.vectorstore = None
        self.embeddings = None
        self.llm = None
        self.initialized = False
        self.shutdown_requested = False
        self.kb_manager = KnowledgeBaseManager()  # Add KB manager
    
    def initialize(self):
        """Initialize all components - following debug.py pattern"""
        try:
            validate_environment()
            
            # Initialize Gemini Embeddings (same as debug.py)
            logger.info("🧬 Initializing Gemini Embeddings...")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            # Test embedding to ensure it works
            test_vec = self.embeddings.embed_query("test vector")
            logger.info(f"✅ Gemini Embeddings ready! Vector size: {len(test_vec)}")
            
            # Initialize Qdrant Cloud Client (same as debug.py)
            logger.info("🔌 Connecting to Qdrant Cloud...")
            qdrant_url = os.getenv('QDRANT_URL')
            qdrant_api_key = os.getenv('QDRANT_API_KEY')
            
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            
            # Verify connection and collection
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            logger.info(f"✅ Connected to: {qdrant_url}")
            logger.info(f"📁 Collections found: {collection_names}")
            
            if COLLECTION_NAME not in collection_names:
                logger.warning(f"⚠️ Collection '{COLLECTION_NAME}' not found!")
                return False
            
            # Get collection stats
            info = self.client.get_collection(COLLECTION_NAME)
            logger.info(f"📊 {COLLECTION_NAME} Stats → Points: {info.points_count}, Size: {info.config.params.vectors.size}")
            
            # Initialize VectorStore (same pattern as debug.py)
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=COLLECTION_NAME,
                embedding=self.embeddings,
            )
            
            # Initialize LLM
            logger.info("🤖 Initializing Gemini LLM...")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.7,
                convert_system_message_to_human=True
            )
            
            # Test search to verify everything works
            logger.info("🔍 Testing vectorstore search...")
            test_results = self.vectorstore.similarity_search("test query", k=1)
            logger.info(f"✅ Search test successful, found {len(test_results)} results")
            
            # Initialize KB manager database connection
            self.kb_manager.init_database_connection()
            
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
    
    # Test initialization
    if chatbot.initialize():
        logger.info("✅ Chatbot initialized successfully on startup")
    else:
        logger.warning("⚠️ Chatbot initialization failed on startup, will retry on first request")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)