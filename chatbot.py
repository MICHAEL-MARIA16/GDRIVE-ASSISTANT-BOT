# === Enhanced RAG Chatbot with Auto-Sync Background Thread and DOCX Support ===
import os
import sys
import sqlite3
import threading
import time
import platform
import subprocess
import psutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
import logging
import pandas as pd
import hashlib
import warnings
import atexit
import json
from contextlib import contextmanager
from filelock import FileLock

# Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*langchain.*")

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# For DOCX support
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
    logger.info("✅ DOCX support available")
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("⚠️ DOCX support not available - install python-docx")

# For Qdrant support - ONLY Qdrant Cloud
try:
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    QDRANT_AVAILABLE = True
    logger.info("✅ Using modern QdrantVectorStore for cloud")
except ImportError:
    try:
        # Try the community package version
        from langchain_community.vectorstores import QdrantVectorStore
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        QDRANT_AVAILABLE = True
        logger.info("✅ Using QdrantVectorStore from community package")
    except ImportError:
        QDRANT_AVAILABLE = False
        logger.error("❌ Qdrant not available! Install with: pip install langchain-qdrant qdrant-client")

# === Configuration ===
KB_PATH = "C:/Users/maria selciya/Desktop/chatbotKB_test"
SQLITE_PATH = "./file_index.db"
COLLECTION_NAME = "kb_collection"
SYNC_INTERVAL = 120  # 2 minutes in seconds as requested

# Supported file types
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.csv', '.docx', '.md', '.py', '.js', '.html', '.xml', '.json'}

# Global variables for auto-sync and connection management
sync_thread = None
stop_sync = threading.Event()
chatbot_needs_reload = threading.Event()
qdrant_client = None  # Global client to avoid conflicts
sync_lock = threading.Lock()  # Prevent concurrent syncs
qdrant_lock = threading.RLock()  # Reentrant lock for Qdrant operations
connection_pool_lock = threading.Lock()  # For connection pool management

# Connection pool settings
MAX_CONNECTIONS = 3
connection_pool = []
active_connections = 0

# File locks for process coordination
QDRANT_LOCK_FILE = "./qdrant_operations.lock"
SYNC_LOCK_FILE = "./sync_operations.lock"

def validate_environment():
    """Validate required environment variables"""
    missing_vars = []
    
    if not QDRANT_URL:
        missing_vars.append("QDRANT_URL")
    
    if not QDRANT_API_KEY:
        missing_vars.append("QDRANT_API_KEY")
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        missing_vars.append("GOOGLE_API_KEY")
    
    if missing_vars:
        logger.error("❌ Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"   - {var}")
        logger.error("   Please add these to your .env file")
        return False
    
    logger.info("✅ All required environment variables found")
    return True

@contextmanager
def qdrant_operation_lock():
    """Context manager for Qdrant operations with file locking"""
    lock = FileLock(QDRANT_LOCK_FILE)
    try:
        with lock.acquire(timeout=30):
            with qdrant_lock:
                yield
    except Exception as e:
        logger.error(f"❌ Failed to acquire Qdrant operation lock: {e}")
        raise

@contextmanager
def sync_operation_lock():
    """Context manager for sync operations with file locking"""
    lock = FileLock(SYNC_LOCK_FILE)
    try:
        with lock.acquire(timeout=60):
            with sync_lock:
                yield
    except Exception as e:
        logger.error(f"❌ Failed to acquire sync operation lock: {e}")
        raise

class QdrantConnectionManager:
    """Manage Qdrant connections with connection pooling and proper cleanup"""
    
    def __init__(self, max_connections=3):
        self.max_connections = max_connections
        self.connections = []
        self.active_count = 0
        self.lock = threading.Lock()
        
    def get_connection(self):
        """Get a connection from the pool or create a new one"""
        with self.lock:
            # Try to reuse existing connection
            if self.connections:
                client = self.connections.pop()
                try:
                    # Test connection
                    client.get_collections()
                    self.active_count += 1
                    return client
                except Exception as e:
                    logger.warning(f"⚠️ Reused connection failed, creating new one: {e}")
            
            # Create new connection if under limit
            if self.active_count < self.max_connections:
                try:
                    client = QdrantClient(
                        url=QDRANT_URL,
                        api_key=QDRANT_API_KEY,
                        timeout=30
                    )
                    # Test the connection
                    client.get_collections()
                    self.active_count += 1
                    logger.info(f"✅ Created new Qdrant connection ({self.active_count}/{self.max_connections})")
                    return client
                except Exception as e:
                    logger.error(f"❌ Failed to create Qdrant connection: {e}")
                    raise
            else:
                raise Exception("Maximum number of Qdrant connections reached")
    
    def return_connection(self, client):
        """Return a connection to the pool"""
        with self.lock:
            if client and self.active_count > 0:
                try:
                    # Test connection before returning to pool
                    client.get_collections()
                    self.connections.append(client)
                    self.active_count -= 1
                except Exception as e:
                    logger.warning(f"⚠️ Connection test failed, not returning to pool: {e}")
                    self.active_count = max(0, self.active_count - 1)
                    try:
                        client.close()
                    except:
                        pass
    
    def close_all(self):
        """Close all connections"""
        with self.lock:
            for client in self.connections:
                try:
                    client.close()
                except:
                    pass
            self.connections.clear()
            self.active_count = 0
            logger.info("✅ All Qdrant connections closed")

# Global connection manager
connection_manager = QdrantConnectionManager(MAX_CONNECTIONS)

# === Database Schema Management ===
def init_database():
    """Initialize database with proper schema"""
    try:
        conn = sqlite3.connect(SQLITE_PATH, timeout=30)
        cursor = conn.cursor()
        
        # Create file_index table with all required columns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                filepath TEXT NOT NULL,
                file_type TEXT NOT NULL,
                mtime REAL NOT NULL,
                hash TEXT NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Check if indexed_at column exists, if not add it
        cursor.execute("PRAGMA table_info(file_index)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Add missing columns if they don't exist
        if 'indexed_at' not in columns:
            cursor.execute("ALTER TABLE file_index ADD COLUMN indexed_at TIMESTAMP")
            cursor.execute("UPDATE file_index SET indexed_at = CURRENT_TIMESTAMP WHERE indexed_at IS NULL")
            logger.info("✅ Added missing indexed_at column to database")
        
        if 'created_at' not in columns:
            cursor.execute("ALTER TABLE file_index ADD COLUMN created_at TIMESTAMP")
            cursor.execute("UPDATE file_index SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
            logger.info("✅ Added missing created_at column to database")
        
        if 'hash' not in columns:
            cursor.execute("ALTER TABLE file_index ADD COLUMN hash TEXT")
            logger.info("✅ Added missing hash column to database")
        
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise

def get_file_hash(file_path):
    """Calculate hash of a file"""
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"❌ Error calculating hash for {file_path}: {e}")
        return ""

# === Rclone Sync Starter ===
def is_rclone_running():
    """Check if rclone is currently running"""
    try:
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                if proc.info['name'] and "rclone" in proc.info['name'].lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
    except Exception as e:
        logger.error(f"❌ Error checking rclone status: {e}")
        return False

def start_rclone_sync():
    """Start rclone sync process"""
    if platform.system() == "Windows":
        script_path = os.path.abspath("sync_loop.bat")
        if not os.path.exists(script_path):
            logger.warning(f"⚠️ sync_loop.bat not found at {script_path}")
            return
        try:
            if not is_rclone_running():
                subprocess.Popen([script_path], shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
                logger.info("🚀 Rclone sync started in background")
            else:
                logger.info("ℹ️ Rclone sync already running")
        except Exception as e:
            logger.warning(f"⚠️ Failed to start rclone sync: {e}")
    else:
        logger.info("🛑 Rclone sync script not supported on this OS.")

# === Dependency Check ===     
def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    if not QDRANT_AVAILABLE:
        missing_deps.append("langchain-qdrant or qdrant-client")
        logger.error("❌ Qdrant is required for this deployment!")
        return False
    
    if not DOCX_AVAILABLE:
        missing_deps.append("python-docx")
        logger.warning("⚠️ DOCX support not available")
    
    # Check for filelock
    try:
        import filelock
        logger.info("✅ Filelock available for process coordination")
    except ImportError:
        missing_deps.append("filelock")
        logger.warning("⚠️ filelock not available - install with: pip install filelock")
    
    if missing_deps:
        logger.warning("⚠️ Missing dependencies:")
        for dep in missing_deps:
            logger.warning(f"   - {dep}")
        logger.warning("   Install with: pip install python-docx langchain-qdrant qdrant-client filelock")
    
    return QDRANT_AVAILABLE

def check_for_changes():
    """Check if knowledge base has changed since last sync"""
    try:
        # Initialize database if it doesn't exist
        if not Path(SQLITE_PATH).exists():
            init_database()
            return True  # Need initial sync
        
        kb_directory = Path(KB_PATH)
        if not kb_directory.exists():
            logger.warning(f"⚠️ Knowledge base directory not found: {KB_PATH}")
            return False
        
        # Get existing files from database
        conn = sqlite3.connect(SQLITE_PATH, timeout=30)
        cursor = conn.cursor()
        
        # Get all tracked files with their info
        cursor.execute("SELECT filename, mtime, hash FROM file_index")
        existing_files = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
        
        conn.close()
        
        # Check current files
        current_files = set()
        changes_detected = False
        
        for file_path in kb_directory.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Skip system files and hidden files
            if file_path.name.startswith('.') or file_path.name.startswith('~'):
                continue
                
            # Only consider supported file types
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            
            current_files.add(file_path.name)
            
            # Check if file is new or modified
            if file_path.name not in existing_files:
                logger.info(f"📝 New file detected: {file_path.name}")
                changes_detected = True
                break
            else:
                # Check if file was modified by comparing hash
                try:
                    current_hash = get_file_hash(file_path)
                    stored_mtime, stored_hash = existing_files[file_path.name]
                    
                    if current_hash != stored_hash:
                        logger.info(f"📝 Modified file detected: {file_path.name}")
                        changes_detected = True
                        break
                except Exception as e:
                    logger.error(f"❌ Error checking file {file_path.name}: {e}")
                    changes_detected = True
                    break
        
        # Check for deleted files
        if not changes_detected:
            deleted_files = set(existing_files.keys()) - current_files
            if deleted_files:
                logger.info(f"📝 Deleted files detected: {list(deleted_files)}")
                changes_detected = True
        
        return changes_detected
        
    except Exception as e:
        logger.error(f"❌ Error checking for changes: {e}")
        return False

def run_sync_process():
    """Run sync process with proper Qdrant connection management"""
    try:
        with sync_operation_lock():
            logger.info("🔄 Starting sync process with file lock...")
            
            # Get the current script directory
            current_dir = Path(__file__).parent
            sync_script = current_dir / "sync.py"
            
            if not sync_script.exists():
                logger.error(f"❌ sync.py not found at {sync_script}")
                return False
            
            logger.info("🔄 Starting background sync process...")
            
            # Run sync.py with proper environment and error handling
            env = os.environ.copy()
            # Pass connection info to sync script
            env['QDRANT_CLOUD_MODE'] = 'true'
            python_executable = sys.executable
            
            result = subprocess.run(
                [python_executable, str(sync_script)], 
                cwd=current_dir,
                capture_output=True, 
                text=True, 
                encoding='utf-8',
                errors='replace',
                timeout=900,  # 15 minutes timeout
                env=env
            )
            
            # Wait a moment after sync completes
            time.sleep(2)
            
            # Log the output for debugging
            if result.stdout:
                logger.info("📋 Sync output:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        logger.info(f"   {line}")
            
            if result.stderr:
                # Filter out harmless warnings
                error_lines = result.stderr.strip().split('\n')
                filtered_errors = []
                for line in error_lines:
                    if ("already accessed by another instance" not in line and 
                        "Storage folder" not in line and
                        "UserWarning" not in line):
                        filtered_errors.append(line)
                
                if filtered_errors:
                    logger.warning("⚠️ Sync warnings/errors:")
                    for line in filtered_errors:
                        if line.strip():
                            logger.warning(f"   {line}")
            
            success = result.returncode == 0
            if success:
                logger.info("✅ Background sync completed successfully!")
            else:
                logger.error(f"❌ Background sync failed with return code: {result.returncode}")
            
            return success
                
    except subprocess.TimeoutExpired:
        logger.error("❌ Background sync timed out!")
        return False
    except FileNotFoundError:
        logger.error(f"❌ sync.py not found or Python executable not found")
        return False
    except Exception as e:
        logger.error(f"❌ Background sync error: {e}")
        return False

def background_sync():
    """Background thread function that runs sync periodically"""
    logger.info(f"🔄 Background sync thread started (runs every {SYNC_INTERVAL//60} minutes)")
    
    while not stop_sync.is_set():
        try:
            # Wait for the interval or until stop is signaled
            if stop_sync.wait(timeout=SYNC_INTERVAL):
                break  # Stop was signaled
            
            logger.info("🔍 Checking for knowledge base changes...")
            
            # Check for changes first
            if check_for_changes():
                logger.info("📝 Changes detected! Running background sync...")
                print(f"\n🔄 [AUTO-SYNC] Changes detected - syncing knowledge base...")
                
                # Run the sync process with proper locking
                if run_sync_process():
                    logger.info("✅ Background sync completed successfully!")
                    print("✅ [AUTO-SYNC] Knowledge base updated successfully!")
                    chatbot_needs_reload.set()  # Signal that chatbot needs to reload
                else:
                    logger.error("❌ Background sync failed!")
                    print("❌ [AUTO-SYNC] Sync failed - continuing with current knowledge base")
            else:
                logger.debug("ℹ️ No changes detected in knowledge base")
                
        except Exception as e:
            logger.error(f"❌ Background sync thread error: {e}")
            time.sleep(30)  # Wait before retrying

def start_background_sync():
    """Start the background sync thread"""
    global sync_thread
    if sync_thread is None or not sync_thread.is_alive():
        stop_sync.clear()
        chatbot_needs_reload.clear()
        sync_thread = threading.Thread(target=background_sync, daemon=True)
        sync_thread.start()
        logger.info("✅ Background sync thread started")

def stop_background_sync():
    """Stop the background sync thread"""
    global sync_thread
    if sync_thread and sync_thread.is_alive():
        stop_sync.set()
        sync_thread.join(timeout=10)
        logger.info("🛑 Background sync thread stopped")

def cleanup_resources():
    """Clean up resources on exit"""
    try:
        logger.info("🧹 Cleaning up resources...")
        stop_background_sync()
        
        # Close all Qdrant connections
        connection_manager.close_all()
        
        # Clean up lock files
        for lock_file in [QDRANT_LOCK_FILE, SYNC_LOCK_FILE]:
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            except Exception as e:
                logger.warning(f"⚠️ Error removing lock file {lock_file}: {e}")
        
        # Small delay to ensure cleanup is complete
        time.sleep(1)
        logger.info("✅ Cleanup completed")
        
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")

def check_sync_status():
    """Check if sync is needed before starting chatbot"""
    try:
        if not Path(SQLITE_PATH).exists():
            logger.warning("⚠️ No sync database found. Please run sync.py first!")
            response = input("Would you like to run sync now? (y/n): ").lower()
            if response == 'y':
                logger.info("🔄 Running sync first...")
                if run_sync_process():
                    logger.info("✅ Sync completed successfully!")
                    return True
                else:
                    logger.error("❌ Initial sync failed!")
                    return False
            else:
                logger.error("❌ Sync is required before using the chatbot!")
                return False
        return True
    except Exception as e:
        logger.error(f"❌ Error checking sync status: {e}")
        return False

def create_qdrant_collection_if_not_exists(client, collection_name, vector_size=768):
    """Create Qdrant collection if it doesn't exist"""
    try:
        with qdrant_operation_lock():
            # Check if collection exists
            collections = client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if collection_name not in collection_names:
                logger.info(f"📁 Creating new Qdrant collection: {collection_name}")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info("✅ Qdrant collection created successfully!")
            else:
                logger.info(f"✅ Qdrant collection '{collection_name}' already exists")
            
            return True
    except Exception as e:
        logger.error(f"❌ Failed to create/check Qdrant collection: {e}")
        return False

def load_existing_vector_store():
    """Load existing Qdrant vector store with connection management"""
    try:
        # Check dependencies first
        if not check_dependencies():
            return None, None, None
        
        # Validate environment
        if not validate_environment():
            return None, None, None
        
        # Load environment variables
        google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # Initialize LLM with better settings
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-flash-latest", 
            google_api_key=google_api_key,
            temperature=0.3,  # Slightly higher for more natural responses
            max_tokens=1024,
            top_p=0.95
        )
        
        # Get Qdrant client from connection manager
        client = connection_manager.get_connection()
        if not client:
            return None, None, None
        
        try:
            # Create collection if it doesn't exist
            if not create_qdrant_collection_if_not_exists(client, COLLECTION_NAME):
                logger.error("❌ Failed to create/verify Qdrant collection")
                return None, None, None
            
            # Check if collection has documents
            collection_info = client.get_collection(COLLECTION_NAME)
            if collection_info.points_count == 0:
                logger.error("❌ Qdrant collection is empty! Please run sync.py first.")
                connection_manager.return_connection(client)
                return None, None, None
            
            logger.info(f"✅ Found {collection_info.points_count} documents in Qdrant")
            
            # Create vector store
            vectordb = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embeddings
            )
            
            logger.info("✅ Loaded Qdrant vector store successfully")
            # Don't return connection to pool as vectordb will use it
            return llm, vectordb, embeddings
            
        except Exception as e:
            # Return connection to pool on error
            connection_manager.return_connection(client)
            logger.error(f"❌ Failed to create Qdrant vector store: {e}")
            return None, None, None
        
    except Exception as e:
        logger.error(f"❌ Failed to load vector store: {e}")
        return None, None, None

def get_knowledge_base_stats():
    """Get statistics about the knowledge base"""
    try:
        # Ensure database exists and has proper schema
        init_database()
        
        conn = sqlite3.connect(SQLITE_PATH, timeout=30)
        cursor = conn.cursor()
        
        # Get file count by type
        cursor.execute("""
            SELECT file_type, COUNT(*) 
            FROM file_index 
            GROUP BY file_type 
            ORDER BY COUNT(*) DESC
        """)
        file_stats = cursor.fetchall()
        
        # Get total files
        cursor.execute("SELECT COUNT(*) FROM file_index")
        total_files = cursor.fetchone()[0]
        
        # Get last sync time
        cursor.execute("SELECT MAX(indexed_at) FROM file_index")
        last_sync_result = cursor.fetchone()
        last_sync = last_sync_result[0] if last_sync_result and last_sync_result[0] else None
        
        conn.close()
        
        # Get Qdrant stats safely
        qdrant_docs = 0
        try:
            client = connection_manager.get_connection()
            if client:
                with qdrant_operation_lock():
                    collection_info = client.get_collection(COLLECTION_NAME)
                    qdrant_docs = collection_info.points_count
                connection_manager.return_connection(client)
        except Exception as e:
            logger.warning(f"⚠️ Could not get Qdrant stats: {e}")
        
        return {
            'total_files': total_files,
            'file_types': dict(file_stats),
            'last_sync': last_sync,
            'qdrant_documents': qdrant_docs
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting KB stats: {e}")
        return None

def reload_chatbot_if_needed(qa_chain, vectordb):
    """Check if chatbot needs to reload due to background sync"""
    global chatbot_needs_reload
    
    if chatbot_needs_reload.is_set():
        logger.info("🔄 Reloading chatbot due to background sync...")
        print("\n🔄 [RELOAD] Updating chatbot with latest knowledge base...")
        try:
            # Give Qdrant a moment to be fully ready
            time.sleep(1)
            
            # Load updated vector store with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    llm, new_vectordb, embeddings = load_existing_vector_store()
                    if new_vectordb:
                        # Create new QA chain with updated invoke method
                        new_qa_chain = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            retriever=new_vectordb.as_retriever(
                                search_type="similarity",
                                search_kwargs={"k": 10, "score_threshold": 0.1}
                            ),
                            return_source_documents=True,
                            verbose=False
                        )
                        chatbot_needs_reload.clear()
                        logger.info("✅ Chatbot reloaded successfully!")
                        print("✅ [RELOAD] Chatbot updated with latest knowledge!")
                        return new_qa_chain, new_vectordb
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Reload attempt {attempt + 1} failed, retrying in 2s: {e}")
                        time.sleep(2)
                    else:
                        raise e
                        
        except Exception as e:
            logger.error(f"❌ Failed to reload chatbot: {e}")
            print(f"❌ [RELOAD] Failed to update chatbot: {e}")
            chatbot_needs_reload.clear()  # Clear flag even if failed to prevent infinite retries
    
    return qa_chain, vectordb

def format_source_info(source_docs):
    """Format source document information"""
    if not source_docs:
        return "No sources found."
    
    sources = []
    seen_sources = set()
    
    for doc in source_docs:
        source = doc.metadata.get('source', 'Unknown')
        filename = Path(source).name if source != 'Unknown' else 'Unknown'
        
        if filename not in seen_sources:
            seen_sources.add(filename)
            page = doc.metadata.get('page', None)
            if page is not None:
                sources.append(f"📄 {filename} (page {page + 1})")
            else:
                sources.append(f"📄 {filename}")
    
    return "\n".join(sources[:5])  # Show max 5 sources

def test_vector_search(vectordb, query="AI technology"):
    """Test if vector search is working properly"""
    try:
        logger.info(f"🔍 Testing vector search with query: '{query}'")
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(query)
        
        logger.info(f"📊 Found {len(docs)} relevant documents")
        for i, doc in enumerate(docs):
            logger.info(f"   {i+1}. Source: {doc.metadata.get('source', 'Unknown')}")
            logger.info(f"      Content preview: {doc.page_content[:100]}...")
        
        return len(docs) > 0
    except Exception as e:
        logger.error(f"❌ Vector search test failed: {e}")
        return False

def main():
    """Main chatbot function"""
    print("🤖 Enhanced RAG Chatbot with Qdrant (Auto-Sync Every 2 Minutes)")
    print("=" * 70)
    
    # Register cleanup function
    atexit.register(cleanup_resources)
    
    # Check dependencies first
    if not check_dependencies():
        logger.error("❌ Missing required dependencies!")
        logger.error("   Install with: pip install langchain-qdrant qdrant-client python-docx")
        return
    
    # Initialize database first
    try:
        init_database()
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        return
    
    # Start rclone sync (optional)
    start_rclone_sync()
    
    # Check sync status first
    if not check_sync_status():
        return
    
    # Load vector store (Qdrant only)
    print("🔄 Loading Qdrant knowledge base...")
    llm, vectordb, embeddings = load_existing_vector_store()
    
    if not llm or not vectordb:
        logger.error("❌ Failed to initialize chatbot! Please ensure:")
        logger.error("   1. Qdrant database exists and has documents")
        logger.error("   2. Run sync.py to populate the database")
        logger.error("   3. GOOGLE_API_KEY is set in .env file")
        return
    
    # Test vector search
    if not test_vector_search(vectordb):
        logger.warning("⚠️ Vector search test failed - search might not work properly")
    
    # Display knowledge base stats
    stats = get_knowledge_base_stats()
    if stats:
        print(f"\n📊 Knowledge Base Statistics:")
        print(f"   📁 Total files: {stats['total_files']}")
        print(f"   🗃️ Qdrant documents: {stats['qdrant_documents']}")
        if stats['file_types']:
            for file_type, count in list(stats['file_types'].items())[:3]:
                print(f"   📄 {file_type}: {count} files")
        if stats['last_sync']:
            print(f"   🕒 Last sync: {stats['last_sync']}")
    
    # Create QA chain with better retrieval settings
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10, "score_threshold": 0.1}  # More lenient search
        ),
        return_source_documents=True,
        verbose=False
    )
    
    # Start background sync
    start_background_sync()
    
    print(f"\n✅ Chatbot ready! (Auto-sync every {SYNC_INTERVAL//60} minutes)")
    print("📝 Knowledge base path:", KB_PATH)
    print("🗃️ Using Qdrant vector database")
    print("🔄 Background sync is ACTIVE - will check for changes every 2 minutes")
    print("💡 Commands: 'quit' or 'exit' to stop, 'stats' for KB info, 'reload' to force reload, 'test' to test search")
    print("-" * 70)
    
    chat_history = []
    
    try:
        while True:
            # Check if reload is needed
            qa_chain, vectordb = reload_chatbot_if_needed(qa_chain, vectordb)
            
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'stats':
                stats = get_knowledge_base_stats()
                if stats:
                    print(f"\n📊 Knowledge Base Statistics:")
                    print(f"   📁 Total files: {stats['total_files']}")
                    print(f"   🗃️ Qdrant documents: {stats['qdrant_documents']}")
                    for file_type, count in stats['file_types'].items():
                        print(f"   📄 {file_type}: {count} files")
                    if stats['last_sync']:
                        print(f"   🕒 Last sync: {stats['last_sync']}")
                continue
            
            if user_input.lower() == 'test':
                print("🔍 Testing vector search...")
                test_vector_search(vectordb, "AI technology triage")
                continue
            
            if user_input.lower() == 'reload':
                print("🔄 Force reloading chatbot...")
                llm, vectordb, embeddings = load_existing_vector_store()
                if llm and vectordb:
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=vectordb.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 10, "score_threshold": 0.1}
                        ),
                        return_source_documents=True,
                        verbose=False
                    )
                    print("✅ Chatbot reloaded successfully!")
                else:
                    print("❌ Failed to reload chatbot!")
                continue
            
            if not user_input:
                continue
            
            try:
                # Use invoke instead of the deprecated __call__ method
                result = qa_chain.invoke({"question": user_input, "chat_history": chat_history})
                
                response = result["answer"]
                source_docs = result.get("source_documents", [])
                
                # Display response
                print(f"\n🤖 Assistant: {response}")
                
                # Display source information
                if source_docs:
                    print(f"\n📚 Sources:")
                    print(format_source_info(source_docs))
                
                # Update chat history
                chat_history.append((user_input, response))
                
                # Keep chat history manageable (last 10 exchanges)
                if len(chat_history) > 10:
                    chat_history = chat_history[-10:]
                    
            except Exception as e:
                logger.error(f"❌ Query processing error: {e}")
                print(f"❌ Sorry, I encountered an error: {e}")
                print("Please try rephrasing your question or check the logs.")
                
    except KeyboardInterrupt:
        print("\n\n👋 Chatbot interrupted by user. Goodbye!")
    except Exception as e:
        logger.error(f"❌ Unexpected error in main loop: {e}")
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        # Clean up resources
        stop_background_sync()
        cleanup_resources()

def manual_sync():
    """Manual sync function that can be called directly"""
    print("🔄 Running manual sync...")
    try:
        if run_sync_process():
            print("✅ Manual sync completed successfully!")
            return True
        else:
            print("❌ Manual sync failed!")
            return False
    except Exception as e:
        logger.error(f"❌ Manual sync error: {e}")
        print(f"❌ Manual sync error: {e}")
        return False

def check_kb_directory():
    """Check if KB directory exists and has supported files"""
    kb_dir = Path(KB_PATH)
    if not kb_dir.exists():
        logger.error(f"❌ Knowledge base directory not found: {KB_PATH}")
        print(f"❌ Knowledge base directory not found: {KB_PATH}")
        print("Please ensure the KB directory exists and contains supported files.")
        return False
    
    # Count supported files
    supported_files = []
    for file_path in kb_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            if not file_path.name.startswith('.') and not file_path.name.startswith('~'):
                supported_files.append(file_path)
    
    if not supported_files:
        logger.warning(f"⚠️ No supported files found in {KB_PATH}")
        print(f"⚠️ No supported files found in {KB_PATH}")
        print(f"Supported file types: {', '.join(SUPPORTED_EXTENSIONS)}")
        return False
    
    logger.info(f"✅ Found {len(supported_files)} supported files in KB directory")
    return True

def interactive_setup():
    """Interactive setup for first-time users"""
    print("\n🔧 First-time setup detected!")
    print("Let me help you get started...")
    
    # Check KB directory
    if not check_kb_directory():
        print(f"\n📁 Please ensure your knowledge base directory exists at: {KB_PATH}")
        print("   And contains supported file types: PDF, TXT, CSV, DOCX, MD, PY, JS, HTML, XML, JSON")
        return False
    
    # Check .env file
    if not Path(".env").exists():
        print("\n🔑 Creating .env file...")
        api_key = input("Please enter your Google API Key: ").strip()
        if api_key:
            with open(".env", "w") as f:
                f.write(f"GOOGLE_API_KEY={api_key}\n")
                # Add Qdrant configuration
                qdrant_url = input("Please enter your Qdrant URL (or press Enter for default): ").strip()
                if qdrant_url:
                    f.write(f"QDRANT_URL={qdrant_url}\n")
                
                qdrant_api_key = input("Please enter your Qdrant API Key (or press Enter if not needed): ").strip()
                if qdrant_api_key:
                    f.write(f"QDRANT_API_KEY={qdrant_api_key}\n")
                    
            print("✅ .env file created!")
        else:
            print("❌ API key is required!")
            return False
    
    # Run initial sync
    print("\n🔄 Running initial sync...")
    if manual_sync():
        print("✅ Setup completed successfully!")
        return True
    else:
        print("❌ Setup failed during sync!")
        return False

if __name__ == "__main__":
    try:
        # Check if this is first run (fixed the condition)
        if not Path(SQLITE_PATH).exists():
            if not interactive_setup():
                print("❌ Setup failed. Please check the requirements and try again.")
                sys.exit(1)
        
        # Start the main chatbot
        main()
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)