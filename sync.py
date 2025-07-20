# === Optimized Knowledge Base Sync with Schema Migration ===
import os
import sqlite3
import hashlib
import json
import uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging
import pandas as pd

# Qdrant imports with fallback
try:
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    from langchain_community.vectorstores import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True

# DOCX support
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
KB_PATH = "C:/Users/maria selciya/Desktop/chatbotKB_test"
QDRANT_PATH = "./qdrant_storage"
SQLITE_PATH = "./file_index.db"
COLLECTION_NAME = "kb_collection"
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.csv', '.docx', '.md', '.py', '.js', '.html', '.xml', '.json'}

class KnowledgeBaseSync:
    def __init__(self):
        self.embedding_model = None
        self.vectorstore = None
        self.client = None
        self.conn = None
        self.cursor = None
        
    def initialize(self):
        """Initialize all components"""
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        # Initialize embeddings
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # Initialize Qdrant
        Path(QDRANT_PATH).mkdir(exist_ok=True)
        self.client = QdrantClient(path=QDRANT_PATH)
        
        try:
            self.client.get_collection(COLLECTION_NAME)
        except Exception:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding=self.embedding_model
        )
        
        # Initialize database with migration
        self.init_database()
        logger.info("✅ All components initialized")
    
    def check_column_exists(self, table_name, column_name):
        """Check if a column exists in a table"""
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [column[1] for column in self.cursor.fetchall()]
        return column_name in columns
    
    def migrate_database_schema(self):
        """Migrate database schema to latest version"""
        logger.info("🔄 Checking database schema...")
        
        # Check if table exists
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='file_index'")
        table_exists = self.cursor.fetchone() is not None
        
        if not table_exists:
            logger.info("📋 Creating new file_index table...")
            self.cursor.execute("""
                CREATE TABLE file_index (
                    filename TEXT PRIMARY KEY,
                    filepath TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER,
                    mtime REAL NOT NULL,
                    hash TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        else:
            # Check and add missing columns
            columns_to_add = [
                ('filepath', 'TEXT NOT NULL DEFAULT ""'),
                ('file_size', 'INTEGER DEFAULT 0'),
                ('chunk_count', 'INTEGER DEFAULT 0'),
                ('indexed_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
            ]
            
            for column_name, column_def in columns_to_add:
                if not self.check_column_exists('file_index', column_name):
                    logger.info(f"➕ Adding missing column: {column_name}")
                    self.cursor.execute(f"ALTER TABLE file_index ADD COLUMN {column_name} {column_def}")
            
            # Update filepath for existing records if empty
            self.cursor.execute("UPDATE file_index SET filepath = filename WHERE filepath = '' OR filepath IS NULL")
        
        self.conn.commit()
        logger.info("✅ Database schema migration completed")
    
    def init_database(self):
        """Initialize SQLite database with migration"""
        self.conn = sqlite3.connect(SQLITE_PATH, timeout=30)
        self.cursor = self.conn.cursor()
        
        # Perform schema migration
        self.migrate_database_schema()
    
    def get_file_hash(self, file_path):
        """Calculate file hash"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash {file_path}: {e}")
            return None
    
    def is_supported(self, file_path):
        """Check if file is supported"""
        return file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    
    def load_document(self, file_path):
        """Load document based on file type"""
        docs = []
        ext = file_path.suffix.lower()
        
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                docs = [doc for doc in docs if doc.page_content.strip()]
            
            elif ext == '.docx' and DOCX_AVAILABLE:
                doc = DocxDocument(str(file_path))
                content = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                if content.strip():
                    docs = [Document(page_content=content, metadata={'source': str(file_path)})]
            
            elif ext == '.csv':
                df = pd.read_csv(file_path)
                if not df.empty:
                    content = f"CSV: {file_path.name}\nColumns: {', '.join(df.columns)}\nRows: {len(df)}\n\n{df.to_string(max_rows=100)}"
                    docs = [Document(page_content=content, metadata={'source': str(file_path)})]
            
            elif ext in ['.txt', '.md', '.py', '.js', '.html', '.xml', '.json']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.strip():
                    docs = [Document(page_content=content, metadata={'source': str(file_path)})]
            
            # Add metadata
            for doc in docs:
                doc.metadata.update({
                    'filename': file_path.name,
                    'file_type': ext,
                    'processed_at': datetime.now().isoformat()
                })
            
            return [doc for doc in docs if doc.page_content.strip()]
            
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
            return []
    
    def remove_from_vectorstore(self, filename):
        """Remove all vectors for a specific file"""
        try:
            # Delete points with matching filename
            self.client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
                )
            )
            logger.info(f"🗑️ Removed vectors for: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove vectors for {filename}: {e}")
            return False
    
    def add_to_vectorstore(self, file_path):
        """Add file to vector store"""
        try:
            filename = file_path.name
            
            # Remove existing vectors first
            self.remove_from_vectorstore(filename)
            
            # Load documents
            docs = self.load_document(file_path)
            if not docs:
                logger.warning(f"No content to index for {filename}")
                return False
            
            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = []
            for doc in docs:
                doc_chunks = splitter.split_documents([doc])
                chunks.extend(doc_chunks)
            
            # Add filename to all chunks
            for chunk in chunks:
                chunk.metadata['filename'] = filename
                chunk.metadata['chunk_id'] = str(uuid.uuid4())
            
            # Add to vectorstore
            self.vectorstore.add_documents(chunks)
            
            # Update database
            file_stats = file_path.stat()
            file_hash = self.get_file_hash(file_path)
            
            self.cursor.execute("""
                INSERT OR REPLACE INTO file_index 
                (filename, filepath, file_type, file_size, mtime, hash, chunk_count, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (filename, str(file_path), file_path.suffix.lower(), 
                  file_stats.st_size, file_stats.st_mtime, file_hash, len(chunks)))
            self.conn.commit()
            
            logger.info(f"✅ Added {filename}: {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add {file_path.name}: {e}")
            return False
    
    def remove_file(self, filename):
        """Completely remove file from database and vectorstore"""
        try:
            # Remove from vectorstore
            self.remove_from_vectorstore(filename)
            
            # Remove from database
            self.cursor.execute("DELETE FROM file_index WHERE filename = ?", (filename,))
            self.conn.commit()
            
            logger.info(f"🗑️ Completely removed: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove {filename}: {e}")
            return False
    
    def get_current_files(self):
        """Get current files from filesystem"""
        kb_dir = Path(KB_PATH)
        if not kb_dir.exists():
            raise FileNotFoundError(f"KB directory not found: {KB_PATH}")
        
        current_files = {}
        for file_path in kb_dir.rglob("*"):
            if (file_path.is_file() and 
                not file_path.name.startswith('.') and 
                self.is_supported(file_path)):
                
                file_hash = self.get_file_hash(file_path)
                if file_hash:
                    current_files[file_path.name] = {
                        'path': file_path,
                        'mtime': file_path.stat().st_mtime,
                        'hash': file_hash
                    }
        
        return current_files
    
    def get_database_files(self):
        """Get files from database"""
        self.cursor.execute("SELECT filename, mtime, hash FROM file_index")
        return {row[0]: {'mtime': row[1], 'hash': row[2]} for row in self.cursor.fetchall()}
    
    def print_current_files(self):
        """Print current files in database"""
        self.cursor.execute("""
            SELECT filename, file_type, chunk_count, indexed_at 
            FROM file_index 
            ORDER BY indexed_at DESC
        """)
        files = self.cursor.fetchall()
        
        print("\n" + "="*60)
        print("📋 CURRENT DATABASE FILES:")
        print("="*60)
        
        if files:
            for filename, file_type, chunk_count, indexed_at in files:
                print(f"✅ {filename} ({file_type}) - {chunk_count} chunks")
        else:
            print("📭 No files in database")
        
        print(f"📊 Total: {len(files)} files")
        print("="*60)
    
    def rebuild_database(self):
        """Rebuild database from scratch"""
        logger.info("🔄 Rebuilding database from scratch...")
        
        # Drop and recreate table
        self.cursor.execute("DROP TABLE IF EXISTS file_index")
        self.cursor.execute("""
            CREATE TABLE file_index (
                filename TEXT PRIMARY KEY,
                filepath TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER DEFAULT 0,
                mtime REAL NOT NULL,
                hash TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        logger.info("✅ Database rebuilt")
    
    def sync(self, force_rebuild=False):
        """Main sync function - optimized for real-time updates"""
        try:
            logger.info("🚀 Starting knowledge base sync...")
            
            if not self.client:
                self.initialize()
            
            # If force rebuild, rebuild database
            if force_rebuild:
                self.rebuild_database()
            
            # Get current state
            current_files = self.get_current_files()
            db_files = self.get_database_files()
            
            print(f"📂 Found {len(current_files)} supported files in directory")
            
            # Find changes
            to_add = []
            to_remove = []
            
            # Check for new/modified files
            for filename, file_info in current_files.items():
                if (force_rebuild or 
                    filename not in db_files or 
                    abs(db_files[filename]['mtime'] - file_info['mtime']) > 1 or
                    db_files[filename]['hash'] != file_info['hash']):
                    to_add.append(file_info['path'])
            
            # Check for deleted files
            for filename in db_files:
                if filename not in current_files:
                    to_remove.append(filename)
            
            # Process changes
            added_count = 0
            failed_count = 0
            
            # Remove deleted files
            for filename in to_remove:
                if self.remove_file(filename):
                    print(f"🗑️ REMOVED: {filename}")
                else:
                    print(f"❌ FAILED TO REMOVE: {filename}")
            
            # Add/update files
            for file_path in to_add:
                print(f"🔄 PROCESSING: {file_path.name}")
                if self.add_to_vectorstore(file_path):
                    added_count += 1
                    print(f"✅ ADDED: {file_path.name}")
                else:
                    failed_count += 1
                    print(f"❌ FAILED: {file_path.name}")
            
            # Show final state
            self.print_current_files()
            
            # Summary
            print(f"\n🎉 SYNC COMPLETE!")
            print(f"✅ Added/Updated: {added_count}")
            print(f"🗑️ Removed: {len(to_remove)}")
            print(f"❌ Failed: {failed_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        if self.conn:
            self.conn.close()

# Main execution functions
def sync_kb(force_rebuild=False):
    """Convenient function to sync knowledge base"""
    syncer = KnowledgeBaseSync()
    try:
        return syncer.sync(force_rebuild)
    finally:
        syncer.cleanup()

def list_files():
    """List current files in database"""
    syncer = KnowledgeBaseSync()
    try:
        syncer.initialize()
        syncer.print_current_files()
    finally:
        syncer.cleanup()

# CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--rebuild":
        sync_kb(force_rebuild=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_files()
    else:
        sync_kb()