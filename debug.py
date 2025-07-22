# === Debug Version - sync.py ===
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# === FIX 1: Set UTF-8 encoding for stdout/stderr (Windows only) ===
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

def is_running_in_docker():
    """Check if script is running inside Docker container"""
    return os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == 'true'

def debug_env_loading():
    """Debug environment variable loading"""
    print("=== DEBUGGING ENVIRONMENT LOADING ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python script location: {__file__}")
    
    # Check if .env file exists
    env_files = ['.env', './.env', '../.env']
    for env_file in env_files:
        if os.path.exists(env_file):
            print(f"Found .env file at: {os.path.abspath(env_file)}")
            with open(env_file, 'r') as f:
                content = f.read()
                print(f"Content of {env_file}:")
                print(content)
                print("---")
        else:
            print(f"No .env file found at: {os.path.abspath(env_file)}")
    
    # Load environment variables
    load_dotenv(verbose=True)  # Enable verbose output
    
    print(f"KB_PATH from os.getenv(): {os.getenv('KB_PATH')}")
    print(f"All environment variables with KB:")
    for key, value in os.environ.items():
        if 'KB' in key.upper():
            print(f"  {key} = {value}")
    print("=====================================")

def get_kb_path():
    """Get the correct KB path based on environment with debugging"""
    debug_env_loading()
    
    if is_running_in_docker():
        # Inside Docker container, use the mounted path
        kb_path = "/app/chatbotKB_test"
        print(f"Docker environment detected, using: {kb_path}")
    else:
        # Outside Docker, use the local path from .env or default
        env_kb_path = os.getenv('KB_PATH')
        print(f"Local environment detected")
        print(f"KB_PATH from .env: {env_kb_path}")
        
        if env_kb_path:
            kb_path = env_kb_path
            # Convert forward slashes to backslashes for Windows
            if os.name == 'nt':  # Windows
                kb_path = kb_path.replace('/', '\\')
                print(f"Converted to Windows path: {kb_path}")
        else:
            kb_path = r'C:\Users\maria selciya\Desktop\chatbotKB_test'
            print(f"No KB_PATH in .env, using default: {kb_path}")
    
    return kb_path

# Set KB_PATH once at module level
KB_PATH = get_kb_path()

print(f"Environment: {'Docker' if is_running_in_docker() else 'Local'}")
print(f"Final KB_PATH: {KB_PATH}")

# Check if path exists and provide detailed info
if os.path.exists(KB_PATH):
    print(f"✅ KB_PATH exists: {KB_PATH}")
    # List contents
    try:
        contents = list(os.listdir(KB_PATH))
        print(f"Contents ({len(contents)} items): {contents[:10]}{'...' if len(contents) > 10 else ''}")
    except Exception as e:
        print(f"Error reading directory: {e}")
else:
    print(f"❌ KB_PATH does not exist: {KB_PATH}")
    
    # Check if parent directory exists
    parent_dir = os.path.dirname(KB_PATH)
    if os.path.exists(parent_dir):
        print(f"✅ Parent directory exists: {parent_dir}")
        try:
            contents = list(os.listdir(parent_dir))
            print(f"Parent contents: {contents}")
        except Exception as e:
            print(f"Error reading parent directory: {e}")
    else:
        print(f"❌ Parent directory also doesn't exist: {parent_dir}")
    
    print("\nSuggested fixes:")
    print("1. Create the directory:")
    print(f'   mkdir "{KB_PATH}"')
    print("2. Or update your .env file with the correct path")
    print("3. Or check if there's another .env file being loaded")

print("\n" + "="*50)
print("DEBUG COMPLETE - Script will exit here for debugging")
print("="*50)
sys.exit(0)
