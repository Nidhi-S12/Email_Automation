#!/usr/bin/env python3
"""
Gmail Dashboard Server
Run this script to start the email dashboard web interface
"""
import uvicorn
import sys
import os
from pathlib import Path

# Add src to path so we can import our modules
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def main():
    """Start the dashboard server"""
    try:
        # Import after adding to path
        from gmail_crew_ai.dashboard import app
        
        print("🚀 Starting Gmail Dashboard...")
        print("📊 Dashboard will be available at: http://localhost:8080")
        print("📖 API docs available at: http://localhost:8080/docs")
        print("🔄 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Run the dashboard server
        uvicorn.run(
            "gmail_crew_ai.dashboard:app",
            host="0.0.0.0",
            port=8080,
            reload=True,
            app_dir=str(src_path)
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the project root directory and dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
