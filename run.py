#!/usr/bin/env python3
"""
Simple script to run the Bird Sound Classification API
"""

import os
import sys
from app import app, initialize_app

if __name__ == '__main__':
    # Initialize the application
    if not initialize_app():
        print("❌ Failed to initialize API")
        sys.exit(1)
    
    # Get configuration from environment
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"\n🚀 Starting Bird Sound Classification API")
    print(f"🌐 Server: http://{host}:{port}")
    print(f"📝 API Documentation: http://{host}:{port}")
    print(f"🔧 Debug mode: {'ON' if debug else 'OFF'}")
    print("\n📡 Available endpoints:")
    print(f"   GET  http://{host}:{port}/health")
    print(f"   GET  http://{host}:{port}/species") 
    print(f"   POST http://{host}:{port}/predict")
    print(f"   POST http://{host}:{port}/predict_url")
    print("\n✋ Press Ctrl+C to stop\n")
    
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 API server stopped")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1) 