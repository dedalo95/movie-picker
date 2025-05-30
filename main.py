#!/usr/bin/env python3
"""
Movie Picker - Tinder-style app for movies with AI recommendations
"""

import os
import sys

# Add the src folder to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

try:
    from app import app
except ImportError:
    print("Error importing modules.")
    print("Make sure all dependencies are installed with:")
    print("   poetry install")
    sys.exit(1)

if __name__ == '__main__':
    print("Starting Movie Picker...")
    print("Server available at: http://localhost:5001")
    print("Tinder-style interface for movies")
    print("AI recommendations integrated")
    print("=" * 50)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        print("\nShutting down Movie Picker...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)
