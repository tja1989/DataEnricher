#!/usr/bin/env python3
"""
Runner script for OpenAI embedder with API key handling
"""

import os
import sys
import getpass

# Check if API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("="*60)
    print("OpenAI API Key Required")
    print("="*60)
    print("\nThe OPENAI_API_KEY environment variable is not set.")
    print("\nYou have two options:")
    print("1. Set it permanently: export OPENAI_API_KEY='your-key'")
    print("2. Enter it now (temporary for this session only)")
    
    choice = input("\nEnter your API key now? (y/n): ").strip().lower()
    
    if choice == 'y':
        api_key = getpass.getpass("Enter your OpenAI API key: ").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            print("✓ API key set for this session")
        else:
            print("❌ No API key provided. Exiting.")
            sys.exit(1)
    else:
        print("\nPlease set your API key using:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\nThen run this script again.")
        sys.exit(1)

# Now import and run the embedder
from openai_embedder import main

if __name__ == "__main__":
    exit(main())