#!/usr/bin/env python3
"""
Google Gemini CLI Tool

A command-line interface for interacting with Google's Gemini models.
"""

import os
import sys
import configparser
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import google.generativeai as genai

# Default configuration
DEFAULT_CONFIG = {
    'DEFAULT': {
        'api_key': '',
    },
    'CHAT': {
        'model': 'gemini-1.5-pro',
        'temperature': '0.7',
        'max_tokens': '2048',
    },
    'OUTPUT': {
        'format': 'markdown',
        'stream': 'True',
    },
    'SAFETY': {
        'safety_filter': 'BLOCK_MEDIUM_AND_ABOVE',
    },
}

class GeminiCLI:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Gemini CLI client."""
        self.config = self._load_config(config_path)
        self._init_gemini()
        self.chat = None

    def _load_config(self, config_path: Optional[str] = None) -> configparser.ConfigParser:
        """Load configuration from file or use defaults."""
        config = configparser.ConfigParser()
        
        # Set defaults
        for section, options in DEFAULT_CONFIG.items():
            config[section] = options
        
        # Load from file if exists
        if config_path and os.path.exists(config_path):
            config.read(config_path)
        
        # Check for environment variable
        if not config['DEFAULT'].get('api_key') and 'GEMINI_API_KEY' in os.environ:
            config['DEFAULT']['api_key'] = os.environ['GEMINI_API_KEY']
        
        return config

    def _init_gemini(self) -> None:
        """Initialize the Gemini API client."""
        api_key = self.config['DEFAULT'].get('api_key')
        if not api_key:
            print("Error: No API key found. Please set GEMINI_API_KEY environment variable or add it to the config file.")
            sys.exit(1)
        
        genai.configure(api_key=api_key)

    def start_chat(self) -> None:
        """Start a new chat session."""
        model_name = self.config['CHAT']['model']
        generation_config = {
            'temperature': float(self.config['CHAT']['temperature']),
            'max_output_tokens': int(self.config['CHAT']['max_tokens']),
        }
        
        safety_settings = self._get_safety_settings()
        
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        self.chat = model.start_chat(history=[])
        print(f"Started new chat with {model_name}")
        print("Type 'exit' to end the session.\n")

    def _get_safety_settings(self) -> list:
        """Get safety settings from config."""
        safety_filter = self.config['SAFETY'].get('safety_filter', 'BLOCK_MEDIUM_AND_ABOVE')
        return [
            {
                "category": category,
                "threshold": safety_filter
            }
            for category in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT"
            ]
        ]

    def run_chat(self) -> None:
        """Run the interactive chat session."""
        if not self.chat:
            self.start_chat()
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ('exit', 'quit', 'q'):
                    print("\nEnding chat session.")
                    break
                
                if not user_input:
                    continue
                
                # Stream the response
                print("\nAssistant: ", end='', flush=True)
                
                response = self.chat.send_message(
                    user_input,
                    stream=bool(self.config['OUTPUT'].getboolean('stream', True))
                )
                
                if hasattr(response, 'parts'):  # For streaming
                    for chunk in response:
                        print(chunk.text, end='', flush=True)
                    print()
                else:  # For non-streaming
                    print(response.text)
                
            except KeyboardInterrupt:
                print("\n\nUse 'exit' to end the session.")
            except Exception as e:
                print(f"\nError: {str(e)}")

def main():
    """Main entry point for the Gemini CLI."""
    parser = argparse.ArgumentParser(description='Google Gemini CLI Tool')
    parser.add_argument('--config', type=str, default='gemini_config.ini',
                      help='Path to config file')
    parser.add_argument('--model', type=str, help='Model to use')
    parser.add_argument('--temperature', type=float, help='Temperature for generation')
    
    args = parser.parse_args()
    
    try:
        cli = GeminiCLI(args.config)
        
        # Override config from command line
        if args.model:
            cli.config['CHAT']['model'] = args.model
        if args.temperature is not None:
            cli.config['CHAT']['temperature'] = str(args.temperature)
        
        print("Google Gemini CLI")
        print("----------------")
        print(f"Model: {cli.config['CHAT']['model']}")
        print(f"Temperature: {cli.config['CHAT']['temperature']}")
        print("Type your message (or 'exit' to quit)\n")
        
        cli.run_chat()
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
