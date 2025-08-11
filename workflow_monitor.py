#!/usr/bin/env python3
"""
Workflow Monitor for Website-to-Agent Application
Filters and displays only workflow-related logs in a clean format.
"""

import time
import os
import re
from datetime import datetime

def follow_workflow_logs(filename='app.log'):
    """
    Follow the log file and display only workflow-related entries.
    """
    print("🚀 Website-to-Agent Workflow Monitor")
    print("=" * 50)
    print("Monitoring workflow steps in real-time...")
    print("Press Ctrl+C to exit\n")
    
    # Workflow keywords to filter for
    workflow_keywords = [
        'WORKFLOW START', 'STEP 1', 'STEP 2', 'STEP 3', 'WORKFLOW COMPLETE',
        'CHAT START', 'CHAT PROCESSING', 'CHAT COMPLETE', 'CHAT ERROR',
        'FIRECRAWL START', 'FIRECRAWL COMPLETE', 'FIRECRAWL ERROR',
        'Starting knowledge extraction', 'Knowledge extraction completed',
        'Creating specialized agent', 'Domain agent created'
    ]
    
    if not os.path.exists(filename):
        print(f"❌ Log file '{filename}' not found. Make sure the app is running.")
        return
    
    # Open and follow the file
    with open(filename, 'r') as file:
        # Go to end of file
        file.seek(0, 2)
        
        try:
            while True:
                line = file.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                
                # Check if line contains workflow keywords
                line_stripped = line.strip()
                if any(keyword in line_stripped for keyword in workflow_keywords):
                    # Clean up the line for display
                    formatted_line = format_workflow_line(line_stripped)
                    if formatted_line:
                        print(formatted_line)
                        
        except KeyboardInterrupt:
            print("\n\n👋 Workflow monitoring stopped.")

def format_workflow_line(line):
    """
    Format a workflow log line for nice display.
    """
    # Extract timestamp and message
    match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - .+ - .+ - (.+)', line)
    if not match:
        return None
    
    timestamp_str, message = match.groups()
    
    # Parse timestamp and format it nicely
    try:
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        time_formatted = timestamp.strftime('%H:%M:%S')
    except:
        time_formatted = timestamp_str
    
    # Add color and formatting based on message type
    if 'ERROR' in message or '❌' in message:
        color = '\033[91m'  # Red
    elif 'COMPLETE' in message or '✅' in message:
        color = '\033[92m'  # Green
    elif 'START' in message or '🚀' in message:
        color = '\033[94m'  # Blue
    elif 'PROCESSING' in message or '🔄' in message:
        color = '\033[93m'  # Yellow
    else:
        color = '\033[96m'  # Cyan
    
    reset_color = '\033[0m'
    
    return f"{color}[{time_formatted}] {message}{reset_color}"

def show_recent_workflow():
    """
    Show the last few workflow entries from the log file.
    """
    filename = 'app.log'
    if not os.path.exists(filename):
        print(f"❌ Log file '{filename}' not found.")
        return
    
    print("📋 Recent Workflow Activity:")
    print("-" * 40)
    
    workflow_keywords = [
        'WORKFLOW START', 'STEP 1', 'STEP 2', 'STEP 3', 'WORKFLOW COMPLETE',
        'CHAT START', 'CHAT PROCESSING', 'CHAT COMPLETE', 'CHAT ERROR',
        'FIRECRAWL START', 'FIRECRAWL COMPLETE', 'FIRECRAWL ERROR',
        'Starting knowledge extraction', 'Knowledge extraction completed',
        'Creating specialized agent', 'Domain agent created'
    ]
    
    recent_entries = []
    with open(filename, 'r') as file:
        for line in file:
            line_stripped = line.strip()
            if any(keyword in line_stripped for keyword in workflow_keywords):
                formatted = format_workflow_line(line_stripped)
                if formatted:
                    recent_entries.append(formatted)
    
    # Show last 10 entries
    for entry in recent_entries[-10:]:
        print(entry)
    
    print(f"\nShowing last {min(10, len(recent_entries))} workflow entries")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--recent':
        show_recent_workflow()
    else:
        follow_workflow_logs()
