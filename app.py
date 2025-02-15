from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Optional, Dict, Any, List, Tuple
import os
import json
import datetime
import sqlite3
import subprocess
import glob
import re
import base64
from pathlib import Path
import sys
import shutil
import requests
from PIL import Image
import io
import numpy as np
from datetime import datetime

app = FastAPI(title="DataWorks Operations Agent")

def is_safe_path(path: str) -> bool:
    """Check if the path is within the /data directory."""
    # Normalize the path to resolve any '..' components
    normalized_path = os.path.normpath(path)
    # Check if the path starts with /data/
    return normalized_path.startswith("/data/") or normalized_path == "/data"

def get_embedding(text: str) -> List[float]:
    """Get embedding for a text using OpenAI's API via HTTP request."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": text,
        "model": "text-embedding-3-small"
    }
    
    response = requests.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/embeddings",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"Error from OpenAI API: {response.text}")
    
    result = response.json()
    return result["data"][0]["embedding"]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_array = np.array(a)
    b_array = np.array(b)
    return np.dot(a_array, b_array) / (np.linalg.norm(a_array) * np.linalg.norm(b_array))

def call_llm(prompt: str) -> str:
    """Call GPT-4o-mini with a prompt via HTTP request."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for DataWorks Solutions."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
    }
    
    response = requests.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"Error from OpenAI API: {response.text}")
    
    result = response.json()
    return result["choices"][0]["message"]["content"]

def extract_email_from_text(text: str) -> str:
    """Use LLM to extract email from text."""
    prompt = f"""
    Extract the sender's email address from the following email message. 
    Return ONLY the email address, nothing else.
    
    {text}
    """
    return call_llm(prompt).strip()

def extract_credit_card_from_image(image_path: str) -> str:
    """Use LLM to extract credit card number from image via HTTP request."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Read and encode image
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system", 
                "content": "You are a helpful assistant for DataWorks Solutions."
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text", 
                        "text": "This image contains a credit card. Extract the 16-digit card number. Return ONLY the digits with no spaces or separators."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 100
    }
    
    response = requests.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"Error from OpenAI API: {response.text}")
    
    result = response.json()
    
    # Clean up the response to get just digits
    card_number = ''.join(filter(str.isdigit, result["choices"][0]["message"]["content"]))
    return card_number

def find_most_similar_comments(comments_path: str) -> Tuple[str, str]:
    """Find the most similar pair of comments using embeddings."""
    with open(comments_path, 'r') as f:
        comments = [line.strip() for line in f if line.strip()]
    
    # Get embeddings for all comments
    embeddings = [get_embedding(comment) for comment in comments]
    
    # Find most similar pair
    max_similarity = -1
    most_similar_pair = (0, 0)
    
    for i in range(len(comments)):
        for j in range(i+1, len(comments)):
            similarity = cosine_similarity(embeddings[i], embeddings[j])
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_pair = (i, j)
    
    return comments[most_similar_pair[0]], comments[most_similar_pair[1]]

def parse_task_with_llm(task_description: str) -> Dict:
    """Parse task description using LLM to identify task type and details."""
    prompt = f"""
    Analyze this task description and identify which task it corresponds to.
    Return a JSON with the following structure:
    {{
        "task_number": "A1", // The task number (A1-A10, B3-B10, or "custom" for others)
        "input_file": "path/to/input", // The input file path mentioned in the task
        "output_file": "path/to/output", // The output file path mentioned in the task
        "additional_params": {{}} // Any other relevant parameters for the specific task
    }}
    
    IMPORTANT SECURITY RULES:
    1. NEVER allow access to files outside the /data directory
    2. NEVER allow deletion of any files anywhere
    3. If the task appears to violate these rules, mark it as "invalid"
    
    The possible tasks are:
    A1. Install uv (if required) and run datagen.py with email as argument
    A2. Format the contents of a markdown file using prettier
    A3. Count days of week in a dates file
    A4. Sort contacts by last_name, first_name
    A5. Write first line of 10 most recent log files
    A6. Create index of markdown H1 headers
    A7. Extract sender's email address from email
    A8. Extract credit card number from image
    A9. Find most similar comments using embeddings
    A10. Query SQLite database for total sales

    B3. Fetch data from an API and save it
    B4. Clone a git repo and make a commit
    B5. Run a SQL query on a SQLite or DuckDB database
    B6. Extract data from (i.e. scrape) a website
    B7. Compress or resize an image
    B8. Transcribe audio from an MP3 file
    B9. Convert Markdown to HTML
    B10. Write an API endpoint that filters a CSV file and returns JSON data
    
    Task description:
    {task_description}
    """
    
    result = call_llm(prompt)
    try:
        parsed = json.loads(result)
        
        # Security check: Ensure input/output paths are within /data
        for key in ["input_file", "output_file"]:
            if key in parsed and parsed[key] and not is_safe_path(parsed[key]):
                parsed["task_number"] = "invalid"
                parsed["error"] = f"Security violation: {key} must be within /data directory"
        
        return parsed
    except json.JSONDecodeError:
        # If we couldn't get valid JSON, try to extract the JSON part
        match = re.search(r'({.*})', result, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        
        # Fallback to a minimal structure if all else fails
        return {
            "task_number": "unknown",
            "input_file": "",
            "output_file": "",
            "additional_params": {}
        }

async def execute_task_a1(user_email: str) -> Dict:
    """Install uv if required and run datagen.py with email as argument."""
    # Check if uv is installed
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        try:
            # Install uv
            subprocess.run(["pip", "install", "uv"], check=True)
        except subprocess.SubprocessError as e:
            return {"success": False, "message": f"Failed to install uv: {str(e)}"}
    
    # Download and run datagen.py
    try:
        script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
        r = requests.get(script_url)
        if r.status_code != 200:
            return {"success": False, "message": f"Failed to download datagen.py: HTTP {r.status_code}"}
        
        # Save the script
        with open("datagen.py", "w") as f:
            f.write(r.text)
        
        # Run the script with email as argument
        result = subprocess.run(
            [sys.executable, "datagen.py", user_email],
            capture_output=True,
            text=True,
            check=True
        )
        
        return {"success": True, "message": "Successfully ran datagen.py", "output": result.stdout}
    
    except Exception as e:
        return {"success": False, "message": f"Error running datagen.py: {str(e)}"}

async def execute_task_a2(input_file: str) -> Dict:
    """Format markdown file using prettier."""
    try:
        # Check if prettier is installed
        try:
            subprocess.run(["npx", "prettier", "--version"], capture_output=True, check=True)
        except:
            # Install prettier
            subprocess.run(["npm", "install", "--no-save", "prettier@3.4.2"], check=True)
        
        # Format the file in-place
        result = subprocess.run(
            ["npx", "prettier", "--write", input_file],
            capture_output=True,
            text=True,
            check=True
        )
        
        return {"success": True, "message": "Successfully formatted file"}
    
    except Exception as e:
        return {"success": False, "message": f"Error formatting file: {str(e)}"}

async def execute_task_a3(input_file: str, output_file: str, day_of_week: str = "Wednesday") -> Dict:
    """Count occurrences of a specific day in a list of dates."""
    try:
        with open(input_file, 'r') as f:
            dates = [line.strip() for line in f if line.strip()]
        
        # Map day names to their number in the week (0 = Monday, 6 = Sunday)
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6,
            "सोमवार": 0, "मंगलवार": 1, "बुधवार": 2, "गुरुवार": 3, "शुक्रवार": 4, "शनिवार": 5, "रविवार": 6,
            "திங்கள்": 0, "செவ்வாய்": 1, "புதன்": 2, "வியாழன்": 3, "வெள்ளி": 4, "சனி": 5, "ஞாயிறு": 6
        }
        
        day_number = day_map.get(day_of_week, 2)  # Default to Wednesday if not found
        
        count = 0
        for date_str in dates:
            try:
                # Try different date formats
                date_formats = [
                    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", 
                    "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y", "%Y.%m.%d"
                ]
                
                parsed_date = None
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                
                if parsed_date and parsed_date.weekday() == day_number:
                    count += 1
            except:
                continue
        
        # Write result to output file
        with open(output_file, 'w') as f:
            f.write(str(count))
        
        return {"success": True, "message": f"Found {count} {day_of_week}s"}
    
    except Exception as e:
        return {"success": False, "message": f"Error counting dates: {str(e)}"}

async def execute_task_a4(input_file: str, output_file: str) -> Dict:
    """Sort contacts by last_name, then first_name."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or "contacts" not in data:
            return {"success": False, "message": "Invalid contacts JSON format"}
        
        # Sort contacts
        sorted_contacts = sorted(
            data["contacts"],
            key=lambda x: (x.get("last_name", ""), x.get("first_name", ""))
        )
        
        # Update and write result
        data["contacts"] = sorted_contacts
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return {"success": True, "message": f"Sorted {len(sorted_contacts)} contacts"}
    
    except Exception as e:
        return {"success": False, "message": f"Error sorting contacts: {str(e)}"}

async def execute_task_a5(logs_dir: str, output_file: str) -> Dict:
    """Write first line of 10 most recent log files."""
    try:
        # Get all log files with their modification times
        log_files = []
        for file in glob.glob(os.path.join(logs_dir, "*.log")):
            mtime = os.path.getmtime(file)
            log_files.append((file, mtime))
        
        # Sort by modification time (newest first) and take top 10
        log_files.sort(key=lambda x: x[1], reverse=True)
        recent_logs = log_files[:10]
        
        # Extract first line from each file
        first_lines = []
        for file, _ in recent_logs:
            try:
                with open(file, 'r') as f:
                    first_line = f.readline().strip()
                    first_lines.append(first_line)
            except:
                first_lines.append(f"[Error reading {os.path.basename(file)}]")
        
        # Write to output file
        with open(output_file, 'w') as f:
            f.write('\n'.join(first_lines))
        
        return {"success": True, "message": f"Extracted first lines from {len(first_lines)} log files"}
    
    except Exception as e:
        return {"success": False, "message": f"Error processing log files: {str(e)}"}

async def execute_task_a6(docs_dir: str, output_file: str) -> Dict:
    """Create index of markdown H1 headers."""
    try:
        index = {}
        for file in glob.glob(os.path.join(docs_dir, "**/*.md"), recursive=True):
            try:
                with open(file, 'r') as f:
                    content = f.read()
                
                # Find first H1 header
                match = re.search(r'^# (.+)$', content, re.MULTILINE)
                if match:
                    relative_path = os.path.relpath(file, docs_dir)
                    index[relative_path] = match.group(1).strip()
            except:
                continue
        
        # Write index to output file
        with open(output_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        return {"success": True, "message": f"Created index with {len(index)} entries"}
    
    except Exception as e:
        return {"success": False, "message": f"Error creating markdown index: {str(e)}"}

async def execute_task_a7(input_file: str, output_file: str) -> Dict:
    """Extract sender's email address from email text."""
    try:
        with open(input_file, 'r') as f:
            email_text = f.read()
        
        # Extract email using LLM
        email_address = extract_email_from_text(email_text)
        
        # Write to output file
        with open(output_file, 'w') as f:
            f.write(email_address)
        
        return {"success": True, "message": f"Extracted email address: {email_address}"}
    
    except Exception as e:
        return {"success": False, "message": f"Error extracting email: {str(e)}"}

async def execute_task_a8(input_file: str, output_file: str) -> Dict:
    """Extract credit card number from image."""
    try:
        # Extract card number using LLM vision
        card_number = extract_credit_card_from_image(input_file)
        
        # Write to output file
        with open(output_file, 'w') as f:
            f.write(card_number)
        
        return {"success": True, "message": f"Extracted card number (last 4 digits: {card_number[-4:]})"}
    
    except Exception as e:
        return {"success": False, "message": f"Error extracting credit card: {str(e)}"}

async def execute_task_a9(input_file: str, output_file: str) -> Dict:
    """Find most similar comments using embeddings."""
    try:
        comment1, comment2 = find_most_similar_comments(input_file)
        
        # Write to output file
        with open(output_file, 'w') as f:
            f.write(f"{comment1}\n{comment2}")
        
        return {"success": True, "message": "Found most similar comment pair"}
    
    except Exception as e:
        return {"success": False, "message": f"Error finding similar comments: {str(e)}"}

async def execute_task_a10(db_file: str, output_file: str, ticket_type: str = "Gold") -> Dict:
    """Query SQLite database for total sales of specific ticket type."""
    try:
        # Connect to database
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute(
            "SELECT SUM(units * price) FROM tickets WHERE type = ?",
            (ticket_type,)
        )
        total_sales = cursor.fetchone()[0]
        conn.close()
        
        if total_sales is None:
            total_sales = 0
        
        # Write result to output file
        with open(output_file, 'w') as f:
            f.write(str(total_sales))
        
        return {"success": True, "message": f"Total {ticket_type} ticket sales: {total_sales}"}
    
    except Exception as e:
        return {"success": False, "message": f"Error querying ticket sales: {str(e)}"}

async def execute_task_b3(api_url: str, output_file: str) -> Dict:
    """Fetch data from an API and save it."""
    try:
        # Ensure output file is within /data
        if not is_safe_path(output_file):
            return {"success": False, "message": "Security violation: output must be within /data directory"}
        
        # Make the API request
        response = requests.get(api_url)
        response.raise_for_status()
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the response content
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        return {"success": True, "message": f"Successfully fetched data from API and saved to {output_file}"}
    
    except Exception as e:
        return {"success": False, "message": f"Error fetching API data: {str(e)}"}

async def execute_task_b4(repo_url: str, output_dir: str, commit_message: str) -> Dict:
    """Clone a git repo and make a commit."""
    try:
        # Ensure output directory is within /data
        if not is_safe_path(output_dir):
            return {"success": False, "message": "Security violation: output must be within /data directory"}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Clone the repository
        result = subprocess.run(
            ["git", "clone", repo_url, output_dir],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Create a simple file to commit
        commit_file = os.path.join(output_dir, "commit_file.txt")
        with open(commit_file, 'w') as f:
            f.write(f"Automated commit at {datetime.now().isoformat()}")
        
        # Add and commit the file
        subprocess.run(["git", "-C", output_dir, "add", "commit_file.txt"], check=True)
        subprocess.run(["git", "-C", output_dir, "commit", "-m", commit_message], check=True)
        
        return {"success": True, "message": f"Successfully cloned repo and made commit"}
    
    except Exception as e:
        return {"success": False, "message": f"Error with git operations: {str(e)}"}

async def execute_task_b5(db_file: str, query: str, output_file: str) -> Dict:
    """Run a SQL query on a SQLite or DuckDB database."""
    try:
        # Ensure input and output files are within /data
        if not is_safe_path(db_file) or not is_safe_path(output_file):
            return {"success": False, "message": "Security violation: files must be within /data directory"}
        
        # Determine database type by extension
        if db_file.endswith('.db') or db_file.endswith('.sqlite'):
            # SQLite database
            conn = sqlite3.connect(db_file)
        else:
            # Assume DuckDB if not SQLite
            try:
                import duckdb
                conn = duckdb.connect(db_file)
            except ImportError:
                return {"success": False, "message": "DuckDB not installed. Please install with 'pip install duckdb'"}
        
        # Execute query
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Fetch results
        results = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        # Convert to list of dictionaries
        result_dicts = []
        for row in results:
            result_dicts.append(dict(zip(column_names, row)))
        
        # Write results to output file
        with open(output_file, 'w') as f:
            json.dump(result_dicts, f, indent=2)
        
        conn.close()
        
        return {"success": True, "message": f"Query executed successfully with {len(results)} results"}
    
    except Exception as e:
        return {"success": False, "message": f"Error executing database query: {str(e)}"}

async def execute_task_b6(url: str, output_file: str) -> Dict:
    """Extract data from a website (web scraping)."""
    try:
        # Ensure output file is within /data
        if not is_safe_path(output_file):
            return {"success": False, "message": "Security violation: output must be within /data directory"}
        
        # Make the request
        response = requests.get(url)
        response.raise_for_status()
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the HTML content
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        return {"success": True, "message": f"Successfully scraped website and saved to {output_file}"}
    
    except Exception as e:
        return {"success": False, "message": f"Error scraping website: {str(e)}"}

async def execute_task_b7(input_file: str, output_file: str, width: int = 800, height: int = None) -> Dict:
    """Compress or resize an image."""
    try:
        # Ensure input and output files are within /data
        if not is_safe_path(input_file) or not is_safe_path(output_file):
            return {"success": False, "message": "Security violation: files must be within /data directory"}
        
        # Open the image
        img = Image.open(input_file)
        
        # Calculate new dimensions while maintaining aspect ratio if height is None
        if height is None:
            wpercent = (width / float(img.size[0]))
            height = int((float(img.size[1]) * float(wpercent)))
        
        # Resize the image
        resized_img = img.resize((width, height), Image.LANCZOS)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the resized image
        resized_img.save(output_file, optimize=True, quality=85)
        
        return {"success": True, "message": f"Successfully resized image to {width}x{height}"}
    
    except Exception as e:
        return {"success": False, "message": f"Error resizing image: {str(e)}"}

async def execute_task_b8(input_file: str, output_file: str) -> Dict:
    """Transcribe audio from an MP3 file using HTTP request."""
    try:
        # Ensure input and output files are within /data
        if not is_safe_path(input_file) or not is_safe_path(output_file):
            return {"success": False, "message": "Security violation: files must be within /data directory"}
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"success": False, "message": "OPENAI_API_KEY environment variable not set"}
        
        # Prepare the HTTP request
        url = "https://aiproxy.sanand.workers.dev/openai/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        with open(input_file, "rb") as audio_file:
            files = {
                "file": (os.path.basename(input_file), audio_file, "audio/mpeg"),
                "model": (None, "whisper-1")
            }
            
            response = requests.post(url, headers=headers, files=files)
            
            if response.status_code != 200:
                return {"success": False, "message": f"Error from OpenAI API: {response.text}"}
            
            result = response.json()
            transcript_text = result.get("text", "")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write the transcript to the output file
        with open(output_file, 'w') as f:
            f.write(transcript_text)
        
        return {"success": True, "message": f"Successfully transcribed audio to text"}
    
    except Exception as e:
        return {"success": False, "message": f"Error transcribing audio: {str(e)}"}

async def execute_task_b9(input_file: str, output_file: str) -> Dict:
    """Convert Markdown to HTML."""
    try:
        # Ensure input and output files are within /data
        if not is_safe_path(input_file) or not is_safe_path(output_file):
            return {"success": False, "message": "Security violation: files must be within /data directory"}
        
        # Read the markdown file
        with open(input_file, 'r') as f:
            markdown_text = f.read()
        
        # Use LLM to convert markdown to HTML
        prompt = f"""
        Convert this Markdown to clean, semantic HTML:
        
        {markdown_text}
        
        Return ONLY the HTML, nothing else.
        """
        
        html_content = call_llm(prompt)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write the HTML to the output file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return {"success": True, "message": f"Successfully converted Markdown to HTML"}
    
    except Exception as e:
        return {"success": False, "message": f"Error converting Markdown to HTML: {str(e)}"}

async def execute_task_b10(input_file: str, filter_column: str = None, filter_value: str = None) -> Dict:
    """Filter a CSV file and return JSON data."""
    try:
        # Ensure input file is within /data
        if not is_safe_path(input_file):
            return {"success": False, "message": "Security violation: input must be within /data directory"}
        
        # Read the CSV file
        import csv
        data = []
        with open(input_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Apply filter if specified
                if filter_column and filter_value:
                    if str(row.get(filter_column, "")) == filter_value:
                        data.append(row)
                else:
                    data.append(row)
        
        return {
            "success": True,
            "message": f"Successfully filtered CSV data",
            "data": data,
            "count": len(data)
        }
    
    except Exception as e:
        return {"success": False, "message": f"Error filtering CSV data: {str(e)}"}

@app.post("/run")
async def run_task(task: str):
    try:
        # Parse the task using LLM
        parsed_task = parse_task_with_llm(task)
        task_number = parsed_task.get("task_number", "unknown").upper()

        if task_number == "INVALID":
            return JSONResponse(
                content={
                    "success": False,
                    "message": "Security violation detected",
                    "error": parsed_task.get("error", "Unknown security violation")
                },
                status_code=403
            )
        
        result = {"task_number": task_number, "success": False, "message": "Task not implemented"}
        
        # Execute the appropriate task
        if task_number == "A1":
            email = parsed_task.get("additional_params", {}).get("email", "${user.email}")
            result = await execute_task_a1(email)
            
        elif task_number == "A2":
            input_file = parsed_task.get("input_file", "/data/format.md")
            result = await execute_task_a2(input_file)
            
        elif task_number == "A3":
            input_file = parsed_task.get("input_file", "/data/dates.txt")
            output_file = parsed_task.get("output_file", "/data/dates-wednesdays.txt")
            day = parsed_task.get("additional_params", {}).get("day_of_week", "Wednesday")
            result = await execute_task_a3(input_file, output_file, day)
            
        elif task_number == "A4":
            input_file = parsed_task.get("input_file", "/data/contacts.json")
            output_file = parsed_task.get("output_file", "/data/contacts-sorted.json")
            result = await execute_task_a4(input_file, output_file)
            
        elif task_number == "A5":
            logs_dir = parsed_task.get("input_file", "/data/logs/")
            output_file = parsed_task.get("output_file", "/data/logs-recent.txt")
            result = await execute_task_a5(logs_dir, output_file)
            
        elif task_number == "A6":
            docs_dir = parsed_task.get("input_file", "/data/docs/")
            output_file = parsed_task.get("output_file", "/data/docs/index.json")
            result = await execute_task_a6(docs_dir, output_file)
            
        elif task_number == "A7":
            input_file = parsed_task.get("input_file", "/data/email.txt")
            output_file = parsed_task.get("output_file", "/data/email-sender.txt")
            result = await execute_task_a7(input_file, output_file)
            
        elif task_number == "A8":
            input_file = parsed_task.get("input_file", "/data/credit-card.png")
            output_file = parsed_task.get("output_file", "/data/credit-card.txt")
            result = await execute_task_a8(input_file, output_file)
            
        elif task_number == "A9":
            input_file = parsed_task.get("input_file", "/data/comments.txt")
            output_file = parsed_task.get("output_file", "/data/comments-similar.txt")
            result = await execute_task_a9(input_file, output_file)
            
        elif task_number == "A10":
            db_file = parsed_task.get("input_file", "/data/ticket-sales.db")
            output_file = parsed_task.get("output_file", "/data/ticket-sales-gold.txt")
            ticket_type = parsed_task.get("additional_params", {}).get("ticket_type", "Gold")
            result = await execute_task_a10(db_file, output_file, ticket_type)
        
        # Include parsed task info in result
        elif task_number == "B3":
            api_url = parsed_task.get("additional_params", {}).get("api_url", "")
            output_file = parsed_task.get("output_file", "")
            if not api_url or not output_file:
                result = {"success": False, "message": "Missing required parameters: api_url and output_file"}
            else:
                result = await execute_task_b3(api_url, output_file)
                
        elif task_number == "B4":
            repo_url = parsed_task.get("additional_params", {}).get("repo_url", "")
            output_dir = parsed_task.get("output_file", "")
            commit_message = parsed_task.get("additional_params", {}).get("commit_message", "Automated commit")
            if not repo_url or not output_dir:
                result = {"success": False, "message": "Missing required parameters: repo_url and output_dir"}
            else:
                result = await execute_task_b4(repo_url, output_dir, commit_message)
                
        elif task_number == "B5":
            db_file = parsed_task.get("input_file", "")
            query = parsed_task.get("additional_params", {}).get("query", "")
            output_file = parsed_task.get("output_file", "")
            if not db_file or not query or not output_file:
                result = {"success": False, "message": "Missing required parameters: db_file, query, and output_file"}
            else:
                result = await execute_task_b5(db_file, query, output_file)
                
        elif task_number == "B6":
            url = parsed_task.get("additional_params", {}).get("url", "")
            output_file = parsed_task.get("output_file", "")
            if not url or not output_file:
                result = {"success": False, "message": "Missing required parameters: url and output_file"}
            else:
                result = await execute_task_b6(url, output_file)
                
        elif task_number == "B7":
            input_file = parsed_task.get("input_file", "")
            output_file = parsed_task.get("output_file", "")
            width = int(parsed_task.get("additional_params", {}).get("width", 800))
            height = parsed_task.get("additional_params", {}).get("height")
            if height is not None:
                height = int(height)
            if not input_file or not output_file:
                result = {"success": False, "message": "Missing required parameters: input_file and output_file"}
            else:
                result = await execute_task_b7(input_file, output_file, width, height)
                
        elif task_number == "B8":
            input_file = parsed_task.get("input_file", "")
            output_file = parsed_task.get("output_file", "")
            if not input_file or not output_file:
                result = {"success": False, "message": "Missing required parameters: input_file and output_file"}
            else:
                result = await execute_task_b8(input_file, output_file)
                
        elif task_number == "B9":
            input_file = parsed_task.get("input_file", "")
            output_file = parsed_task.get("output_file", "")
            if not input_file or not output_file:
                result = {"success": False, "message": "Missing required parameters: input_file and output_file"}
            else:
                result = await execute_task_b9(input_file, output_file)
                
        elif task_number == "B10":
            input_file = parsed_task.get("input_file", "")
            filter_column = parsed_task.get("additional_params", {}).get("filter_column")
            filter_value = parsed_task.get("additional_params", {}).get("filter_value")
            if not input_file:
                result = {"success": False, "message": "Missing required parameter: input_file"}
            else:
                result = await execute_task_b10(input_file, filter_column, filter_value)
        
        # Include parsed task info in result
        result["parsed_task"] = parsed_task
        
        if result.get("success", False):
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result, status_code=400)
            
    except Exception as e:
        return JSONResponse(
            content={"success": False, "message": f"Internal server error: {str(e)}"},
            status_code=500
        )

@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str):
    """Read and return the contents of a file."""
    # Security check: Ensure path is within /data directory
    if not is_safe_path(path):
        return JSONResponse(
            content={"error": "Access denied. Can only access files within /data directory."},
            status_code=403
        )
    
    try:
        if not os.path.exists(path):
            return Response(status_code=404)
        
        if os.path.isdir(path):
            return JSONResponse(
                content={"error": "Path is a directory, not a file."},
                status_code=400
            )
        
        with open(path, 'r') as f:
            content = f.read()
        
        return content
    
    except Exception as e:
        return JSONResponse(
            content={"error": f"Error reading file: {str(e)}"},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)