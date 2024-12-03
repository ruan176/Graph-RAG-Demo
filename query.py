import requests
import logging

def query_llama(prompt, model="llama3"):
    """
    Function to query local llama model.
    """

    # Define parameters to send api request.
    url = "http://localhost:11434/api/generate"  # Default Ollama API endpoint
    headers = {
        "Content-Type": "application/json", 
        "Accept": "application/json"
    }
    data = {
        "model": model,  # This could be any model installed locally using ollama.
        "prompt": prompt,
        "max_tokens": 100,
        "stream": False,
    }

    try:
        # Send a request to the local instance.
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raise an error for HTTP codes 4xx/5xx

        # Extract and return the response text
        response_json = response.json()
        content = response_json['response']
        
        return content
    
    except requests.exceptions.RequestException as e:
        logging.info(f"An error occurred: {e}")