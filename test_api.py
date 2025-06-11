import requests
import json
import sys

def test_chatbot_api():
    print("Testing chatbot API...")
    
    # Replace with your actual server URL
    url = "http://127.0.0.1:8000/chat/"
    
    # Headers
    headers = {
        'Content-Type': 'application/json',
    }
    
    # Initial empty message to start conversation
    payload = json.dumps({
        "message": ""
    })
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("Test successful!")
        else:
            print("Test failed!")
    except Exception as e:
        print(f"Error: {e}")
        
if __name__ == "__main__":
    test_chatbot_api()