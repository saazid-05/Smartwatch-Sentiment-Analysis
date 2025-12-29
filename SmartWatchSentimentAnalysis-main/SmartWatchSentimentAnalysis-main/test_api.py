"""
Test the actual API endpoint
"""
import requests

url = "http://localhost:8000/predict"

# Test data
test_reviews = [
    ("This smartwatch is amazing! Love it!", "classical"),
    ("Terrible! Waste of money!", "classical"),
    ("It's okay. Does the job.", "classical"),
]

print("="*60)
print("TESTING API ENDPOINT")
print("="*60)

for text, model_type in test_reviews:
    data = {
        'text': text,
        'model_type': model_type
    }
    
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nReview: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Model: {result['model_used']}")
    else:
        print(f"\nError: {response.status_code}")
        print(response.text)
