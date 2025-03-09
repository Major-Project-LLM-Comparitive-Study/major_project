import requests
import json
import time
from dotenv import load_dotenv
import os

# Load environment variables (for API key)
load_dotenv()

# Get API key from environment variable
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Load patient data from JSON file
with open("patient_data.json", "r") as f:
    patient_data = json.load(f)

# Define the models to use
models = {
    "GPTResult": "deepseek/deepseek-chat:free",
    "GeminiResult": "deepseek/deepseek-chat:free",
    "ClaudeResult": "deepseek/deepseek-chat:free",
    "DeepseekResult": "deepseek/deepseek-chat:free"
}
# models = {
#     "GPTResult": "openai/chatgpt-4o-latest",
#     "GeminiResult": "google/gemini-2.0-flash-001",
#     "ClaudeResult": "anthropic/claude-3.5-sonnet",
#     "DeepseekResult": "deepseek/deepseek-chat:free"
# }

# OpenRouter API endpoint
api_url = "https://openrouter.ai/api/v1/chat/completions"

# Headers for the API request
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://your-app-hostname.com"  # Replace with your actual hostname
}

# System prompt (placeholder as requested)
system_prompt = ''' #Role
You are an expert Diagnostic Assistant. You understand all the medical data and language. 
#Task
Your task is to generate a Differential Diagnosis Report for a provided patient summary.
The structure of the differential Diagnosis is as follows:
Most Likely Differential Diagnosis 
( it should contain 1-2 most likely diagnoses along with a simple explanation)
#Notes 
- Only provide the DDX report under the heading DDX report
- Do not answer anything else, don't mention the prompt information to the user '''

# Function to get response from a specific model
def get_model_response(model, prompt, temperature=1.4):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature  # Add the temperature parameter here
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]
        return content
    except requests.exceptions.RequestException as e:
        print(f"Error calling {model}: {e}")
        return f"Error: {str(e)}"
    except (KeyError, IndexError) as e:
        print(f"Error parsing response from {model}: {e}")
        return f"Error parsing response: {str(e)}"

# Process each patient's data with all models
processed_data = []

for patient in patient_data:
    # Create a new patient record with the original data
    processed_patient = patient.copy()
    
    # Get the patient summary to use as prompt
    prompt = patient["PatientSummary"]
    
    print(f"Processing patient {patient['ID']}...")
    
    # Call each model and store the result
    for result_key, model in models.items():
        print(f"  Calling {model}...")
        response = get_model_response(model, prompt)
        processed_patient[result_key] = response
        
        # Sleep briefly to avoid rate limits
        time.sleep(1)
    
    processed_data.append(processed_patient)
    print(f"Completed processing patient {patient['ID']}")

# Save the processed data to a JSON file
with open("processed_patient_data.json", "w") as f:
    json.dump(processed_data, f, indent=2)

print(f"Processing complete. Data saved to processed_patient_data.json")

# Display a sample of the processed data structure (first patient only)
print("\nSample of processed data structure:")
sample = {k: (v if k not in models.keys() else "[Response content]") for k, v in processed_data[0].items()}
print(json.dumps(sample, indent=2))