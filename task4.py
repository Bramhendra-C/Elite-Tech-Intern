# Install required packages if needed
# !pip install transformers torch

from transformers import pipeline, set_seed

# Load pre-trained GPT-2 model
generator = pipeline('text-generation', model='gpt2')
set_seed(42)

# User Prompt
prompt = "The impact of artificial intelligence on education is"

# Generate text
output = generator(prompt, max_length=150, num_return_sequences=1)

# Display
print("Generated Text:\n")
print(output[0]['generated_text'])
