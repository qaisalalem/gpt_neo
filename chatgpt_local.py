import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Download and load the GPT-2 model
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_response(input_text, model, tokenizer, max_length=200, num_return_sequences=1):
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=max_length, truncation=True)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences, no_repeat_ngram_size=3, do_sample=True, top_k=50, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    while True:
        input_text = input("Enter your question: ")
        if input_text.lower() == "exit":
            break
        response = generate_response(input_text, model, tokenizer)
        print(response)
