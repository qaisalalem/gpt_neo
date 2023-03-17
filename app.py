from flask import Flask, render_template, request
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

app = Flask(__name__)

model_name = "EleutherAI/gpt-neo-2.7B"
model = GPTNeoForCausalLM.from_pretrained(model_name)
#model = GPTNeoForCausalLM.from_pretrained(model_name, use_auth_token=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

import torch

def generate_response(input_text, model, tokenizer, max_length=50, num_return_sequences=1):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    inputs = tokenizer(input_text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    inputs = inputs.to('cuda')  # Move input tensors to GPU
    model.to('cuda')  # Move the model to GPU
    outputs = model.generate(**inputs, max_length=max_length + 50, num_return_sequences=num_return_sequences, no_repeat_ngram_size=3, do_sample=True, top_k=100, temperature=1.0)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        response = generate_response(user_input, model, tokenizer)
        return render_template('index.html', user_input=user_input, response=response)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
