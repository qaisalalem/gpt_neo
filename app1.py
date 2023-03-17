from flask import Flask, render_template, request
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import nltk
import os

app = Flask(__name__)

# Download the Brown Corpus
nltk.download('brown')

# Load the GPT-Neo model and tokenizer
model_name = 'EleutherAI/gpt-neo-1.3B'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Preprocess and clean the corpus
corpus = nltk.corpus.brown.sents()
corpus = [sentence for sentence in corpus if len(sentence) > 10]
corpus_text = '\n'.join([' '.join(sentence) for sentence in corpus])
corpus_tokenized = tokenizer.encode(corpus_text)

# Save the tokenized corpus to a file
with open('corpus.txt', 'w') as f:
    f.write('\n'.join([str(token) for token in corpus_tokenized]))

# Create a TextDataset and DataCollatorForLanguageModeling
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='corpus.txt',  # Pass the file containing the tokenized corpus
    block_size=128
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

# Fine-tune the model on the corpus
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10000,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=1000,
    logging_first_step=True,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()

# Define the response generation function using the fine-tuned model
def generate_response(input_text, model, tokenizer, max_length=50, num_return_sequences=1):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    inputs = tokenizer(input_text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    inputs = inputs.to('cuda')
    model.to('cuda')
    outputs = model.generate(**inputs, max_length=max_length + 50, num_return_sequences=num_return_sequences, no_repeat_ngram_size=3, do_sample=True, top_k=100, temperature=1.0)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Define the Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        response = generate_response(user_input, model, tokenizer)
        return render_template('index.html', user_input=user_input, response=response)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

# Clean up
os.remove('corpus.txt')

