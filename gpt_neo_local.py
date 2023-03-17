import os
import torch
from torch.utils.data import Dataset
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def generate_response(input_text, model, tokenizer, max_length=50, num_return_sequences=1):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    inputs = tokenizer(input_text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    inputs = inputs.to('cuda')  # Move input tensors to GPU
    model.to('cuda')  # Move the model to GPU
    outputs = model.generate(**inputs, max_length=max_length + 50, num_return_sequences=num_return_sequences, no_repeat_ngram_size=3, do_sample=True, top_k=100, temperature=1.0)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def load_dataset(text_file, tokenizer):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=text_file,
        block_size=128,
    )
    return dataset

def fine_tune(model, tokenizer, train_dataset):
    training_args = TrainingArguments(
        output_dir="./chatgpt_output",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=2,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    model_name = "EleutherAI/gpt-neo-1.3B"
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set pad_token for tokenizer
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    train_file = "text_data.txt"
    train_dataset = load_dataset(train_file, tokenizer)

    fine_tune(model, tokenizer, train_dataset)

    while True:
        input_text = input("Enter your question: ")
        if input_text.lower() == "exit":
            break
        response = generate_response(input_text, model, tokenizer)
        print(response)


