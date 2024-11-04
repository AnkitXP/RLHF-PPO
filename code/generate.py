import torch
from model import PolicyModel
from transformers import AutoTokenizer
from config import config

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PolicyModel(config, trainable=False).to(device)
    model = model.load_model(config.model_dir + args.model_name) if args.model_name is not None else model
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Set up prompt and parameters for generation
    prompt = input("Enter the prompt: ")  # Get the input prompt from the user
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate text based on the input prompt
    with torch.no_grad():
        generated_ids = model.gpt2.generate(
            input_ids,
            max_length=50,  
            num_return_sequences=1,  
            pad_token_id=tokenizer.eos_token_id,  
            do_sample=True,  
            top_k=50,  
            top_p=0.95,  
            temperature=1.0,
        )

    # Decode the generated tokens to text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Output the generated text
    print(f"\nGenerated Sequence:\n{generated_text}")

