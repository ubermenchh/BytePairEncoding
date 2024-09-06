import os, time 
from basic import BasicTokenizer
from gpt import GPTTokenizer 

text = open("input.txt", "r", encoding="utf-8").read()

os.makedirs("models", exist_ok=True)

start_time = time.time()

name = "gpt4" # name of tokenizer
tokenizer = GPTTokenizer()
tokenizer.train(text, 512, verbose=True)

prefix = os.path.join("models", name)
tokenizer.save(prefix)

end_time = time.time()

print(f"Time Taken: {end_time - start_time:.2f} seconds.")
