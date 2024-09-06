import os, time 
from src import BasicTokenizer, GPTTokenizer 

text = open("input.txt", "r", encoding="utf-8").read()

os.makedirs("models", exist_ok=True)

start_time = time.time()

name = "basic" # name of tokenizer
tokenizer = BasicTokenizer()
tokenizer.train(text, 512, verbose=True)

prefix = os.path.join("models", name)
tokenizer.save(prefix)

end_time = time.time()

print(f"Time Taken: {end_time - start_time:.2f} seconds.")
