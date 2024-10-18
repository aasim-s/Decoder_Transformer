import time
from RegexTokenizer import RegexTokenizer

with open("taylorswift.txt", 'r') as file:
    test_data = file.read()
t0 = time.time()

tokenizer = RegexTokenizer()
tokenizer.train(test_data, vocab_size=32768, verbose=True)

# vocab size is 0 to 32767 so first ST starts from 32768
tokenizer.register_special_tokens({"<|endoftext|>": 32768})

tokenizer.save("test32")
tokenizer.load("test32.model")

t1 = time.time()
print(f"Traiing took {t1-t0:.2f} seconds")
