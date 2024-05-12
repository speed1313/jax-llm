# jax-llm
JAX implementation of Large Language Models.
We can train GPT-2-like model with 青空文庫([aozora bunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean?row=0) dataset).

## How to use
- Prepare [aozora bunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean?row=0) dataset.
One txt file will be created. We use only 100 books for now.
```bash
cd src/jax_llm
rye run python3 prepare_aozora.py --book_num 100
```

- Train BPE(Byte Pair Encoding) tokenizer.
We need to specify the path of the txt file created in the previous step. It takes about 10 seconds.
```bash
rye run python3 train_tokenizer.py --data_path "aozora.txt"
```

- Train GPT-2-like model with aozora bunko dataset.
```bash
rye run python3 generate.py --tokenizer_path "data/tokenizer-aozora.json" --model_path "model/aozora_variables.pkl""
```
We can change hyperparameters of the model in `src/jax_llm/train.py`'s `GPTConfig` dataclass.

- Generate text with the trained model.
```bash
rye run python3 generate.py --tokenizer_path "data/tokenizer-aozora.json" --model_path "model/aozora_variables.pkl"
```


## Acknowledgements
Rasbt's implementation and explanation is very helpful. I learned a lot from this work. Thank you very much!
- [rasbt, Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)

Thank you for providing aozora-clean dataset!
- [akeyhero, aozora bunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean?row=0)

Karpathy's nanoGPT is also very helpful. Karpathy's projects are always very interesting. Thank you very much!
- [karpathy, nanoGPT](https://github.com/karpathy/nanoGPT)

## References
- https://github.com/rasbt/LLMs-from-scratch
- https://github.com/karpathy/nanoGPT
- https://github.com/openai/gpt-2
- [Radford et al., Language Models are Unsupervised Multitask Learners, 2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- https://github.com/cgarciae/nanoGPT-jax

### aozora-clean dataset
- akeyhero, https://qiita.com/akeyhero/items/b53eae1c0bc4d54e321f
- [akeyhero, aozora bunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean?row=0)
- 青空文庫, https://www.aozora.gr.jp/