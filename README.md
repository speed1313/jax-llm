# jax-llm
JAX implementation of Large Language Models.
We can train GPT-2-like model with 青空文庫 ([aozora bunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean) dataset).

## How to use

###  Prepare the [aozora bunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean) dataset.

```bash
cd src/jax_llm
rye run python3 prepare_aozora.py --book_num 100
```
This command generates a single text file. Currently, only 100 books are used.

> [!NOTE]
> You can use any dataset for training by simply preparing a suitable txt file, without executing this command.

###  Train the BPE (Byte Pair Encoding) tokenizer.
Specify the path to the text file created in the previous step. This process takes approximately 10 seconds.
```bash
rye run python3 train_tokenizer.py --data_path "aozora.txt"
```

###  Train GPT-2-like model with [aozora bunko-clean](https://huggingface.co/datasets/globis-university/Aozorabunko-clean) dataset.
```bash
rye run python3 generate.py --tokenizer_path "data/tokenizer-aozora.json" --model_path "model/aozora_variables.pkl""
```
Hyperparameters of the model can be adjusted in `src/jax_llm/train.py`'s `GPTConfig` dataclass.

### Generate text with the trained model.
```bash
rye run python3 generate.py --tokenizer_path "data/tokenizer-aozora.json" --model_path "model/aozora_variables.pkl" --prompt "深いおどろきにうたれて、" --temperature 0.7 --max_length 50 --top_k 30
```




## Acknowledgements
Rasbt's implementation and explanation is very helpful. I learned a lot from this work. Thank you very much!
- [rasbt, Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)

Special thanks to akeyhero for providing the aozora-clean dataset!
- [akeyhero, aozora bunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean)

Karpathy's nanoGPT is also very helpful. Karpathy's projects are always very interesting. Thank you very much!
- [karpathy, nanoGPT](https://github.com/karpathy/nanoGPT)

## References
- https://github.com/rasbt/LLMs-from-scratch
- https://github.com/karpathy/nanoGPT
- https://github.com/openai/gpt-2
- [Radford et al., Language Models are Unsupervised Multitask Learners, 2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- https://github.com/cgarciae/nanoGPT-jax

### aozora bunko-clean dataset
- akeyhero, https://qiita.com/akeyhero/items/b53eae1c0bc4d54e321f
- [akeyhero, aozora bunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean)
- 青空文庫, https://www.aozora.gr.jp/