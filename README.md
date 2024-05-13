# jax-llm
JAX implementation of Large Language Models.
You can train GPT-2-like model with 青空文庫 ([aozora bunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean) dataset).
Model implementation is based on [NanoLM](https://optax.readthedocs.io/en/latest/_collections/examples/nanolm.html).

## How to use

###  Prepare the [aozora bunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean) dataset.

```bash
cd src/jax_llm
rye run python3 prepare_aozora.py --book_num 1000
```
This command generates a single text file. Currently, only 1000 books (9359840 Tokens) are used.

> [!NOTE]
> You can use any dataset for training by simply preparing a suitable txt file, without executing this command. For example, [Wikitext-JA's Featured Contents(1037109 Tokens)](http://www.lsta.media.kyoto-u.ac.jp/resource/data/wikitext-ja/Featured_Contents.txt) is a good choice.

###  Train the BPE (Byte Pair Encoding) tokenizer.
Specify the path to the text file created in the previous step. This process takes approximately 20 seconds.
```bash
rye run python3 train_tokenizer.py --data_path "input.txt"
```

###  Train [NanoLM model](https://optax.readthedocs.io/en/latest/_collections/examples/nanolm.html) with [aozora bunko-clean](https://huggingface.co/datasets/globis-university/Aozorabunko-clean) dataset.
```bash
rye run python3 train.py
```
Hyperparameters of the model can be adjusted in `src/jax_llm/train.py`.


### Generate text with the trained model.
```bash
rye run python3 generate.py  --prompt "深いおどろきにうたれて、" --temperature 0.7 --max_length 50 --top_k 30
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

### aozora bunko-clean dataset
- akeyhero, https://qiita.com/akeyhero/items/b53eae1c0bc4d54e321f
- [akeyhero, aozora bunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean)
- 青空文庫, https://www.aozora.gr.jp/