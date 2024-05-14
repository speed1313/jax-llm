# jax-llm
JAX implementation of Large Language Models.
You can train GPT-2-like models with 青空文庫 ([aozora bunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean) dataset) or any other text dataset.
Model implementation is based on [NanoLM](https://optax.readthedocs.io/en/latest/_collections/examples/nanolm.html).

## How to use

###  Prepare the [aozora bunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean) dataset.

```bash
python3 src/jax_llm/prepare_aozora.py --book_num 100
```
This command generates a single text file. Currently, 10 books (80M Tokens) are used.

> [!NOTE]
> You can use any dataset for training by simply preparing a suitable txt file, without executing this command. For example, [Wikitext-JA](http://www.lsta.media.kyoto-u.ac.jp/resource/data/wikitext-ja) is a good choice.

###  Train the BPE (Byte Pair Encoding) tokenizer.
Specify the path to the text file created in the previous step. This process takes approximately 20 seconds.
```bash
python3 src/jax_llm/train_tokenizer.py --data_name "aozora_100"
```

###  Train the NanoLM model with aozora bunko-clean dataset.

This command takes about 2 minutes on CPU.
```bash
python3 src/jax_llm/train.py --data_name "aozora_100" --batch_size 8 --n_iterations 500 --n_freq_eval 100 --dropout_rate 0.0 --learning_rate 0.0005 --num_layers 4 --embed_size 128  --head_size 32 --num_heads 4
```

### Generate text with the trained model.
```bash
$ python3 src/jax_llm/generate.py --prompt "遠く" --data_name "aozora_100" --max_new_tokens 50
Output: 遠く 、 それが平衡をとりもどし 、 どうにかその衝撃をまぬがれたのだ 。 その精神がどんどん成長して 、 また 、 色のさめたたてがみや尾はもつれたうえに 、 色のさめたたてがみや尾はもつれたうえに 、 弔いのときの人の泣き声のようだった 。 わたしは 、 ほかに心をなだめてくれるものはない わたしは 、 まさに耳もとで咆哮するのを聞くと 、 マストはきしみ 、 それが平衡をとりもどし 、 ほかに心をなだめてくれるものはない 。 年老い 、 マストはきしみ 、 生命の潮を健康な流れにして血管に送りこむのだが 。 「 聞くところによると 。 じっさい 、 こんな陰鬱な思いはたちまちにして 。 しかし 、 色のさめたたてがみや尾はもつれたうえに 、 獲物をもとめているような気がした 、 こんな陰鬱な思いはたちまちにして 、 あのなつかしい
```

## Directory structure
```
src
data
├── aozora_10246
│   ├── input.txt
│   ├── config.json
│   ├── tokenizer.json
model
├── aozora_10246
│   ├── params.pkl
│   ├── config.json
│   ├── opt_state.pkl
```

## References
Special thanks to the following repositories, papers, and datasets.
- https://optax.readthedocs.io/en/latest/_collections/examples/nanolm.html
- https://github.com/rasbt/LLMs-from-scratch
- https://github.com/karpathy/nanoGPT
- https://github.com/openai/gpt-2
- [Radford et al., Language Models are Unsupervised Multitask Learners, 2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- https://github.com/lxaw/shijin

### aozora bunko-clean dataset
- akeyhero, https://qiita.com/akeyhero/items/b53eae1c0bc4d54e321f
- [akeyhero, aozora bunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean)
- 青空文庫, https://www.aozora.gr.jp/
