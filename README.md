# jax-llm
JAX implementation of Large Language Models.
You can train GPT-2-like models with 青空文庫 ([aozora bunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean) dataset) or any other text dataset.
Model implementation is based on [NanoLM](https://optax.readthedocs.io/en/latest/_collections/examples/nanolm.html).

## How to use

###  Prepare the [aozora bunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean) dataset.

```bash
python3 src/jax_llm/prepare_aozora.py --book_num 10246
```
This command generates a single text file. We use 10246 books (80M Tokens).

> [!NOTE]
> You can use any dataset for training by simply preparing a suitable txt file, without executing this command. For example, [Wikitext-JA](http://www.lsta.media.kyoto-u.ac.jp/resource/data/wikitext-ja) is a good choice.

###  Train the BPE (Byte Pair Encoding) tokenizer.
Specify the path to the text file created in the previous step. This process takes approximately 20 seconds.
```bash
python3 src/jax_llm/train_tokenizer.py --data_name "aozora_10246"
```

###  Train the NanoLM model with aozora bunko-clean dataset.

```bash
python3 src/jax_llm/train.py --data_name "aozora_10246" --batch_size 128 --n_iterations 5000 --n_freq_eval 100 --dropout_rate 0.1 --learning_rate 0.001 --num_layers 12 --embed_size 512  --head_size 64 --num_heads 8 --block_size 256
```
This command takes about 20 miniutes with a single GPU (A100).
If you don't have a GPU, you can scale down the model size by reducing the dataset size and the model size.

![loss_dynamic](./train_loss_aozora_10246.png)

### Generate text with the trained model.
```bash
$ python3 src/jax_llm/generate.py --prompt "国境の 長いトンネルを抜けると雪国であった。" --data_name "aozora_10246" --max_new_tokens 200

Output: 国境 の長い トンネル を抜 けると 雪 国 であった 。 「 オヤ 、 この 寒い ところ 、 こんなに 早くから 寒 気が する 。 それでも 、 その 熱 湯 は 湯 から 上って 来る 。 そうして 、 小 川は 、 「 どうした もんだ ッ 。 この 石 置き 場の 石 地蔵 へ 、 この 小屋 へと 、 と まって 、 お とし 穴 の上 におし こめられて しまいました 。 お しまいに 、 お へや から 、 お 位牌 や 、 位牌 や 、 お 位牌 と一緒に 、 お 線香 を持って 、 お 線香 を持って 帰って来た ら 、 お 涌 が 帰って くれ と云 いたい ことがある 。 お 涌 さんの ことは 、 もう とっくに 承知 していた のでございます 。」 と 、 彼女は いいました 。 娘は 、 彼女の 頭を じっと 握り 開いて 、 「 私は もう 死んだ 方が ええ ですから 」 彼女は 、 そう 云うと 、 「 いや 、 そんなこと ばかり 。 お前 はお 祖父 さんと 一緒 に出 掛けて 、 それから お 妾 、 又 、 お 婿 様 をお 連れ 申 したい 」 「 はい 。 どうぞ 」 「 はい 、 はい 」 と 、 小 女が 小 男に いった 。 「 この 娘を 、 どう 思って いい かわからない ね 」 「 はい 、 あの 通りの 、 お 二人が 、 その 、 小 太郎の
```



## Training with [Wikitext-JA](http://www.lsta.media.kyoto-u.ac.jp/resource/data/wikitext-ja) dataset.
###  Prepare the Wikitext-JA dataset.
- Download the dataset from [here](http://www.lsta.media.kyoto-u.ac.jp/resource/data/wikitext-ja).
```bash
python3 src/jax_llm/prepare_wiki_ja_featured.py
```
- Substitute "aozora_10246" with "wiki-ja-featured" in the commands in the `How to use` section.



## References
Special thanks to the following repositories, papers, and datasets.
- https://optax.readthedocs.io/en/latest/_collections/examples/nanolm.html
- https://github.com/rasbt/LLMs-from-scratch
- https://github.com/karpathy/nanoGPT
- https://github.com/openai/gpt-2
- [Radford et al., Language Models are Unsupervised Multitask Learners, 2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- https://github.com/lxaw/shijin

### Dataset
- akeyhero, https://qiita.com/akeyhero/items/b53eae1c0bc4d54e321f
- [akeyhero, aozora bunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean)
  青空文庫, https://www.aozora.gr.jp/
- [Wikitext-JA](http://www.lsta.media.kyoto-u.ac.jp/resource/data/wikitext-ja)