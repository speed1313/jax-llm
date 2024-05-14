from datasets import load_dataset
import click


@click.command()
@click.option("--book_num", type=int, default=10)
def main(book_num: int):
    import os

    os.makedirs("data/aozora", exist_ok=True)
    save_path = "data/aozora/input.txt"
    ds = load_dataset(
        "globis-university/aozorabunko-clean", cache_dir="data/aozora/cache"
    )
    ds = ds.filter(lambda row: row["meta"]["文字遣い種別"] == "新字新仮名")
    # concat each bokk with <|endoftext|> token
    print(len(ds["train"]))
    with open(save_path, "w") as f:
        for i, book in enumerate(ds["train"]):
            if i > book_num:
                break
            f.write(book["text"])
            f.write("<|endoftext|>")


if __name__ == "__main__":
    main()
