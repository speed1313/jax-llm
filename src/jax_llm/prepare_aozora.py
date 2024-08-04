from datasets import load_dataset
import click


@click.command()
@click.option("--book_num", type=int, default=10)
def main(book_num: int):
    data_name = f"aozora_{book_num}"
    import os

    os.makedirs(f"data/{data_name}", exist_ok=True)
    save_path = f"data/{data_name}/input.txt"
    ds = load_dataset(
        "globis-university/aozorabunko-clean", cache_dir="data/aozora/cache"
    )
    ds = ds.filter(lambda row: row["meta"]["文字遣い種別"] == "新字新仮名")
    # concat each bokk with <|endoftext|> token
    print(f"{book_num} books out of {len(ds['train'])} are used")
    with open(save_path, "w") as f:
        for i, book in enumerate(ds["train"]):
            if i > book_num:
                break
            # remove lines with only a few characters(ignore whitespace)
            lines = book["text"].split("\n")
            lines = [line.replace("\u3000", " ") for line in lines]
            # remove head whitespace
            lines = [line.strip() for line in lines]
            lines = [line for line in lines if len(line.replace(' ', ''))>= 10]
            book["text"] = "\n".join(lines)
            f.write(book["text"])
            f.write("<|endoftext|>")


if __name__ == "__main__":
    main()
