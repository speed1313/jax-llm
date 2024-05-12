from datasets import load_dataset

ds = load_dataset("globis-university/aozorabunko-clean", cache_dir="data")
print(ds)

ds = ds.filter(lambda row: row["meta"]["文字遣い種別"] == "新字新仮名")

print(ds)


# concat with <|endoftext|> token
data = ""

with open("aozora.txt", "w") as f:
    for i, book in enumerate(ds["train"]):
        if i > 10:
            break
        print(book["text"])
        f.write(book["text"])
        f.write("<|endoftext|>")
