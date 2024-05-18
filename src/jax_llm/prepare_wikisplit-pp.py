from datasets import load_dataset
import os
import click

@click.command()
@click.option(
    "--field",
    type=str,
    default="complex",
    help="Name of the dataset",
)
def main(field: str):

    save_dir = "data/wikisplit-pp-{field}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "input.txt")

    dataset = load_dataset("cl-nagoya/wikisplit-pp")
    print('dataset["train"] length: ', len(dataset["train"]))

    with open(save_path, "w") as file:
        for i, data in enumerate(dataset["train"]):
            file.write(data[field])
            file.write("<|endoftext|>")


    print("Data saved to", save_path)

if __name__ == "__main__":
    main()