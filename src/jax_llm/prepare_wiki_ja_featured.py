import os
import requests

# URLと保存ディレクトリの定義
url = "http://www.lsta.media.kyoto-u.ac.jp/resource/data/wikitext-ja/Featured_Contents.txt"
save_dir = "data/wiki-ja-featured-2"
save_path = os.path.join(save_dir, "input.txt")

# 保存ディレクトリが存在しない場合は作成
os.makedirs(save_dir, exist_ok=True)

# ファイルをダウンロードして保存
response = requests.get(url)
if response.status_code == 200:
    with open(save_path, "wb") as file:
        file.write(response.content)
    print(f"File successfully downloaded and saved to {save_path}")
else:
    print(f"Failed to download file. HTTP Status Code: {response.status_code}")
