# Notebooks

### 実行環境について
GPU のメモリは 30 GB 用意するべし．

### ライブラリについて
torchtext==0.11 で実行するべし．\
その他は適宜用意するべし．\
torchtext をインストールすると，torchtext に合わせた\
バージョンの pytorch がインストールされるので注意するべし．

### Language Model について
spaCy の言語モデルを利用．\
ドイツ語は "de_core_news_sm" を利用\
英語は "en_core_web_sm" を利用\
両方とも事前にダウンロードしておくべし．
