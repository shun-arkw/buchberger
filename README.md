# 環境構築

<!-- ビルドは`docker.sh`参照
```
    bash docker.sh [build|shell|root|help]
``` -->

docker imageの作成
```
bash sh/docker.sh build
```

コンテナの起動
```
bash sh/docker.sh shell
```

セットアップは，コンテナ内で以下を実行
```
bash sh/setup.sh
```

jupyter notebook
```
sage -n jupyter --port=8888 --ip=0.0.0.0
```
# buchberger_algorithm
