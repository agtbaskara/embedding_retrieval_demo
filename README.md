# embedding_retrieval_demo
embedding_retrieval_demo

tested only on x86_64 Machine with NVIDIA GPU

# Setup Environment

1. Download and Extract Dataset

    We are using a subset of IMDB Face dataset with only unique faces (~36.000 images) taken from https://github.com/fwang91/IMDb-Face

    Download pre-downloaded dataset here (~23 GB): https://1drv.ms/u/c/e5a63d34bccb21b3/IQSH33248xAjT5arssfB68NpAdcsj9uzr8w1dLHjJ54tH8E

    ```
    unzip IMDb-Face_clean_unique.zip
    ```

2. Build Docker Image

    Dockerfile already contain most of the library needed (if somehting missing then install it by yourself)

    ```
    docker build -t embedding_demo .
    ```

3. Run Docker Container

    ```
    docker compose up -d
    ```

# Prepare YOLO face detection weight

Download `yolov11s-face.pt` from https://github.com/akanametov/yolo-face

Put inside `models` directory

# Setup and Run rqlite + sqlite-vec

Download latest rqlite release: https://github.com/rqlite/rqlite/releases

Replace the version as needed, to extract

```
tar -xvf rqlite-v8.36.11-linux-amd64.tar.gz
```

Download latest sqlite-vec into the rqlite directory: https://github.com/asg017/sqlite-vec/releases

```
curl -L https://github.com/asg017/sqlite-vec/releases/download/v0.1.6/sqlite-vec-0.1.6-loadable-linux-x86_64.tar.gz -o sqlite-vec.tar.gz
```

Run rqlite with sqlite-vec enabled, `data` will be the database file location

```
./rqlited -extensions-path=sqlite-vec.tar.gz data
```

# Experiment List

- Experiment 1: Naive Face Matching

    test face embedding extraction and matching using edgeface model: https://github.com/otroshi/edgeface

- Experiment 2: FAISS Retrieval

    PoC of embedding retrievel using FAISS: https://github.com/facebookresearch/faiss

    Currently not working

- Experiment 3: sqlite-vec

    PoC of embedding retrievel using sqlite-vec: https://github.com/asg017/sqlite-vec

    Require yolo-face model

- Experiment 4: sqlite-vec query only

    Load a database and query directly, continuation of Experiment 3

    Require yolo-face model

- Experiment 5: sqlite-vec update only

    Add new image into the database, continuation of Experiment 3

    Require yolo-face model

- Experiment 6: rqlite + sqlite-vec

    PoC of embedding retrievel using rqlite: https://github.com/rqlite/rqlite

    This is to test how to use API with distributed database framework

    Require yolo-face model

    Require `rqlited` to run first

- Experiment 7: Realtime Query

    PoC of realtime update using video input `exp7_update.py` and simultaneous query

    Prepare `sample_video.mp4` for testing, "Never Gonna Give You Up"

    Require yolo-face model

    Require `rqlited` to run first
