# codellm

This project is adapted from [lit-llama](https://github.com/Lightning-AI/lit-llama) to pre-train a code LLM for educational purposes.

### Dataset

The model is trained on the [RedPajama-Data-1T-Sample](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample) dataset, which provides a diverse collection of code and text data from sources like GitHub, ArXiv, CommonCrawl, and Wikipedia. The data preparation process involves four steps:

1. Download the raw dataset (requires Git LFS):

```bash
# Install Git LFS if needed: brew install git-lfs
git lfs install

# Clone and pull dataset (approximately 3.6GB)
git clone https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample
cd RedPajama-Data-1T-Sample
git lfs pull
```

2. Get tokenizer codellama/CodeLlama-7b-hf by running

```bash
python preprocess/get_tokenizer.py
```

3. Process raw data into tokenized format:

```bash
# Creates redpajama_sample/ directory with tokenized data
python preprocess/prepare_redpajama.py
```

4. Split data into training and validation sets (90/10 split):

```bash
# Creates redpajama_train/ and redpajama_val/ directories
python preprocess/split_data.py
```

After splitting, you'll have the following distribution of files:

#### CommonCrawl (CC) Snapshots

| Source     | Training Files | Validation Files |
| ---------- | -------------- | ---------------- |
| cc_2023-06 | 3,006          | 334              |
| cc_2020-05 | 2,980          | 331              |
| cc_2021-04 | 2,838          | 315              |
| cc_2022-05 | 2,596          | 288              |
| cc_2019-30 | 2,447          | 271              |

#### Code and Knowledge Sources

| Source        | Training Files | Validation Files |
| ------------- | -------------- | ---------------- |
| github        | 853            | 94               |
| book          | 416            | 46               |
| arxiv         | 413            | 45               |
| wikipedia     | 327            | 36               |
| stackexchange | 305            | 33               |

#### Cleaned Web Data

| Source | Training Files | Validation Files |
| ------ | -------------- | ---------------- |
| c4     | 2,717          | 301              |

This maintains a consistent 90/10 split ratio across all data sources while ensuring at least one validation file per source.

After processing, your workspace will contain:

```
.
├── RedPajama-Data-1T-Sample/    # Raw data
    ├── .gitattributes
    ├── arxiv_sample.jsonl
    ├── book_sample.jsonl
    ├── c4_sample.jsonl
    ├── cc_2019-30_sample.jsonl
    ├── cc_2020-05_sample.jsonl
    ├── cc_2021-04_sample.jsonl
    ├── cc_2022-05_sample.jsonl
    ├── cc_2023-06_sample.jsonl
    ├── github_sample.jsonl
    ├── README.md
    ├── RedPajama-Data-1T-Sample.py
    ├── stackexchange_sample.jsonl
    └── wikipedia_sample.jsonl
├── redpajama_sample/         # Tokenized data
├── redpajama_train/          # Training split
└── redpajama_val/            # Validation split
```

Note: Ensure you have at least 10GB of free disk space for the dataset and its processed versions.

### Training Visualization

This project uses TensorBoard to track and visualize training metrics. The following metrics are tracked:

-   Training loss
-   Validation loss
-   Training speed (tokens/second)

#### Setup TensorBoard

```bash
# Launch TensorBoard (suppress warnings if needed)
PYTHONWARNINGS=ignore tensorboard --logdir=runs
```

Then open your browser and navigate to:

```
http://localhost:6006
```

The logs are organized by:

-   Date (YYYY-MM-DD)
-   Experiment name
-   Model name
-   Extra tags (if any)

### Progress

**02-17-2025**: Completed initial training of the model for 100,000 iterations on Apple M2 hardware. The model shows basic understanding of code structure but needs improvement in code completion quality. Here's a sample output from the model:

![progress#1](screenshots/progress1.png)
