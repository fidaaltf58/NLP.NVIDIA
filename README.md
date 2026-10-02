# Authorship Attribution with BERT: NVIDIA DLI NLP Assessment

My completed coding assessment from an **NVIDIA Deep Learning Institute (DLI)** natural language processing course. It fine-tunes a **BERT** text classifier with **NVIDIA NeMo** to work out who wrote the disputed **Federalist Papers**: Alexander Hamilton or James Madison.

---

## The problem

The *Federalist Papers* are 85 essays written in 1787–1788 by Alexander Hamilton, James Madison, and John Jay under the shared pseudonym *Publius*. The authorship of several papers is still disputed between Hamilton and Madison.

Authorship attribution is a **text classification** task where the label is the *author* rather than the topic. The question is whether a transformer language model can pick up the differences in writing style.

## Approach

| Step | What was done |
|---|---|
| **1. Prepare the data** | Converted `train.tsv` / `dev.tsv` to the NeMo text-classification format (`sentence<TAB>label`, no header) |
| **2. Model config** | `bert-base-uncased`, 2 classes (Hamilton = 0, Madison = 1), max sequence length 256, batch size 16, learning rate 1e-4 |
| **3. Trainer config** | 5 epochs, automatic mixed precision (AMP level `O1`, FP16) |
| **4. Train** | Ran NeMo's `text_classification_with_bert.py` with Hydra/OmegaConf command-line overrides |
| **5. Infer** | Restored the `.nemo` checkpoint and classified every chunk of the disputed papers (`test49`–`test57` and `test62`). Each paper is assigned to the author with the majority of chunk predictions. |

## Results

| Paper | Predicted author |
|---|---|
| 49, 50, 51, 52, 53 | Hamilton |
| 54 | **Madison** |
| 55, 56, 57, 62 | Hamilton |

Earlier SVM-based stylometry ([Fung, 2003](http://pages.cs.wisc.edu/~gfung/federalist.pdf)) attributes **all** the disputed papers to Madison. This baseline run mostly predicts Hamilton, which suggests room for improvement:

- a **cased** model (capitalisation carries style information)
- more epochs and a lower learning rate
- a larger language model, or a different `MAX_SEQ_LENGTH` (most chunks were truncated at 256 tokens)

The assessment grades the pipeline, not the final attribution.

## Tech stack

- **NVIDIA NeMo** (`TextClassificationModel`)
- **PyTorch Lightning** (trainer, AMP / FP16)
- **Hugging Face BERT** (`bert-base-uncased`)
- **OmegaConf / Hydra** for configuration
- Jupyter Notebook on an NVIDIA GPU (DLI cloud environment)

## Repository contents

| File | Description |
|---|---|
| `NLP. assessment.ipynb` | The full notebook: data prep, configuration, training logs, inference, and results |

## Running it yourself

The notebook was written for the NVIDIA DLI environment and uses its paths (`/dli/task/...`) and pre-installed NeMo examples. To run it elsewhere you need:

1. An NVIDIA GPU with CUDA
2. NeMo installed (`pip install nemo_toolkit[nlp]`), or the [NeMo container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo)
3. The NeMo `examples/nlp/text_classification` script and config
4. The Federalist Papers dataset split into `train.tsv`, `dev.tsv`, and `test49.tsv`–`test57.tsv` and `test62.tsv`

Then update `DATA_DIR`, `CONFIG_DIR`, and `TC_DIR` in the notebook.

## Author

**Fidaa Letaief** · [@fidaaltf58](https://github.com/fidaaltf58)
