<div align="center">

<h1>Molecule Attention Transformer Compression</h1>

<b> Authors: </b> <a href="https://albmlpy.github.io/">Albert Saiapin</a>

<p align="center">
  <a href=""><img src="assets/compression.png?raw=True" alt="Figure 1" width="60%"></a>
</p>

</div>


## 💫 Project Description
This project studies **model compression techniques for molecular transformers**, using the [Molecule Attention Transformer (MAT)](https://arxiv.org/abs/2002.08264) for molecular solubility prediction.

### Setting
- **Dataset**: ESOL molecular solubility dataset
- **Task**: Regression of experimental solubility (logS)
- **Input**: Molecular structure represented through atom features, connectivity, and inter-atomic distances

### Approach
Different compression strategies are applied to selected MAT layers and compared with a dense baseline:

- Tensor Train (TT) decomposition
- Low-rank factorization
- Magnitude pruning

Different compression levels and layer selections are evaluated to understand the trade-off between model size and predictive performance.

#### The goal is to identify which compression approaches and model components provide the best accuracy–efficiency trade-off for molecular transformers.


## ⚙️ Environment
To install required python packages `conda` package manager is used. Once `conda is installed`, **run** the following command (while in the root of the repository):
```shell
conda env create -f environment.yml
```
This will create a new environment named `mat_compress_env` with all required packages already installed. You can **install** additional packages by running:
```shell
conda install <package name>
```

In order to read and run `Jupyter Notebooks` you may follow either of two options:
1. [*recommended*] using notebook-compatibility features of IDEs, e.g. via `python` and `jupyter` extensions of [VS Code](https://code.visualstudio.com/).
2. install jupyter notebook packages:
  either with `conda install jupyterlab` or with `conda install jupyter notebook`

## 🚀 How to Reproduce the Numerical Experiments

1. **Open** `experiment.ipynb` in VS Code and select the `mat_compress_env` environment (should be installed beforehand), or open the notebook using Jupyter Lab.

2. **Run** all cells in the notebook sequentially. The results will be displayed in the notebook and saved to:
  - `artifacts/results/` — experimental results
  - `artifacts/training_artifacts/` - training artifacts and logs

## 🔨 Useful References and Tools Used

- [Molecule Attention Transformer](https://arxiv.org/abs/2002.08264)
- [Tensor Methods for LLMs](https://github.com/ma-tt-a/awesome-tensor-methods-for-llms)
- OpenAI ChatGPT / GitHub Copilot 
