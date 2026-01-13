# Music Interpretation using basic sparse autoencoder

Music Interpret is a project meant for extracting, analyzing, and interpreting
internal representations of music transformer models. It is implementation of a 
base **ReLU + L1 sparase autoencoder**.

This project provides a workflow for:

- Extracting token-level activations from trained music transformer models
- Training sparse autoencoders (SAEs)
- Tying tokens to SAE's latent activations that fired them 

The focus of this repository is analysis and interpretability, not
training music transformer models. Current version of the project is not 
yet adjusted to universally work on variety of model architectures and is 
highly depended on base project **Multitrack Music Transformer (MMT)**. 

See Acknowledgements section for details.

## Project Organization

```
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with convenience commands like 'init'
├── README.md          <- The top-level README for developers using this project.
├── mkdocs.yml         <- MkDocs configuration file for project documentation
│
├── data
│   ├── base           <- Preprocessed note datasets used by the base model.
│   └── activations    <- Extracted model activations and associated metadata.
│
├── docs               <- Project documentation built with MkDocs.
│
├── models             <- Base model checkpoints.
│
├── notebooks          <- Jupyter notebooks for explorations.
│
├── reports            <- Training logs, TensorBoard files, and experiment outputs.
│
├── music_interpret    <- Core Python module for activation processing and
│                         sparse autoencoder (SAE) training.
│
└── environment.yml    <- Conda environment specification for reproducing
                          the development and training setup.
```

## Documentation

For more in depth description of project usage and implementation view project's 
documentation that may be run locally:

```bash
mkdocs serve
```
Then open your browser at:
```
http://127.0.0.1:8000/
```

## Project usage

### 1. Initialize the project

Clone this repository and run:

```bash
make init
```

### 2. Download and insert base model checkpoint and dataset
This project was developed using pretrained checkpoints and preprocessed dataset provided by 
base project authors. The preprocessed datasets can be found [here](https://ucsdcloud-my.sharepoint.com/:f:/g/personal/h3dong_ucsd_edu/Er7nrsVc7NhNtYVSdWpHMQwBS5U1dXo0q0eQEi2LW-DVGw).
The pretrained models can be found [here](https://ucsdcloud-my.sharepoint.com/:f:/g/personal/h3dong_ucsd_edu/EqYq6KHrcltHvgJTmw7Nl6MBtv4szg4RUZUPXc4i_RgEkw).

- For the dataset download one of the '{dataset_name}_notes.zip' 
files into ./data/base and unzip it the same folder. 
- For model checkpoint download model checkpoints while keeping 
folder structure from the link.

An example of target dependency structure:
```
data/
└── base/
    └── lmd_notes/
        └── notes/
            └── ...
models/
└── lmd/
    └── ape/
        ├── checkpoints/
        │   └── best_model.pt
        └── train-args.json
```
### 3. Extract activations

```bash
python3 -m music_interpret.activation_extraction <dataset_name> <data_repr> --layers <layer_index>
```

### 4. Preprocess activations
```bash
python3 -m music_interpret.preprocess <dataset_name> <data_repr> --layers <layer_index>
```

### 5. Train a sparse autoencoder
```bash
python3 -m music_interpret.train <dataset_name> <data_repr> \
--series_name <exps_name> --experiment_name <exp_name> --layers <layer_index>
```
### 6. Inspect features
Run and follow feature_exploration.ipynb for feature analysis
```bash
jupyter lab
```

## Acknowledgements

This project builds directly on the work of:

> Hao-Wen Dong, Ke Chen, Shlomo Dubnov, Julian McAuley, and Taylor Berg-Kirkpatrick, "Multitrack Music Transformer," _IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_, 2023.

[[homepage](https://salu133445.github.io/mmt/)]
[[paper](https://arxiv.org/pdf/2207.06983.pdf)]
[[code](https://github.com/salu133445/mmt)]
[[reviews](https://salu133445.github.io/pdf/mmt-icassp2023-reviews.pdf)]

All credit for the original model architecture, datasets, and training
pipeline belongs to the original authors.

---

## License

See the LICENSE file for details.


