# Music Interpret

Music Interpret is a project meant for extracting, analyzing, and interpreting
internal representations of music transformer models. It is implementation of a 
base **ReLU + L1 sparase autoencoder**.

The project is designed to work with the **Multitrack Music Transformer (MMT)**
by salu133445 and builds on top of the original implementation to provide basic
tools for interpretability and representation analysis.

---

## Purpose of this project
- Understanding what symbolic music model's activation represent internally  
- Extracting token-level activations from transformer layers  
- Discovering latent structure using sparse autoencoders  
- Providing basic tools for inspection, diagnostics, and analysis  

This project does not aim to replace the original training pipeline.
Instead, it assumes that a trained model and preprocessed datasets already
exist and builds an analysis workflow on top of them.

---

## Typical workflow

A standard analysis run follows these steps:

1. Prepare a trained base model using the original MMT workflow or use pretrained checkpoints or datasets.  
2. Extract activations from selected transformer layers.  
3. Preprocess activation shards (normalization and statistics).  
4. Train sparse autoencoders on processed activations.  
5. Inspect learned features using logs, metrics, and notebooks.  

---

## Getting started

### 1. Initialize the project

Clone the repository and run:

```
make init
```

This command:

1. Clones the base MMT repository into the local `mmt/` directory.  
2. Creates the development environment using the existing
   `create_environment` Makefile target.  

---

## Using the base MMT model

Music Interpret depends on trained models and datasets provided by the
original MMT project.

To get started, follow the instructions in:

https://github.com/salu133445/mmt

From there, you can either:

- Train your own model, or  
- Download pretrained checkpoints and preprocessed datasets  

Afterwards, place the assets in the expected locations:

```
data/
└── base/
    └── <dataset_name>_notes/
        └── notes/
            └── ...

models/
└── <dataset_name>/
    └── <model_name>/
        ├── checkpoints/
        │   └── best_model.pt
        └── train-args.json
```

---

## End-to-end usage

### Step 1 - Extract activations

```
python3 -m music_interpret.activation_extraction   <dataset_name> <data_repr>   --layers 0 1 2 3
```

---

### Step 2 - Preprocess activations

```
python3 -m music_interpret.preprocess   <dataset_name> <data_repr>   --layers 0 1 2 3
```

---

### Step 3 - Train sparse autoencoders

```
python3 -m music_interpret.train   <series_name> <experiment_name> <layer_dir>   --input-dim 768   --latent-dim 256
```

---

### Step 4 - Monitor training

```
tensorboard --logdir reports/
```

Open:

```
http://localhost:6006/
```

### Step 5 - Inspect features
Run and follow feature_exploration.ipynb for feature analysis
```
jupyter lab
```

---

## Project organization

```
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with convenience commands like 'init'
├── README.md          <- Top-level project overview
├── mkdocs.yml         <- MkDocs configuration file
│
├── data
│   ├── base           <- Preprocessed note datasets
│   └── activations    <- Extracted activations and metadata
│
├── docs               <- Project documentation
│
├── models             <- Base model checkpoints
│
├── notebooks          <- Jupyter notebooks for explorations
│
├── reports            <- Training logs and experiment outputs
│
├── music_interpret    <- Core Python module for activation processing
│                         and sparse autoencoder training
│
└── environment.yml    <- Conda environment specification
```

---

## Acknowledgements

This project builds directly on the work of:



> Hao-Wen Dong, Ke Chen, Shlomo Dubnov, Julian McAuley, and Taylor Berg-Kirkpatrick,  
> *“Multitrack Music Transformer”*, IEEE ICASSP 2023.

- Project homepage: https://salu133445.github.io/mmt/
- Paper: https://arxiv.org/pdf/2207.06983.pdf
- Source code: https://github.com/salu133445/mmt


All credit for the original model architecture, datasets, and training
pipeline belongs to the original authors.

---

## License

See the LICENSE file for details.
