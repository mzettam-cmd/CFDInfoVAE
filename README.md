# Learning Nonlinear Reduced-Order Models of Fluid Flows with InfoVAE

## Introduction

This repository provides a Python implementation of a reduced-order modelling (ROM) framework for fluid flows based on:

- **Proper Orthogonal Decomposition (POD)** as a classical linear ROM baseline.
- **Information Maximizing Variational Autoencoder (InfoVAE)** for nonlinear dimensionality reduction.
- **Easy-Attention Transformer** for temporal prediction in the InfoVAE latent space.

The framework is designed to learn a low-dimensional representation of CFD flow fields, model the temporal evolution of the latent variables, and reconstruct the flow field from the predicted latent representation.

The workflow implemented in the code is:

```text
                         CFD snapshots
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
               POD                       InfoVAE
          (ROM baseline)                Encoder
                                            │
                                            ▼
                                      Latent space
                                            │
                                            ▼
                                  latent_data.h5py
                                            │
                                            ▼
                              Easy-Attention Transformer
                                            │
                                            ▼
                                  Predicted latent space
                                            │
                                            ▼
                                      InfoVAE Decoder
                                            │
                                            ▼
                                  Reconstructed flow
```

POD is used as an independent reference method, while the InfoVAE and Easy-Attention Transformer form the nonlinear ROM and temporal prediction pipeline.

More details about the methodology, numerical experiments, and results are available in:

> **Reduced-order modelling of fluid flows with the Information Maximizing Variational Autoencoder (InfoVAE)**

Research Square preprint:

https://doi.org/10.21203/rs.3.rs-10594643/v1

---

## Main Features

The repository includes:

- Proper Orthogonal Decomposition (POD).
- Information Maximizing Variational Autoencoder (InfoVAE).
- Convolutional encoder-decoder architecture for CFD flow fields.
- Latent-space extraction from the InfoVAE encoder.
- Automatic generation of the latent-space dataset.
- Easy-Attention Transformer for temporal prediction.
- Training and testing procedures.
- Spatial post-processing.
- Temporal post-processing.
- Reconstruction and energy analysis.
- Visualization utilities.

---

## Data

The main CFD dataset used in the experiments is:

```text
data/Data2PlatesGap1Re40_Alpha-00_downsampled_v6.hdf5
```

The dataset contains the CFD snapshots used to train and evaluate the InfoVAE model.

The dataset is managed using **Git Large File Storage (Git LFS)**.

After cloning the repository, install Git LFS and retrieve the dataset:

```bash
git lfs install
git lfs pull
```

### Latent-space dataset

The file

```text
data/latent_data.h5py
```

is **generated automatically by the InfoVAE post-processing stage**.

It is not an independent CFD dataset.

The file contains:

```text
vector
vector_test
```

where:

- `vector` contains the latent vectors obtained from the InfoVAE encoder for the training data.
- `vector_test` contains the latent vectors obtained from the InfoVAE encoder for the test data.

The latent vectors are generated from the encoder output:

```python
out = model.encoder(batch)
mean, logvariance = torch.chunk(out, 2, dim=1)
```

The mean of the latent distribution is used as the latent representation.

The file is then created using:

```python
with h5py.File(pathsBib.data_path + 'latent_data.h5py', 'w') as f:
    f.create_dataset('vector', data=means_train)
    f.create_dataset('vector_test', data=means_test)
```

Therefore, `latent_data.h5py` is reproduced by running the InfoVAE training/inference and post-processing workflow.

---

## Installation

The code is implemented in Python.

A dedicated Python environment is recommended.

### Create a virtual environment

Using `venv`:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### Install dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- Python
- PyTorch
- NumPy
- SciPy
- h5py
- Matplotlib
- scikit-learn

The exact package versions may depend on the computational environment.

---

## Running the Code

The main entry point is:

```text
python main.py -re 40 -m run -nn easy
```

The experiments are controlled through configuration files located in:

```text
configs/
```

The InfoVAE configuration is defined in:

```text
configs/infovae.py
```

The temporal prediction configuration is defined according to the selected temporal model, including the Easy-Attention Transformer configuration.

The main execution structure is:

```python
if __name__ == "__main__":

    datafile = init.init_env(args.re)

    infoVAE = infoVAERunner(device, datafile)

    if args.m == 'train':
        infoVAE.train()

    elif args.m == 'test':
        infoVAE.infer(args.t)

    elif args.m == 'run':
        infoVAE.run()

    latentRunner = latentRunner(args.nn, device)

    if args.m == 'train':
        latentRunner.train()

    elif args.m == 'test':
        latentRunner.infer(args.t)

    elif args.m == 'run':
        latentRunner.train()
        latentRunner.infer(args.t)
```

The exact command-line arguments depend on the configuration implemented in `main.py`.

---

# InfoVAE

## Architecture

The InfoVAE architecture contains a convolutional encoder and a convolutional decoder.

The implementation is located in:

```text
nns/info_vae.py
```

The encoder progressively reduces the spatial dimensions of the CFD flow field using convolutional layers and finally maps the representation to the latent space.

The encoder produces:

```text
mean
logvariance
```

which are used by the reparameterization trick to obtain the latent variable:

```text
z = mean + epsilon * exp(0.5 * logvariance)
```

The decoder maps the latent vector back to the physical flow field using fully connected and transposed convolutional layers.

The InfoVAE architecture can therefore be summarized as:

```text
CFD flow field
      │
      ▼
Convolutional Encoder
      │
      ▼
mean + logvariance
      │
      ▼
Reparameterization
      │
      ▼
Latent vector z
      │
      ▼
Convolutional Decoder
      │
      ▼
Reconstructed flow field
```

---

## InfoVAE Loss

The training procedure is implemented in:

```text
lib/train.py
```

The InfoVAE training objective combines reconstruction error, Kullback–Leibler divergence, and Maximum Mean Discrepancy (MMD).

The implementation uses the parameters:

```text
alpha
lambda
```

to control the contribution of the regularization terms.

The training procedure monitors:

- Reconstruction loss (MSE).
- Kullback–Leibler divergence (KLD).
- Maximum Mean Discrepancy (MMD).
- Total loss.
- Validation loss.
- Latent-space collapse.

---

# Latent-Space Generation

After training the InfoVAE, the encoder is used to extract the latent representation of the CFD snapshots.

The encoding procedure is implemented in the spatial post-processing code.

The encoder is called using:

```python
out = model.encoder(batch)
mean, logvariance = torch.chunk(out, 2, dim=1)
```

The resulting latent means are stored in:

```text
data/latent_data.h5py
```

with the datasets:

```text
vector
vector_test
```

This file is subsequently used by the temporal prediction model.

---

# Easy-Attention Transformer

The Easy-Attention Transformer is used for **temporal modelling in the InfoVAE latent space**.

It is therefore not part of the InfoVAE encoder-decoder architecture itself.

The temporal prediction model is initialized by:

```python
self.model, self.filename, self.config = get_predictors(name)
```

in:

```text
lib/runners.py
```

The temporal runner loads the latent data using:

```python
hdf5 = h5py.File(pathsBib.data_path + "latent_data.h5py")
data = np.array(hdf5['vector'])
```

The latent vectors are then converted into temporal input/output sequences using:

```python
X, Y = make_Sequence(self.config, data=data)
```

The temporal model is trained on these latent-space sequences.

The general process is:

```text
InfoVAE Encoder
      │
      ▼
Latent vectors
      │
      ▼
Temporal sequences
      │
      ▼
Easy-Attention Transformer
      │
      ▼
Predicted latent vectors
```

The Transformer therefore learns the temporal dynamics without directly operating on the high-dimensional CFD flow field.

---

# Reconstruction Using the Decoder

Once the temporal model predicts a latent vector, the InfoVAE decoder can map the latent vector back to the physical flow space.

The decoder is called using:

```python
rec = model.decoder(batch)
```

The decoding procedure is implemented in the spatial post-processing code.

Therefore, the complete nonlinear ROM workflow is:

```text
CFD snapshots
      │
      ▼
InfoVAE Encoder
      │
      ▼
Latent representation
      │
      ▼
Easy-Attention Transformer
      │
      ▼
Predicted latent representation
      │
      ▼
InfoVAE Decoder
      │
      ▼
Reconstructed CFD flow
```

---

# Proper Orthogonal Decomposition (POD)

POD is implemented as a classical linear reduced-order modelling method and is used as a reference for evaluating the nonlinear InfoVAE representation.

The implementation is located in:

```text
lib/POD.py
```

POD is an independent ROM approach and is not used as a mandatory preprocessing step for InfoVAE.

The comparison can therefore be represented as:

```text
                    CFD snapshots
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
            POD                     InfoVAE
       Linear ROM                Nonlinear ROM
                                      │
                                      ▼
                              Latent representation
                                      │
                                      ▼
                         Easy-Attention Transformer
                                      │
                                      ▼
                              Latent prediction
                                      │
                                      ▼
                                InfoVAE Decoder
                                      │
                                      ▼
                             Reconstructed flow
```

---

# Spatial Post-Processing

The spatial post-processing and latent-space analysis are implemented in:

```text
lib/pp_space.py
```

The main function is:

```python
spatial_Mode(...)
```

The post-processing includes:

- Encoding training and test data.
- Generation of nonlinear spatial modes.
- Ranking of latent modes.
- Cumulative energy analysis.
- Reconstruction error analysis.
- Reconstructed kinetic energy.
- Generation of latent-space datasets.
- Saving of post-processing results.

The generated results include:

```text
mean
std
vector
vector_test
stds_vector
stds_vector_test
modes
zero_output
NLvalues
NLmodes
order
Ecum
Ecum_test
Ek_t
```

---

# Temporal Post-Processing

Temporal prediction and evaluation are handled by the temporal runner in:

```text
lib/runners.py
```

The temporal model can be evaluated using:

- Prediction error.
- Sliding-window error.
- Poincaré maps.
- Comparison between predicted and reference latent trajectories.

The results are stored in compressed NumPy files:

```text
.npz
```

---

# Repository Structure

```text
CFDInfoVAE/
│
├── configs/
│   ├── infovae.py          # InfoVAE configuration
│   ├── easyAttn.py         # Easy-Attention configuration
│   ├── lstm.py             # LSTM configuration
│   ├── selfAttn.py         # Self-Attention configuration
│   └── nomenclature.py     # Experiment naming
│
├── data/
│   ├── Data2PlatesGap1Re40_Alpha-00_downsampled_v6.hdf5
│   └── latent_data.h5py    # Generated latent-space dataset
│
├── lib/
│   ├── POD.py              # Proper Orthogonal Decomposition
│   ├── datas.py            # Data loading and processing
│   ├── train.py            # Training and testing procedures
│   ├── runners.py          # InfoVAE and temporal runners
│   ├── model.py            # Model construction utilities
│   ├── pp_space.py         # Spatial post-processing
│   └── pp_time.py          # Temporal post-processing
│
├── nns/
│   ├── info_vae.py         # InfoVAE architecture
│   ├── transformer.py      # Transformer architecture
│   ├── attns.py            # Attention mechanisms
│   ├── embedding.py        # Embedding layers
│   ├── layers.py           # Neural-network layers
│   └── RNNs.py             # Recurrent neural networks
│
├── utils/
│   ├── figs.py             # Visualization utilities
│   ├── figs_time.py        # Temporal visualization
│   ├── io.py               # Input/output utilities
│   └── plt_rc_setup.py     # Plot configuration
│
├── main.py                 # Main entry point
├── README.md               # Project documentation
├── LICENSE                 # Project license
└── .gitattributes          # Git LFS configuration
```

---

# Reproducibility

The repository is intended to facilitate the reproduction of the numerical experiments presented in the associated research work.

The main parameters controlling the experiments include:

- Reynolds number.
- Latent-space dimension.
- Batch size.
- Learning rate.
- Number of training epochs.
- InfoVAE `alpha` parameter.
- InfoVAE `lambda` parameter.
- Weight decay.
- Temporal model architecture.
- Temporal sequence length.
- Prediction horizon.

The corresponding parameters can be modified in:

```text
configs/
```

The latent-space dataset required by the temporal prediction stage is generated from the trained InfoVAE model through the post-processing pipeline.

---

# Citation

If you use this code in your research, please cite:

**Reduced-order modelling of fluid flows with the Information Maximizing Variational Autoencoder (InfoVAE)**

```bibtex
@article{zettam2026infovae,
  title={Reduced-order modelling of fluid flows with the Information Maximizing Variational Autoencoder (InfoVAE)},
  author={Zettam, Manal},
  year={2026},
  doi={10.21203/rs.3.rs-10594643/v1}
}
```

Research Square:

https://doi.org/10.21203/rs.3.rs-10594643/v1

---

# License

This project is released under the **MIT License**.

See the `LICENSE` file for details.

---

# Acknowledgements

This repository contains the computational implementation associated with the research work on reduced-order modelling of fluid flows using InfoVAE, POD, and latent-space temporal prediction.

For questions, suggestions, or discussions regarding the implementation, please open an issue in the repository.