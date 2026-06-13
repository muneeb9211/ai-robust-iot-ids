# AI-Powered IoT Intrusion Detection System with Adversarial Robustness Testing

**Author:** Muneeb Ahmad — CTO at [AllysAI](https://allysai.com), MS Computer Science, PLOS ONE (Q1) published researcher on IoT/RPL security and distributed systems.

## Motivation

IoT networks are increasingly targeted by sophisticated cyberattacks. Machine learning-based Intrusion Detection Systems (IDS) have emerged as the state-of-the-art defense, achieving high accuracy on benchmark datasets. However, **adversarial machine learning** exposes a critical vulnerability: carefully crafted input perturbations can evade detection with high success rates while remaining imperceptible.

This project systematically studies the **robustness gap** between clean-data accuracy and adversarial-data accuracy across multiple attack strategies and defense mechanisms. The goal is to quantify how much ML-based IDS performance degrades under adversarial conditions and evaluate which defenses provide meaningful robustness improvements.

## Architecture

```
                           +------------------+
                           |   NSL-KDD Data   |
                           | (or Synthetic)   |
                           +--------+---------+
                                    |
                           +--------v---------+
                           |  Preprocessing   |
                           |  - One-hot encode|
                           |  - Standardize   |
                           +--------+---------+
                                    |
                    +---------------+---------------+
                    |               |               |
            +-------v------+ +-----v------+ +------v-------+
            | Random Forest| |  Deep NN   | | Autoencoder  |
            | (sklearn)    | | (PyTorch)  | | (PyTorch)    |
            | Baseline     | | 4-layer    | | Anomaly Det. |
            +--------------+ +-----+------+ +--------------+
                                    |
                    +---------------+---------------+
                    |               |               |
             +------v-----+ +------v-----+ +-------v------+
             |   FGSM     | |    PGD     | |   C&W L2     |
             | single-step| | iterative  | | optimization |
             +------+-----+ +------+-----+ +-------+------+
                    |               |               |
                    +-------+-------+-------+-------+
                            |               |
                    +-------v------+ +------v--------+
                    | Adv.Training | | Input Transf. |
                    | (PGD-AT)     | | Squeezing /   |
                    +--------------+ | Smoothing     |
                                     +---------------+
                            |
                    +-------v--------+
                    |   Ensemble     |
                    | RF + DNN + AE  |
                    | Majority Vote  |
                    +-------+--------+
                            |
                    +-------v--------+
                    |   Evaluation   |
                    | Acc, F1, DR,   |
                    | Robustness     |
                    +----------------+
```

## Methodology

The evaluation follows a four-phase cycle:

1. **Train** baseline models on clean NSL-KDD data
2. **Attack** the DNN with FGSM, PGD, and C&W to measure vulnerability
3. **Defend** using adversarial training, input transformations, and ensembling
4. **Evaluate** defended models under the same attacks to measure robustness gain

## Dataset: NSL-KDD

NSL-KDD is a refined version of the KDD Cup 1999 dataset, addressing duplicate records and unrealistic class distributions. It contains 41 features describing network connections:

| Feature Group | Examples | Count |
|---|---|---|
| Basic | duration, protocol_type, service, flag, src_bytes, dst_bytes | 9 |
| Content | hot, num_failed_logins, logged_in, num_compromised, root_shell | 13 |
| Traffic | count, srv_count, serror_rate, same_srv_rate | 9 |
| Host | dst_host_count, dst_host_srv_count, dst_host_same_srv_rate | 10 |

Labels are mapped to binary: **normal** (0) vs. **attack** (1). Attack categories include DoS, Probe, R2L, and U2R.

A synthetic data generator is included as fallback if NSL-KDD download fails, producing feature-compatible samples with realistic distributions.

## Adversarial Attacks

### FGSM (Fast Gradient Sign Method)
**Goodfellow et al., 2015** — Single-step perturbation along the gradient direction:

```
x_adv = x + epsilon * sign(nabla_x L(theta, x, y))
```

Fast to compute but produces suboptimal adversarial examples. Serves as a lower bound on adversarial vulnerability.

### PGD (Projected Gradient Descent)
**Madry et al., 2018** — Iterative refinement of FGSM with projection:

```
x^{t+1} = Pi_{B(x, epsilon)} ( x^t + alpha * sign(nabla_x L(theta, x^t, y)) )
```

PGD is the canonical first-order adversary. The projection operator `Pi` ensures perturbations stay within the L-infinity epsilon-ball. Considered the strongest first-order attack.

### Carlini & Wagner L2
**Carlini and Wagner, 2017** — Optimization-based attack minimizing:

```
min_delta  ||delta||_2  +  c * max( Z(x+delta)_y - max_{i != y} Z(x+delta)_i,  -kappa )
```

Significantly stronger than gradient-sign methods. Finds minimal perturbations that cause misclassification, making it the gold standard for robustness evaluation.

## Defenses

### Adversarial Training (PGD-AT)
Augments each training mini-batch with PGD adversarial examples, forcing the model to learn robust representations. This is the most principled defense with theoretical backing from robust optimization.

### Input Transformations
- **Feature Squeezing** (Xu et al., 2018): Reduces feature precision via quantization, destroying small adversarial perturbations.
- **Spatial Smoothing**: Applies local averaging over feature windows to disrupt gradient-aligned perturbations.

### Ensemble Defense
Combines predictions from Random Forest (non-differentiable), DNN (differentiable), and Autoencoder (anomaly-based) via majority voting. Adversarial examples crafted against the DNN are unlikely to simultaneously fool the RF and autoencoder.

## Results

Results are generated by running the full pipeline. Example output format:

| Model | Context | Accuracy | Precision | Recall | F1 | Detection Rate |
|---|---|---|---|---|---|---|
| RandomForest | clean | — | — | — | — | — |
| DNN | clean | — | — | — | — | — |
| DNN | FGSM eps=0.1 | — | — | — | — | — |
| DNN | PGD eps=0.1 | — | — | — | — | — |
| DNN | C&W L2 | — | — | — | — | — |
| DNN-AT | FGSM eps=0.1 | — | — | — | — | — |
| DNN-AT | PGD eps=0.1 | — | — | — | — | — |
| Ensemble | FGSM eps=0.1 | — | — | — | — | — |

Run the pipeline to populate these values: `python -m experiments.full_pipeline --synthetic`

## How to Run

### Setup

```bash
cd iot-adversarial-ids
pip install -r requirements.txt
```

### Quick Start (Synthetic Data)

```bash
# Full pipeline with synthetic data (no download needed)
python -m experiments.full_pipeline --synthetic
```

### With NSL-KDD Data

```bash
# Full pipeline (auto-downloads NSL-KDD if not present)
python -m experiments.full_pipeline
```

### Step by Step

```bash
# 1. Train baseline models
python -m experiments.train_baseline --synthetic

# 2. Run adversarial attacks
python -m experiments.run_attacks --synthetic

# 3. Apply defenses and evaluate
python -m experiments.run_defenses --synthetic
```

### Configuration

Edit `configs/default.yaml` to adjust:
- Model hyperparameters (hidden layers, dropout, learning rate)
- Attack parameters (epsilon values, PGD steps, C&W iterations)
- Defense settings (adversarial training epochs, squeezing bit depth)

## Project Structure

```
iot-adversarial-ids/
├── src/
│   ├── data/           # Dataset loading, preprocessing, synthetic fallback
│   ├── models/         # RF baseline, DNN (PyTorch), Autoencoder (PyTorch)
│   ├── attacks/        # FGSM, PGD, Carlini & Wagner L2
│   ├── defenses/       # Adversarial training, input transforms, ensemble
│   └── evaluation/     # Metrics, comparison tables, result export
├── experiments/        # Runnable scripts (train, attack, defend, full pipeline)
├── configs/            # YAML hyperparameter configs
├── results/            # Generated results (JSON, CSV, tables)
├── requirements.txt
├── setup.py
└── README.md
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0 (CPU, CUDA, or MPS)
- scikit-learn >= 1.3
- pandas, numpy, pyyaml, matplotlib

## References

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). *Explaining and Harnessing Adversarial Examples.* ICLR 2015.
2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018.
3. Carlini, N., & Wagner, D. (2017). *Towards Evaluating the Robustness of Neural Networks.* IEEE S&P 2017.
4. Xu, W., Evans, D., & Qi, Y. (2018). *Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks.* NDSS 2018.
5. Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). *A Detailed Analysis of the KDD CUP 99 Data Set.* IEEE CISDA 2009.

## Author

**Muneeb Ahmad**
- CTO at AllysAI
- Published: PLOS ONE (Q1) — IoT/RPL security in distributed networks
- Research focus: AI-powered security for IoT, adversarial machine learning, formal verification, post-quantum cryptography
