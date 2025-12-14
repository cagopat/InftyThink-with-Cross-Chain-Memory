# InftyThink with Cross-Chain Memory

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](./COMP545_Report.pdf)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Directional Attractors in LLM Reasoning: How Similarity Retrieval Steers Iterative Summarization Based Reasoning**

*Charbel Barakat, Luis Joseph Luna Limgenco, Cagatay Tekin*  
*McGill University - COMP 545: Natural Language Understanding with Deep Learning*

---

## Overview

This repository implements **InftyThink with Cross-Chain Memory**, an extension of the InftyThink framework that augments iterative reasoning with an embedding-based semantic cache. Our system stores previously successful reasoning patterns ("lemmas") and retrieves the most semantically similar ones to guide future inference, enabling self-improving reasoning without indiscriminate context window expansion.

### Key Contributions

- **System Design**: Integration of BGE-based semantic cache with InftyThink's iterative summarization framework
- **Empirical Analysis**: Comprehensive evaluation across MATH500, AIME2024, and GPQA-Diamond benchmarks
- **Geometric Analysis**: Discovery of directional attractors in embedding space that predict fix/break outcomes
- **Transfer Learning**: Demonstration of cross-dataset lemma transfer from MATH500 to AIME2024

### Main Findings

- **Domain-Dependent Performance**: 3.0% improvement on MATH500, 10.4% on AIME2024, but degradation on heterogeneous GPQA-Diamond
- **Directional Attractors**: Fix and break outcomes correspond to geometrically separable trajectories in embedding space (98.8% cosine similarity yet statistically distinct)
- **Early Predictability**: Final outcomes can be predicted from initial tokens with up to 82% accuracy
- **Cache Size Effects**: Optimal cache size varies by domain (k=10 for AIME2024, k=5 for GPQA-Diamond)

---

## Architecture

![InftyThink with Cross-Chain Memory Architecture](https://github.com/user-attachments/assets/edf40732-e66e-406d-a8b0-3ed3d70ca9bc)




The system operates through a five-step pipeline:
1. Initialize reasoning from input question
2. Retrieve top-k most similar lemmas via BGE embeddings and cosine similarity
3. Inject lemma-induced bias into context window
4. Generate reasoning trajectory that diverges toward fix or break attractor
5. Produce final answer aligned with corresponding prototype

---

## Installation

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook (optional, for experimentation)
```

### Setup

```bash
# Clone the repository
git clone https://github.com/cagopat/InftyThink-with-Cross-Chain-Memory.git
cd InftyThink-with-Cross-Chain-Memory

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries

```python
openai              # API calls to LLMs
transformers        # HuggingFace models
datasets            # Benchmark datasets
torch               # PyTorch backend
numpy               # Numerical operations
pandas              # Data manipulation
scikit-learn        # Similarity metrics
pydantic            # Output validation
matplotlib          # Visualization
seaborn             # Statistical plots
```

---

## Datasets

The experiments use three benchmark datasets:

### MATH500
- 500 high school math competition problems
- Structured domain with consistent solution strategies
- 92.4% retention after format validation (n=462)

### AIME2024
- 30 problems from 2024 American Invitational Mathematics Examinations
- Highly challenging, invitation-only competition level
- 96.7% retention (n=29)
- Transfer learning applied using MATH500 cache

### GPQA-Diamond
- 198 graduate-level multiple-choice questions
- Covers Biology, Physics, and Chemistry
- Heterogeneous domain structure
- 81.8% retention (n=162)

Datasets are automatically downloaded via HuggingFace `datasets` library.

---

## Experimental Results

### Accuracy Comparison

| Dataset    | n   | Vanilla | k=5   | k=10  | k=15  | Best Δ |
|-----------|-----|---------|-------|-------|-------|--------|
| MATH500   | 462 | 77.3%   | 80.3% | 77.7% | 80.3% | +3.0%  |
| AIME2024  | 29  | 10.3%   | 13.8% | 20.7% | 13.8% | +10.4% |
| GPQA      | 162 | 37.0%   | 38.9% | 37.0% | 34.6% | +1.9%  |

### Key Insights

1. **MATH500**: U-shaped performance curve; resilient to cache size variation
2. **AIME2024**: Optimal at k=10; transfer learning validated
3. **GPQA-Diamond**: Performance degradation beyond k=5; context pollution in heterogeneous domains

### Computational Cost

Cache retrieval increases average steps per question:
- MATH500: +0.26 steps (14% increase, p=0.04)
- GPQA: +0.36 steps (16% increase, p=0.02)

---

## Geometric Analysis Framework

Our analysis reveals that correctness is encoded in subtle directional shifts rather than semantic content:

### Fix and Break Prototypes
- Constructed via mean-pooling of reasoning chains
- 98.8% cosine similarity yet statistically separable
- Alignment predicts outcomes with up to 82% accuracy

### Early Divergence Analysis
- Summary chain alignment to fix prototype → 75.6% fix rate
- Alignment to break prototype → 82% break rate
- Outcome predictable from initial tokens

### Manifold Analysis
- Semantic overlap between fix/break chains: 58% purity
- Cache effects manifest as small geometric perturbations
- Distinct from clear semantic category shifts

---

## Limitations

- Single LLM (Qwen-2.5-32B) and embedding model (BGE-small) tested
- Small AIME2024 sample size (n=29) introduces variance
- Inference order affects cache maturation (not controlled)
- Non-deterministic generation requires multiple runs for robustness

---

## Future Work

1. **Smart Retrieval**: Develop filtering mechanisms to avoid break attractors
2. **Alternative Similarity Metrics**: Explore beyond cosine similarity for heterogeneous domains
3. **Early Prediction**: Leverage initial token alignment for cheaper training methods
4. **Controlled Cache Maturation**: Study impact of inference ordering
5. **Multi-Model Validation**: Test generalization across architectures

---

## Acknowledgements

This study was carried out as part of **COMP 545: Natural Language Understanding with Deep Learning** at McGill University. We thank:
- **Professor Siva Reddy** for initial research direction insights
- **Megh Thakkar** for continued guidance and supervision throughout the project

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaboration inquiries:
- Charbel Barakat: charbel.barakat@mail.mcgill.ca
- Luis Joseph Luna Limgenco: luis.limgenco@mail.mcgill.ca
- Cagatay Tekin: cagatay.tekin@mail.mcgill.ca

---

## Related Work

This work builds upon:
- **InftyThink** (Yan et al., 2025): Iterative summarization-based reasoning
- **Self-Refine** (Madaan et al., 2023): Iterative refinement with feedback
- **Analogical Reasoning** (Yasunaga et al., 2024): Strategy reuse in LLMs
- **Geometric Analysis** (Marks & Tegmark, 2024): Truth directions in embedding space
