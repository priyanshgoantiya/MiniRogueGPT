BabyGPT

# MiniRogueGPT 🤖
[Deploy Link](https://miniroguegpt.streamlit.app/)

A lightweight, from-scratch implementation of a Transformer-based language model built with PyTorch, featuring a user-friendly Streamlit web interface for text generation.

## 🌟 Features

- **Custom Transformer Architecture**: Built from scratch using PyTorch with multi-head attention
- **Optimized Training Pipeline**: Efficient batch processing and gradient optimization
- **Interactive Web Interface**: Streamlit-based UI for easy text generation
- **Model Persistence**: Save and load trained models for inference
- **Configurable Generation**: Adjustable token count and prompt-based generation
- **GPU Acceleration**: CUDA support for faster training and inference

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Usage](#usage)
- [Model Configuration](#model-configuration)
- [File Structure](#file-structure)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## 🏗️ Architecture

MiniRogueGPT implements a decoder-only transformer architecture with the following components:

### Core Components
- **Token Embeddings**: Learned representations for each character in vocabulary
- **Positional Embeddings**: Position-aware encodings for sequence understanding
- **Multi-Head Attention**: Parallel attention mechanisms for capturing different aspects of relationships
- **Feed-Forward Networks**: Two-layer MLPs with GELU activation
- **Layer Normalization**: Pre-norm architecture for stable training
- **Dropout Regularization**: Prevents overfitting during training

### Model Specifications
- **Context Window**: 256 tokens
- **Embedding Dimension**: 384
- **Attention Heads**: 6
- **Transformer Layers**: 6
- **Vocabulary**: Character-level tokenization
- **Parameters**: ~2.3M (23 lakh) trainable parameters

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install streamlit
pip install numpy
```

### Clone Repository
```bash
git clone https://github.com/priyanshgoantiya/MiniRogueGPT.git
cd MiniRogueGPT
```

## 📁 Dataset Setup

1. **Prepare Training Data**: Place your text file as `input.txt` in the project root
   ```
   MiniRogueGPT/
   ├── input.txt          # Your training text data
   ├── main.py            # Training script
   ├── app.py             # Streamlit interface
   └── README.md
   ```

2. **Data Format**: The model expects plain text files with UTF-8 encoding
   - Any text corpus works (books, articles, code, etc.)
   - Larger datasets generally produce better results
   - Recommended minimum: 1MB of text data

## 🎯 Training

### Quick Start
```bash
python main.py
```

### Training Process
The training script will:
1. **Load and preprocess** your text data
2. **Create character-level vocabulary** mapping
3. **Split data** into training (90%) and validation (10%) sets
4. **Train the model** for 5000 iterations with evaluation every 500 steps
5. **Save model checkpoints** every 500 iterations
6. **Generate sample text** upon completion

### Training Output
```
step 0: train loss 4.2817, test loss 4.2796
step 500: train loss 1.9665, test loss 2.0664
step 1000: train loss 1.5992, test loss 1.7774
step 1500: train loss 1.4365, test loss 1.6367
step 2000: train loss 1.3425, test loss 1.5668
step 2500: train loss 1.2754, test loss 1.5156
step 3000: train loss 1.2227, test loss 1.4931
step 3500: train loss 1.1815, test loss 1.4806
step 4000: train loss 1.1432, test loss 1.4715
step 4500: train loss 1.1079, test loss 1.4686
step 4999: train loss 1.0728, test loss 1.4791
```

## 💻 Usage

### Streamlit Web Interface
Launch the interactive web application:
```bash
streamlit run app.py
```

**Features:**
- **Prompt Input**: Enter starting text for generation
- **Token Control**: Adjust the number of tokens to generate (50-1000)
- **Real-time Generation**: Click "Generate" for instant results
- **Copy-friendly Output**: Generated text in scrollable text area

### Programmatic Usage
```python
import torch
from your_model import BigramLanguageModel

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BigramLanguageModel().to(device)
model.load_state_dict(torch.load("language_model_step4999.pth"))
model.eval()

# Generate text
prompt = "Once upon a time"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=200)
output = decode(generated[0].tolist())
print(output)
```

## ⚙️ Model Configuration

### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 64 | Number of samples per training batch |
| `block_size` | 256 | Context window size |
| `max_iters` | 5000 | Total number of training iterations |
| `eval_interval` | 500 | Interval for evaluating loss |
| `eval_iter` | 200 | Number of iterations to compute average loss |
| `lr` | 3e-4 | Learning rate for AdamW optimizer |
| `num_embd` | 384 | Embedding dimension |
| `dropout` | 0.2 | Dropout rate to prevent overfitting |
| `n_head` | 6 | Number of attention heads |
| `n_layer` | 6 | Number of transformer layers |

### Customization
Modify hyperparameters in `main.py` to experiment with different configurations:
```python
# Larger model for better quality
num_embd = 512
n_head = 8
n_layer = 8

# Longer context for better coherence
block_size = 512

# More training for convergence
max_iters = 10000
```

## 📂 File Structure

```
MiniRogueGPT/
├── main.py                 # Training script with full model implementation
├── app.py                  # Streamlit web interface
├── input.txt              # Training data (user-provided)
├── language_model_step4999.pth  # Final trained model weights
├── language_model_stepXXX.pth   # Training checkpoints (500, 1000, 1500...)
├── generated_text.txt     # Sample output from training
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## 📊 Performance

### Training Metrics
- **Training Time**: ~20-40 minutes (depending on GPU and dataset size)
- **Total Iterations**: 5000 (with evaluation every 500 steps)
- **Batch Size**: 64 samples per batch
- **Memory Usage**: ~4-6GB GPU memory
- **Loss Convergence**: Typically achieves <1.1 train loss, <1.5 validation loss
- **Generation Speed**: ~200-800 tokens/second

### Quality Indicators
- **Coherent Long Phrases**: Model learns complex grammar and vocabulary patterns
- **Context Awareness**: Maintains consistency within 256-token context window
- **Style Mimicry**: Closely adapts to training data's writing style
- **Character-Level Accuracy**: Produces valid characters and coherent words

## 🛠️ Advanced Usage

### Custom Training Data
```python
# For specialized domains
# Poetry: Use poetry collections
# Code: Use programming repositories  
# Dialogue: Use script/conversation data
# Technical: Use documentation/papers
```

### Model Scaling
```python
# Small model (faster training)
num_embd, n_head, n_layer = 256, 4, 4

# Standard model (current)
num_embd, n_head, n_layer = 384, 6, 6

# Large model (better quality)
num_embd, n_head, n_layer = 512, 8, 8
```

### Generation Strategies
```python
# Temperature sampling (add to model)
def generate_with_temperature(self, idx, max_tokens, temperature=1.0):
    for _ in range(max_tokens):
        logits, _ = self(idx[:, -block_size:])
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```python
# Reduce batch size
batch_size = 32  # or 16

# Reduce model size
num_embd = 256
n_layer = 4
```

**Poor Generation Quality**
- Increase training iterations (`max_iters = 8000`)
- Use larger/better training data
- Reduce dropout during inference
- Try different learning rates

**Slow Training**
- Enable GPU acceleration
- Reduce model size for experimentation
- Use smaller datasets for testing

### Model Not Loading
```python
# Check device compatibility
model.load_state_dict(torch.load("language_model_step4999.pth", map_location="cpu"))
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Ideas
- Add temperature/top-k/top-p sampling
- Implement beam search decoding
- Add model visualization tools
- Create training progress monitoring
- Implement different tokenization schemes
- Add model comparison utilities

## 📈 Roadmap

- [ ] **Advanced Sampling**: Temperature, top-k, top-p sampling
- [ ] **Larger Context**: Extend context window to 512+ tokens
- [ ] **Better Tokenization**: Subword tokenization (BPE/SentencePiece)
- [ ] **Model Variants**: Experiment with different architectures
- [ ] **Training Improvements**: Learning rate scheduling, gradient clipping
- [ ] **Evaluation Metrics**: Perplexity, BLEU scores
- [ ] **Web Deployment**: Deploy to cloud platforms

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Attention Is All You Need** paper by Vaswani et al.
- **GPT** series by OpenAI for architectural inspiration  
- **PyTorch** team for the amazing deep learning framework
- **Streamlit** for the intuitive web framework
- **Andrej Karpathy** for educational content on transformers

## 📞 Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Join conversations in GitHub Discussions  
- **Contact**: Reach out via GitHub profile

---

**Made with ❤️ by [priyanshgoantiya](https://github.com/priyanshgoantiya)**

*Star ⭐ this repo if you found it helpful!*
