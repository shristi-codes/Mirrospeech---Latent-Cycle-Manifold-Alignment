# Mirrospeech---Latent-Cycle-Manifold-Alignment

🚀 MirrorSpeech
Latent Cycle Manifold Alignment for Unseen-Accent Robust ASR

Strip the accent. Keep the meaning. Make ASR fair.

📌 Overview

MirrorSpeech is an accent-robust Automatic Speech Recognition (ASR) system built on top of Whisper-small.
It improves transcription accuracy across diverse accents by learning to separate:

Content → what is being said
Accent → how it is being said

The system introduces Latent Cycle Manifold Alignment (LCMA) to enforce this separation.

🎯 Problem

Speech recognition systems often struggle with non-native accents, leading to:

High Word Error Rate (WER)
Bias toward native speakers
Poor real-world usability
💡 Solution

MirrorSpeech addresses this using:

⚡ LoRA (Low-Rank Adaptation) → train <1% parameters
🧠 Content / Accent Disentanglement
🔁 LCMA Cycle Consistency Loss
🧠 Core Idea (LCMA)

Mirror Test:
Remove accent → reconstruct → check if meaning is preserved

Loss Function:

L_LCMA = β * MSE(content, content')

This ensures the model learns accent-invariant content representations.

🏗️ Architecture
Audio → Whisper Encoder → Content Head + Accent Head
                        ↓
               Accent Removal / Swap
                        ↓
                Reconstruct & Align
📊 Results
Metric	Baseline (Whisper)	MirrorSpeech
Overall WER	26.0%	11.5%
Arabic WER	109.2%	~11%
Korean (Rel.)	—	↓ 21%
Training Params	100%	<1%

👉 55.8% relative improvement in WER
👉 Trained in 3 epochs on a single GPU

🔬 Dataset
L2-ARCTIC (Accented English)
LibriSpeech (Native English)
⚙️ Tech Stack
Python
OpenAI Whisper
HuggingFace Transformers
LoRA (PEFT)
Evaluation: WER, CER, PER, WIL, BERTScore
🧪 Code Demo

Notebook:

Phase2_Task8_colab_demo_mirrorspeech_V4.ipynb

Includes:

Audio clips
Ground truth
Baseline predictions
MirrorSpeech predictions
Per-sample WER improvement
📂 Project Structure
.
├── Phase1_Task1_...        # Data loading
├── Phase1_Task2_...        # Preprocessing
├── Phase2_Task5_LCMA...    # Core LCMA
├── Phase2_Task6_...        # Evaluation
├── Phase2_Task8_...        # Demo
├── results/                # Outputs
└── README.md
🚀 Future Scope
Stronger disentanglement using adversarial learning
Generalization to unseen accents & multilingual speech
Accent control and transfer
Real-time streaming ASR deployment
📈 Key Contributions
Accent-aware representation learning
Parameter-efficient fine-tuning (<1%)
Novel LCMA-based disentanglement
Significant improvement on challenging accents
👥 Team
Shreya Akotiya
Vidushi Verma
Mamta Jha
Shristi Kumar
🙌 Acknowledgements
OpenAI Whisper
HuggingFace
L2-ARCTIC Dataset
LibriSpeech Dataset
⭐ Support

If you found this useful, consider giving it a ⭐ on GitHub!
