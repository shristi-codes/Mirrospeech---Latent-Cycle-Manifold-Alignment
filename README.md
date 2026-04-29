# MirrorSpeech  
### Latent Cycle Manifold Alignment for Unseen-Accent Robust ASR  

<p align="center"><b>Strip the accent. Keep the meaning. Make ASR fair.</b></p>

---

## Overview  

MirrorSpeech is an **accent-robust Automatic Speech Recognition (ASR) system** built on top of Whisper-small.

It improves transcription accuracy across diverse accents by learning to separate:

- **Content** — what is being said  
- **Accent** — how it is being said  

The system introduces **Latent Cycle Manifold Alignment (LCMA)** to enforce this separation.

---

## Problem  

Speech recognition systems struggle with non-native accents:

-  High Word Error Rate (WER)  
-  Bias toward native speakers  
-  Poor real-world usability  

---

##  Solution  

MirrorSpeech addresses this using:

- **LoRA (Low-Rank Adaptation)** — train <1% parameters  
- **Content / Accent Disentanglement**  
- **LCMA Cycle Consistency Loss**  

---

## Core Idea  

**Mirror Test:**  
Remove accent → reconstruct → check if meaning is preserved  

**Loss Function:**
L_LCMA = β * MSE(content, content')


## Architecture  
Audio → Whisper Encoder → Content Head + Accent Head
↓
Accent Removal / Swap
↓
Reconstruct & Align


---

## Results  

| Metric            | Baseline | MirrorSpeech |
|------------------|----------|-------------|
| Overall WER      | 26.0%    | **11.5%**   |
| Arabic WER       | 109.2%   | **~11%**    |
| Korean (Rel.)    | —        | ↓ 21%       |
| Trainable Params | 100%     | **<1%**     |

 **55.8% relative improvement in WER**  
 Trained in **3 epochs on a single GPU**

---

## Code Demo  

Run the demo notebook:

## Results  

| Metric            | Baseline | MirrorSpeech |
|------------------|----------|-------------|
| Overall WER      | 26.0%    | **11.5%**   |
| Arabic WER       | 109.2%   | **~11%**    |
| Korean (Rel.)    | —        | ↓ 21%       |
| Trainable Params | 100%     | **<1%**     |

**55.8% relative improvement in WER**  
Trained in **3 epochs on a single GPU**

Run the demo notebook:
Phase2_Task8_colab_demo_mirrorspeech_V4.ipynb


Includes:
- Audio clips  
- Ground truth  
- Baseline vs MirrorSpeech predictions  
- Per-sample WER improvements  

---

##Future Scope  

- Stronger disentanglement (adversarial learning)  
- Generalization to unseen accents & multilingual speech  
- Accent control & transfer  
- Real-time streaming ASR deployment  

---

##  Team  

- Shreya Akotiya  
- Vidushi Verma  
- Mamta Jha  
- Shristi Kumar  

---

## ⭐ Support  

If you like this project, give it a ⭐ on GitHub!
