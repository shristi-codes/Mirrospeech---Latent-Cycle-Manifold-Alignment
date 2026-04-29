import os
import gc
import re
import json
import math
import random
import zipfile
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from jiwer import wer, cer, wil
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from bert_score import score as bertscore_score

from torch.cuda.amp import GradScaler, autocast

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# Config
# =========================================================
SEED = 42

PER_ACCENT_TRAIN = 2000        # 2k each accent => 10k total
VAL_LIMIT = 200
TEST_LIMIT = None

BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 3
PATIENCE = 2
LR = 2e-5
LOG_EVERY = 50
MAX_VAL_WER_BATCHES = None     # full validation
SAVE_EVERY_N_BATCHES = 100

RUN_NAME = "task7_balanced10k_3ep_lr2e5_alpha05"

RESUME_PATH = None

TASK4_BASELINE = {
    "overall": {
        "wer": 0.2600,
        "cer": 0.1824,
        "ser": 0.5639,
        "per": 0.1849,
        "wil": 0.3186,
        "bertscore_f1": 0.9622
    }
}

ACCENT_LOSS_ALPHA = 0.5
LCMA_BETA = 0.5

# =========================================================
# Reproducibility
# =========================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("MirrorSpeech Balanced 10k Improvement Run")
print("=" * 70)
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("WARNING: CUDA not available. Training will be very slow.")
print("=" * 70)

# =========================================================
# Paths
# =========================================================
DRIVE_ROOT = r"C:\Users\shreya\Desktop\Mirrorspeech"

DRIVE_ZIP_DIR = rf"{DRIVE_ROOT}\l2arctic_zips"
SPLITS_DIR = rf"{DRIVE_ROOT}\splits"
LIBRI_ROOT = rf"{DRIVE_ROOT}\librispeech"
CKPT_DIR = rf"{DRIVE_ROOT}\checkpoints_balanced10k_27Aprrun"

EXTRACT_PATH = rf"{DRIVE_ROOT}\data\l2arctic"
LIBRI_LOCAL = rf"{DRIVE_ROOT}\data\librispeech"

os.makedirs(EXTRACT_PATH, exist_ok=True)
os.makedirs(LIBRI_LOCAL, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

print("SPLITS_DIR :", SPLITS_DIR)
print("LIBRI_ROOT :", LIBRI_ROOT)
print("CKPT_DIR   :", CKPT_DIR)

assert os.path.exists(f"{SPLITS_DIR}/config.json"), f"Missing {SPLITS_DIR}/config.json"
assert os.path.exists(DRIVE_ZIP_DIR), f"Missing {DRIVE_ZIP_DIR}"

with open(f"{SPLITS_DIR}/config.json") as f:
    config = json.load(f)

ACCENT_MAP = {int(k): v.capitalize() for k, v in config["id_to_accent"].items()}
NUM_ACCENT_CLASSES = int(config.get("num_accent_classes", len(ACCENT_MAP)))
TARGET_SR = int(config.get("target_sample_rate", 16000))

print("Accent map:", ACCENT_MAP)
print("Num accent classes:", NUM_ACCENT_CLASSES)
print("Target sample rate:", TARGET_SR)

# =========================================================
# Extract data
# =========================================================
SPEAKERS = [
    "RRBI", "SVBI", "TNI", "NJS",
    "HQTV", "MBMPS", "NCC", "TXHC",
    "HJK", "HKK", "YDCK", "YKWK",
    "ABA", "YBAA", "SKA", "ZHAA",
    "ASI", "BWC", "EBVS", "ERMS",
    "LXC", "PNV", "THV", "TLV"
]

for spk in SPEAKERS:
    spk_wav_dir = os.path.join(EXTRACT_PATH, spk, "wav")
    zip_path = os.path.join(DRIVE_ZIP_DIR, f"{spk}.zip")

    if os.path.isdir(spk_wav_dir) and len(os.listdir(spk_wav_dir)) > 0:
        print(f"Already extracted: {spk}")
        continue

    if not os.path.exists(zip_path):
        print(f"Missing zip for {spk}: {zip_path}")
        continue

    print(f"Extracting {spk}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(EXTRACT_PATH)

libri_on_drive = os.path.isdir(os.path.join(LIBRI_ROOT, "LibriSpeech", "dev-clean"))
libri_on_local = os.path.isdir(os.path.join(LIBRI_LOCAL, "LibriSpeech", "dev-clean"))

if libri_on_drive:
    print("LibriSpeech found on Drive.")
elif libri_on_local:
    print("LibriSpeech already present locally.")
else:
    print("Downloading LibriSpeech dev-clean locally...")
    torchaudio.datasets.LIBRISPEECH(root=LIBRI_LOCAL, url="dev-clean", download=True)

def resolve_librispeech_path(orig_path):
    tail = orig_path.split("librispeech/", 1)[-1]
    drive_path = os.path.join(LIBRI_ROOT, tail)
    local_path = os.path.join(LIBRI_LOCAL, tail)
    if os.path.isfile(drive_path):
        return drive_path
    return local_path

def load_and_fix_paths(split_name):
    with open(f"{SPLITS_DIR}/{split_name}.json") as f:
        records = json.load(f)

    for r in records:
        orig = r["wav_path"].replace("\\", "/")
        if "librispeech" in orig.lower():
            r["wav_path"] = resolve_librispeech_path(orig)
        else:
            r["wav_path"] = os.path.join(
                EXTRACT_PATH, r["speaker"], "wav", os.path.basename(orig)
            )
        r["transcript"] = str(r["transcript"]).strip().lower()
        r["accent_id"] = int(r["accent_id"])
    return records

def maybe_limit(records, limit=None, seed=SEED):
    if limit is None:
        return records
    records = records.copy()
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:limit]

def build_balanced_train_subset(records, per_accent=2000, seed=SEED):
    rng = random.Random(seed)
    buckets = defaultdict(list)

    for r in records:
        buckets[int(r["accent_id"])].append(r)

    balanced = []
    for aid in sorted(buckets.keys()):
        group = buckets[aid]
        rng.shuffle(group)
        if len(group) >= per_accent:
            selected = group[:per_accent]
        else:
            selected = group.copy()
            while len(selected) < per_accent:
                selected.append(rng.choice(group))
        balanced.extend(selected)

    rng.shuffle(balanced)
    return balanced

all_train_records = load_and_fix_paths("train")
train_records = build_balanced_train_subset(all_train_records, per_accent=PER_ACCENT_TRAIN, seed=SEED)
val_records = maybe_limit(load_and_fix_paths("val"), VAL_LIMIT)
test_records = maybe_limit(load_and_fix_paths("test"), TEST_LIMIT)

print("Dataset sizes:")
print("  Train:", len(train_records))
print("  Val  :", len(val_records))
print("  Test :", len(test_records))
print("Accent classes:", NUM_ACCENT_CLASSES)

for aid, cnt in sorted(Counter(r["accent_id"] for r in train_records).items()):
    print(f"  {ACCENT_MAP[aid]:10s}: {cnt}")

# =========================================================
# Processor + Augmentation + Dataset
# =========================================================
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
print("Whisper processor ready.")

def speed_perturb_augment(waveform, sample_rate, speed_range=(0.95, 1.05)):
    speed_factor = random.uniform(*speed_range)
    waveform_out, _ = T.Speed(orig_freq=sample_rate, factor=speed_factor)(waveform)
    new_sr = int(sample_rate * speed_factor)
    if new_sr != sample_rate:
        waveform_out = T.Resample(new_sr, sample_rate)(waveform_out)
    return waveform_out

def spec_augment(features, freq_mask_param=15, time_mask_param=40, num_masks=2):
    features = features.clone()
    for _ in range(num_masks):
        features = T.FrequencyMasking(freq_mask_param)(features)
        features = T.TimeMasking(time_mask_param)(features)
    return features

class MirrorSpeechDataset(Dataset):
    def __init__(self, records, processor, augment=False, target_sr=16000):
        self.records = records
        self.processor = processor
        self.augment = augment
        self.target_sr = target_sr

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        waveform, sr = sf.read(r["wav_path"])
        waveform = torch.tensor(waveform, dtype=torch.float32)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            waveform = T.Resample(sr, self.target_sr)(waveform)

        if self.augment and random.random() < 0.5:
            waveform = speed_perturb_augment(waveform, self.target_sr)

        audio = waveform.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

        input_features = self.processor(
            audio,
            sampling_rate=self.target_sr,
            return_tensors="pt",
        ).input_features.squeeze(0)

        if self.augment:
            input_features = spec_augment(input_features)

        return {
            "input_features": input_features,
            "transcript": r["transcript"],
            "accent_id": torch.tensor(r["accent_id"], dtype=torch.long),
        }

def collate_fn(batch):
    input_features = torch.stack([x["input_features"] for x in batch])
    transcripts = [x["transcript"] for x in batch]
    accent_ids = torch.stack([x["accent_id"] for x in batch])

    tokenized = processor.tokenizer(
        transcripts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=448,
    )
    labels = tokenized.input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "input_features": input_features,
        "labels": labels,
        "transcripts": transcripts,
        "accent_ids": accent_ids,
    }

train_dataset = MirrorSpeechDataset(train_records, processor, augment=True, target_sr=TARGET_SR)
val_dataset = MirrorSpeechDataset(val_records, processor, augment=False, target_sr=TARGET_SR)
test_dataset = MirrorSpeechDataset(test_records, processor, augment=False, target_sr=TARGET_SR)

print("Datasets ready.")

_pin = torch.cuda.is_available()
_persist = NUM_WORKERS > 0

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=_pin,
    persistent_workers=_persist,
    collate_fn=collate_fn,
    drop_last=True,
    prefetch_factor=2 if _persist else None,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=_pin,
    persistent_workers=_persist,
    collate_fn=collate_fn,
    prefetch_factor=2 if _persist else None,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=_pin,
    persistent_workers=_persist,
    collate_fn=collate_fn,
    prefetch_factor=2 if _persist else None,
)

print("Dataloaders ready:")
print("  Train batches:", len(train_loader))
print("  Val batches  :", len(val_loader))
print("  Test batches :", len(test_loader))

# =========================================================
# Model
# =========================================================
class ContentHead(nn.Module):
    def __init__(self, input_dim=768, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class AccentHead(nn.Module):
    def __init__(self, input_dim=768, accent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, accent_dim),
            nn.LayerNorm(accent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x).mean(dim=1)

def accent_swap(content_vec, accent_vec):
    accent_expanded = accent_vec.unsqueeze(1).expand(-1, content_vec.size(1), -1)
    return torch.cat([content_vec, accent_expanded], dim=-1)

class AccentClassifier(nn.Module):
    def __init__(self, accent_dim=128, num_accents=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(accent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_accents),
        )

    def forward(self, x):
        return self.net(x)

class ReEncoder(nn.Module):
    def __init__(self, swap_dim=384, whisper_input_dim=80):
        super().__init__()
        self.projector = nn.Linear(swap_dim, whisper_input_dim)

    def forward(self, whisper_encoder, content_head, swapped):
        projected = self.projector(swapped).transpose(1, 2)
        whisper_input = F.interpolate(projected, size=3000, mode="linear", align_corners=False)
        reencoded = whisper_encoder(whisper_input).last_hidden_state
        return content_head(reencoded)

class LCMALoss(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, original_content, reencoded_content):
        return self.beta * self.mse(original_content, reencoded_content)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.accent_ce = nn.CrossEntropyLoss()
        self.lcma = LCMALoss(beta=beta)

    def forward(
        self,
        asr_loss,
        accent_logits,
        accent_labels,
        original_content=None,
        reencoded_content=None,
        use_lcma=True,
    ):
        accent_loss = self.accent_ce(accent_logits, accent_labels)
        if use_lcma and original_content is not None and reencoded_content is not None:
            lcma_loss = self.lcma(original_content, reencoded_content)
        else:
            lcma_loss = torch.tensor(0.0, device=asr_loss.device)

        total_loss = asr_loss + self.alpha * accent_loss + lcma_loss
        return {
            "total_loss": total_loss,
            "asr_loss": asr_loss,
            "accent_loss": accent_loss,
            "lcma_loss": lcma_loss,
        }

def build_model(device, num_accent_classes=NUM_ACCENT_CLASSES):
    whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    whisper.config.use_cache = False
    whisper.generation_config.forced_decoder_ids = forced_decoder_ids
    whisper.generation_config.suppress_tokens = []

    for p in whisper.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    whisper = get_peft_model(whisper, lora_cfg).to(device)

    return {
        "whisper": whisper,
        "content_head": ContentHead().to(device),
        "accent_head": AccentHead().to(device),
        "accent_classifier": AccentClassifier(num_accents=num_accent_classes).to(device),
        "reencoder": ReEncoder().to(device),
        "criterion": CombinedLoss(alpha=ACCENT_LOSS_ALPHA, beta=LCMA_BETA),
    }

def build_optimizer(components, lr=LR):
    params = []
    for name, module in components.items():
        if name == "criterion":
            continue
        params.extend([p for p in module.parameters() if p.requires_grad])
    return torch.optim.AdamW(params, lr=lr)

def get_whisper_encoder(whisper):
    if hasattr(whisper, "get_encoder"):
        return whisper.get_encoder()
    if hasattr(whisper, "model") and hasattr(whisper.model, "encoder"):
        return whisper.model.encoder
    if hasattr(whisper, "base_model"):
        base = whisper.base_model
        if hasattr(base, "model") and hasattr(base.model, "encoder"):
            return base.model.encoder
        if hasattr(base, "model") and hasattr(base.model, "model") and hasattr(base.model.model, "encoder"):
            return base.model.model.encoder
    raise AttributeError("Could not find Whisper encoder")

def shift_tokens_right(labels, pad_token_id, decoder_start_token_id):
    shifted = labels.new_zeros(labels.shape)
    shifted[:, 1:] = labels[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    shifted[shifted == -100] = pad_token_id
    return shifted

def get_whisper_seq2seq_model(whisper):
    if hasattr(whisper, "get_base_model"):
        return whisper.get_base_model()
    if hasattr(whisper, "base_model") and hasattr(whisper.base_model, "model"):
        return whisper.base_model.model
    return whisper

def forward_pass(components, input_features, labels, accent_labels, use_accent_swap=True, use_lcma=True):
    whisper = components["whisper"]
    content_head = components["content_head"]
    accent_head = components["accent_head"]
    accent_classifier = components["accent_classifier"]
    reencoder = components["reencoder"]
    criterion = components["criterion"]

    seq2seq_model = get_whisper_seq2seq_model(whisper)
    model_cfg = seq2seq_model.config

    decoder_input_ids = shift_tokens_right(
        labels,
        pad_token_id=model_cfg.pad_token_id,
        decoder_start_token_id=model_cfg.decoder_start_token_id,
    )

    seq2seq_out = seq2seq_model(
        input_features=input_features,
        decoder_input_ids=decoder_input_ids,
        use_cache=False,
        return_dict=True,
    )

    logits = seq2seq_out.logits
    vocab_size = logits.size(-1)

    asr_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )

    whisper_encoder = get_whisper_encoder(whisper)
    encoder_hidden = whisper_encoder(input_features).last_hidden_state
    content_vec = content_head(encoder_hidden)
    accent_vec = accent_head(encoder_hidden)
    accent_logits = accent_classifier(accent_vec)

    reencoded_content = None
    if use_accent_swap:
        neutral_accent = torch.zeros_like(accent_vec)
        swapped = accent_swap(content_vec, neutral_accent)
        reencoded_content = reencoder(whisper_encoder, content_head, swapped)

    return criterion(
        asr_loss=asr_loss,
        accent_logits=accent_logits,
        accent_labels=accent_labels,
        original_content=content_vec,
        reencoded_content=reencoded_content,
        use_lcma=use_lcma,
    )

# =========================================================
# Metrics
# =========================================================
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sentence_error_rate(refs, hyps) -> float:
    if len(refs) == 0:
        return 0.0
    wrong = sum(1 for r, h in zip(refs, hyps) if normalize_text(r) != normalize_text(h))
    return wrong / len(refs)

_g2p_singleton = None

def _get_g2p():
    global _g2p_singleton
    if _g2p_singleton is None:
        from g2p_en import G2p
        _g2p_singleton = G2p()
    return _g2p_singleton

def text_to_phones(text: str) -> str:
    try:
        g2p = _get_g2p()
        phones = [p for p in g2p(normalize_text(text)) if str(p).strip() and p != " "]
        return " ".join(map(str, phones))
    except Exception:
        return " ".join(list(normalize_text(text).replace(" ", "")))

def phoneme_error_rate(refs, hyps) -> float:
    if len(refs) == 0:
        return 0.0
    ref_phones = [text_to_phones(x) for x in refs]
    hyp_phones = [text_to_phones(x) for x in hyps]
    return wer(ref_phones, hyp_phones)

def compute_all_metrics(refs, hyps):
    refs = [normalize_text(x) for x in refs]
    hyps = [normalize_text(x) for x in hyps]

    metrics = {
        "wer": wer(refs, hyps) if refs else 0.0,
        "cer": cer(refs, hyps) if refs else 0.0,
        "ser": sentence_error_rate(refs, hyps),
        "per": phoneme_error_rate(refs, hyps),
        "wil": wil(refs, hyps) if refs else 0.0,
        "bertscore_f1": 0.0,
    }

    if refs:
        _, _, f1 = bertscore_score(hyps, refs, lang="en", verbose=False)
        metrics["bertscore_f1"] = float(f1.mean().item())

    return metrics

@torch.no_grad()
def evaluate_wer(whisper_model, processor, loader, device, accent_map, max_batches=None):
    whisper_model.eval()
    refs_by_accent = defaultdict(list)
    hyps_by_accent = defaultdict(list)

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        feats = batch["input_features"].to(device, non_blocking=True)
        transcripts = batch["transcripts"]
        accent_ids = batch["accent_ids"]

        pred_ids = whisper_model.generate(
            input_features=feats,
            language="en",
            task="transcribe",
        )
        predictions = processor.batch_decode(pred_ids, skip_special_tokens=True)

        for ref, hyp, aid in zip(transcripts, predictions, accent_ids.tolist()):
            accent_name = accent_map[aid]
            refs_by_accent[accent_name].append(ref.lower().strip())
            hyps_by_accent[accent_name].append(hyp.lower().strip())

    results = {}
    all_refs = []
    all_hyps = []

    for accent_name in sorted(refs_by_accent.keys()):
        refs = refs_by_accent[accent_name]
        hyps = hyps_by_accent[accent_name]
        results[accent_name] = {
            "wer": round(float(wer(refs, hyps)), 4),
            "cer": round(float(cer(refs, hyps)), 4),
            "num_samples": len(refs),
        }
        all_refs.extend(refs)
        all_hyps.extend(hyps)

    results["overall"] = {
        "wer": round(float(wer(all_refs, all_hyps)), 4),
        "cer": round(float(cer(all_refs, all_hyps)), 4),
        "num_samples": len(all_refs),
    }
    return results

@torch.no_grad()
def evaluate_all_metrics(whisper_model, processor, loader, device, accent_map):
    whisper_model.eval()
    refs_by_accent = defaultdict(list)
    hyps_by_accent = defaultdict(list)

    for batch_idx, batch in enumerate(loader):
        feats = batch["input_features"].to(device, non_blocking=True)
        transcripts = batch["transcripts"]
        accent_ids = batch["accent_ids"]

        pred_ids = whisper_model.generate(
            input_features=feats,
            language="en",
            task="transcribe",
        )
        predictions = processor.batch_decode(pred_ids, skip_special_tokens=True)

        for ref, hyp, aid in zip(transcripts, predictions, accent_ids.tolist()):
            accent_name = accent_map[aid]
            refs_by_accent[accent_name].append(ref)
            hyps_by_accent[accent_name].append(hyp)

    results = {}
    all_refs = []
    all_hyps = []

    for accent_name in sorted(refs_by_accent.keys()):
        refs = refs_by_accent[accent_name]
        hyps = hyps_by_accent[accent_name]
        metrics = compute_all_metrics(refs, hyps)
        results[accent_name] = {
            "wer": round(metrics["wer"], 4),
            "cer": round(metrics["cer"], 4),
            "ser": round(metrics["ser"], 4),
            "per": round(metrics["per"], 4),
            "wil": round(metrics["wil"], 4),
            "bertscore_f1": round(metrics["bertscore_f1"], 4),
            "num_samples": len(refs),
        }
        all_refs.extend(refs)
        all_hyps.extend(hyps)

    overall = compute_all_metrics(all_refs, all_hyps)
    results["overall"] = {
        "wer": round(overall["wer"], 4),
        "cer": round(overall["cer"], 4),
        "ser": round(overall["ser"], 4),
        "per": round(overall["per"], 4),
        "wil": round(overall["wil"], 4),
        "bertscore_f1": round(overall["bertscore_f1"], 4),
        "num_samples": len(all_refs),
    }

    return results

def print_full_eval_results(title, results):
    print("=" * 100)
    print(title)
    print("=" * 100)
    for name, metrics in results.items():
        print(
            f"{name:10s}  "
            f"WER={metrics['wer']:.4f}  CER={metrics['cer']:.4f}  "
            f"SER={metrics['ser']:.4f}  PER={metrics['per']:.4f}  "
            f"WIL={metrics['wil']:.4f}  BERT={metrics['bertscore_f1']:.4f}  "
            f"(n={metrics['num_samples']})"
        )

# =========================================================
# Checkpointing
# =========================================================
def save_checkpoint(components, optimizer, epoch, history, path, extra=None):
    ckpt = {
        "epoch": epoch,
        "whisper_lora_state": components["whisper"].state_dict(),
        "content_head_state": components["content_head"].state_dict(),
        "accent_head_state": components["accent_head"].state_dict(),
        "accent_classifier_state": components["accent_classifier"].state_dict(),
        "reencoder_state": components["reencoder"].state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "num_accent_classes": NUM_ACCENT_CLASSES,
        "accent_map": ACCENT_MAP,
        "run_name": RUN_NAME,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)
    print(f"Saved checkpoint -> {path}")

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    components = build_model(device, ckpt.get("num_accent_classes", NUM_ACCENT_CLASSES))
    components["whisper"].load_state_dict(ckpt["whisper_lora_state"], strict=False)
    components["content_head"].load_state_dict(ckpt["content_head_state"])
    components["accent_head"].load_state_dict(ckpt["accent_head_state"])
    components["accent_classifier"].load_state_dict(ckpt["accent_classifier_state"])
    components["reencoder"].load_state_dict(ckpt["reencoder_state"])

    optimizer = build_optimizer(components, lr=LR)
    optimizer.load_state_dict(ckpt["optimizer_state"])

    epoch = ckpt.get("epoch", 0)
    history = ckpt.get("history", {})
    return components, optimizer, epoch, history

# =========================================================
# Training
# =========================================================
def train_epoch(
    components,
    optimizer,
    scaler,
    scheduler,
    loader,
    device,
    epoch,
    use_accent_swap=True,
    use_lcma=True,
    log_every=25,
):
    for module in components.values():
        if isinstance(module, nn.Module):
            module.train()

    totals = defaultdict(float)
    n_batches = 0

    trainable_params = [
        p for group in components.values()
        if isinstance(group, nn.Module)
        for p in group.parameters()
        if p.requires_grad
    ]

    for batch_idx, batch in enumerate(loader):
        input_features = batch["input_features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        accent_ids = batch["accent_ids"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=torch.cuda.is_available()):
            loss_dict = forward_pass(
                components=components,
                input_features=input_features,
                labels=labels,
                accent_labels=accent_ids,
                use_accent_swap=use_accent_swap,
                use_lcma=use_lcma,
            )

        scaler.scale(loss_dict["total_loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        for k, v in loss_dict.items():
            totals[k] += float(v.item())
        n_batches += 1

        if (batch_idx + 1) % SAVE_EVERY_N_BATCHES == 0:
            safety_path = os.path.join(
                CKPT_DIR, f"{RUN_NAME}_epoch{epoch}_batch{batch_idx+1}_safety.pt"
            )
            save_checkpoint(
                components,
                optimizer,
                epoch,
                {},
                safety_path,
                extra={
                    "experiment": "full",
                    "batch_idx": batch_idx + 1,
                    "checkpoint_type": "mid_epoch_safety",
                },
            )

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == len(loader):
            print(
                f"Epoch {epoch} | Batch {batch_idx + 1}/{len(loader)} | "
                f"total={totals['total_loss']/n_batches:.4f} | "
                f"asr={totals['asr_loss']/n_batches:.4f} | "
                f"accent={totals['accent_loss']/n_batches:.4f} | "
                f"lcma={totals['lcma_loss']/n_batches:.4f}"
            )

        del loss_dict

    return {k: v / max(n_batches, 1) for k, v in totals.items()}

@torch.no_grad()
def validate_epoch(components, loader, device, use_accent_swap=True, use_lcma=True):
    for module in components.values():
        if isinstance(module, nn.Module):
            module.eval()

    totals = defaultdict(float)
    n_batches = 0

    for batch in loader:
        input_features = batch["input_features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        accent_ids = batch["accent_ids"].to(device, non_blocking=True)

        with autocast(enabled=torch.cuda.is_available()):
            loss_dict = forward_pass(
                components=components,
                input_features=input_features,
                labels=labels,
                accent_labels=accent_ids,
                use_accent_swap=use_accent_swap,
                use_lcma=use_lcma,
            )

        for k, v in loss_dict.items():
            totals[k] += float(v.item())
        n_batches += 1
        del loss_dict

    return {k: v / max(n_batches, 1) for k, v in totals.items()}

def run_training(
    experiment_name,
    train_loader,
    val_loader,
    use_accent_swap=True,
    use_lcma=True,
    num_epochs=NUM_EPOCHS,
    lr=LR,
    patience=PATIENCE,
    resume_path=None,
    device=DEVICE,
):
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {experiment_name.upper()}")
    print(f"use_accent_swap={use_accent_swap} | use_lcma={use_lcma} | epochs={num_epochs} | lr={lr}")
    print("=" * 80)

    scaler = GradScaler(enabled=torch.cuda.is_available())

    if resume_path and os.path.exists(resume_path):
        components, optimizer, start_epoch, history = load_checkpoint(resume_path, device)
        print(f"Resumed from: {resume_path}")
        best_val_wer = min(history["val_wer"]) if history.get("val_wer") else math.inf
    else:
        components = build_model(device)
        optimizer = build_optimizer(components, lr=lr)
        start_epoch = 0
        best_val_wer = math.inf
        history = {
            "train_total": [],
            "train_asr": [],
            "train_accent": [],
            "train_lcma": [],
            "val_total": [],
            "val_asr": [],
            "val_accent": [],
            "val_lcma": [],
            "val_wer": [],
            "val_cer": [],
        }

    total_steps = len(train_loader) * num_epochs
    warmup_steps = min(100, max(10, total_steps // 20))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    patience_counter = 0
    best_ckpt_path = os.path.join(CKPT_DIR, f"{RUN_NAME}_{experiment_name}_best.pt")

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{start_epoch + num_epochs} ---")

        train_losses = train_epoch(
            components=components,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            loader=train_loader,
            device=device,
            epoch=epoch,
            use_accent_swap=use_accent_swap,
            use_lcma=use_lcma,
            log_every=LOG_EVERY,
        )

        val_losses = validate_epoch(
            components=components,
            loader=val_loader,
            device=device,
            use_accent_swap=use_accent_swap,
            use_lcma=use_lcma,
        )

        val_metrics = evaluate_wer(
            components["whisper"],
            processor,
            val_loader,
            device,
            ACCENT_MAP,
            max_batches=MAX_VAL_WER_BATCHES,
        )

        val_wer_overall = val_metrics["overall"]["wer"]
        val_cer_overall = val_metrics["overall"]["cer"]

        history["train_total"].append(train_losses["total_loss"])
        history["train_asr"].append(train_losses["asr_loss"])
        history["train_accent"].append(train_losses["accent_loss"])
        history["train_lcma"].append(train_losses["lcma_loss"])
        history["val_total"].append(val_losses["total_loss"])
        history["val_asr"].append(val_losses["asr_loss"])
        history["val_accent"].append(val_losses["accent_loss"])
        history["val_lcma"].append(val_losses["lcma_loss"])
        history["val_wer"].append(val_wer_overall)
        history["val_cer"].append(val_cer_overall)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Train total={train_losses['total_loss']:.4f} | "
            f"asr={train_losses['asr_loss']:.4f} | "
            f"accent={train_losses['accent_loss']:.4f} | "
            f"lcma={train_losses['lcma_loss']:.4f} | "
            f"lr={current_lr:.2e}"
        )
        print(
            f"Val   total={val_losses['total_loss']:.4f} | "
            f"asr={val_losses['asr_loss']:.4f} | "
            f"accent={val_losses['accent_loss']:.4f} | "
            f"lcma={val_losses['lcma_loss']:.4f} | "
            f"WER={val_wer_overall:.4f} | CER={val_cer_overall:.4f}"
        )

        epoch_ckpt_path = os.path.join(CKPT_DIR, f"{RUN_NAME}_{experiment_name}_epoch{epoch}.pt")
        save_checkpoint(
            components,
            optimizer,
            epoch,
            history,
            epoch_ckpt_path,
            extra={
                "experiment": experiment_name,
                "use_accent_swap": use_accent_swap,
                "use_lcma": use_lcma,
                "best_monitor": "val_wer",
            },
        )

        if val_wer_overall < best_val_wer:
            best_val_wer = val_wer_overall
            patience_counter = 0
            save_checkpoint(
                components,
                optimizer,
                epoch,
                history,
                best_ckpt_path,
                extra={
                    "experiment": experiment_name,
                    "use_accent_swap": use_accent_swap,
                    "use_lcma": use_lcma,
                    "best_monitor": "val_wer",
                },
            )
            print(f"New best validation WER: {best_val_wer:.4f}")
        else:
            patience_counter += 1
            print(f"No WER improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    best_components, _, _, history = load_checkpoint(best_ckpt_path, device)
    return best_components, history, best_ckpt_path

def main():
    components_full, history_full, best_ckpt_path = run_training(
        experiment_name="full",
        train_loader=train_loader,
        val_loader=val_loader,
        use_accent_swap=True,
        use_lcma=True,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        patience=PATIENCE,
        resume_path=RESUME_PATH,
        device=DEVICE,
    )

    results_full = evaluate_all_metrics(
        components_full["whisper"],
        processor,
        test_loader,
        DEVICE,
        ACCENT_MAP,
    )

    print_full_eval_results("Task 7 Test Results - Full", results_full)

    results = {
        "full": results_full,
        "task4_baseline": TASK4_BASELINE,
        "overall_delta_vs_task4": {
            "wer_delta": round(results_full["overall"]["wer"] - TASK4_BASELINE["overall"]["wer"], 4),
            "cer_delta": round(results_full["overall"]["cer"] - TASK4_BASELINE["overall"]["cer"], 4),
            "ser_delta": round(results_full["overall"]["ser"] - TASK4_BASELINE["overall"]["ser"], 4),
            "per_delta": round(results_full["overall"]["per"] - TASK4_BASELINE["overall"]["per"], 4),
            "wil_delta": round(results_full["overall"]["wil"] - TASK4_BASELINE["overall"]["wil"], 4),
            "bertscore_f1_delta": round(
                results_full["overall"]["bertscore_f1"] - TASK4_BASELINE["overall"]["bertscore_f1"], 4
            ),
        },
    }

    results_path = os.path.join(CKPT_DIR, f"{RUN_NAME}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    history_path = os.path.join(CKPT_DIR, f"{RUN_NAME}_history.json")
    with open(history_path, "w") as f:
        json.dump({"full": history_full}, f, indent=2)

    print("Saved:", results_path)
    print("Saved:", history_path)
    print("Best checkpoint:", best_ckpt_path)

    epochs = range(1, len(history_full["train_total"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history_full["train_total"], "o-", label="Train Total")
    axes[0].plot(epochs, history_full["val_total"], "s--", label="Val Total")
    axes[0].set_title("Balanced10k Full - Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history_full["val_wer"], "o-", label="Val WER")
    axes[1].plot(epochs, history_full["val_cer"], "s--", label="Val CER")
    axes[1].axhline(TASK4_BASELINE["overall"]["wer"], linestyle="--", alpha=0.7, label="Task4 WER")
    axes[1].set_title("Balanced10k Full - Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(CKPT_DIR, f"{RUN_NAME}_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", plot_path)

    print("=" * 100)
    print("MIRRORSPEECH BALANCED10K FINAL SUMMARY")
    print("=" * 100)
    print(f"Task 4 overall WER     : {TASK4_BASELINE['overall']['wer']:.4f}")
    print(f"Task 4 overall CER     : {TASK4_BASELINE['overall']['cer']:.4f}")
    print(f"Task 7 overall WER     : {results_full['overall']['wer']:.4f}")
    print(f"Task 7 overall CER     : {results_full['overall']['cer']:.4f}")
    print(f"Task 7 overall SER     : {results_full['overall']['ser']:.4f}")
    print(f"Task 7 overall PER     : {results_full['overall']['per']:.4f}")
    print(f"Task 7 overall WIL     : {results_full['overall']['wil']:.4f}")
    print(f"Task 7 overall BERT F1 : {results_full['overall']['bertscore_f1']:.4f}")
    print(f"Delta WER              : {results_full['overall']['wer'] - TASK4_BASELINE['overall']['wer']:+.4f}")
    print(f"Delta CER              : {results_full['overall']['cer'] - TASK4_BASELINE['overall']['cer']:+.4f}")
    print("=" * 100)
    print("Files saved in :", CKPT_DIR)
    print("Best checkpoint:", best_ckpt_path)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
