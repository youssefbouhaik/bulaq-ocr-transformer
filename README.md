# 🕌 Bulaq 1280 AH ByT5 OCR Transformer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/youssefbouhaik/bulaq-ocr-transformer/blob/main/Train_Bulaq_Transformer_Colab.ipynb)

Neural sequence-to-sequence Transformer (`google/byt5-small`) fine-tuned to heal and de-noise 19th-century Arabic lithographic OCR corruptions, specifically trained on the **Bulaq 1280 AH (1863 CE)** edition of *Alf Layla wa-Layla*.

---

## ⚡ 1-Click Training in Google Colab Pro

1. Click the **Open in Colab** badge above or visit:  
   👉 [Open in Colab](https://colab.research.google.com/github/youssefbouhaik/bulaq-ocr-transformer/blob/main/Train_Bulaq_Transformer_Colab.ipynb)
2. In Colab, enable GPU acceleration:  
   **Runtime** ➔ **Change runtime type** ➔ Select **A100 GPU** or **L4 GPU**.
3. Select **Runtime** ➔ **Run all** (`Cmd+F9`).
4. Training on **28,468 pairs** takes **~10–12 minutes** on an A100.
5. The notebook saves fine-tuned weights directly into your Google Drive or session folder.

---

## 📁 Repository Contents

- `Train_Bulaq_Transformer_Colab.ipynb`: Ready-to-run Jupyter notebook with GPU mixed precision, evaluation, and live interactive test widget.
- `train_transformer.py`: Standalone CLI training script.
- `inference.py`: Python module to load the fine-tuned model and heal raw OCR strings.
- `data/bulaq_ocr_fallacies_dataset.jsonl`: 28,468 verified empirical Bulaq OCR fallacy pairs.

---

## 🚀 Why ByT5 for Historical Arabic OCR?

Standard subword tokenizers (like BERT, RoBERTa, or standard T5) break down on historical Arabic lithography because broken character connections, missing i'jam dots, and OCR noise produce `<unk>` tokens. 

**ByT5 operates at the raw UTF-8 byte level**, which provides:
- **0% Out-Of-Vocabulary (OOV) rate**.
- Native ability to learn character-level substitutions (e.g. `ب` ↔ `ن` ↔ `ي` ↔ `ت`).
- Native healing of kerning fractures and fused words without fixed vocabulary restrictions.
