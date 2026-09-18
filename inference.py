# -*- coding: utf-8 -*-
"""
Bulaq 1280 AH ByT5 OCR Corrector - Inference Engine
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class BulaqOCRCorrector:
    def __init__(self, model_dir="./bulaq_byt5_ocr_corrector", device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Loading Bulaq ByT5 Corrector from {model_dir} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def correct(self, text, max_length=128):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    import sys
    corrector = BulaqOCRCorrector()
    test_samples = [
        "بلذنى يا الماك السعيد",
        "الملا كس لسانقا ل",
        "بولدىكاذما كان",
        "صارعبدة الملبا نمع أبيهضوء المكان",
        "سعد انتوشتكاليباحالة"
    ]
    print("\n--- Inference Demonstrations ---")
    for s in test_samples:
        print(f"Corrupt OCR: {s}")
        print(f"Healed Text: {corrector.correct(s)}\n")
