import torch
import polars as pl 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from tqdm import tqdm
import warnings
import re
import gc 

warnings.filterwarnings("ignore")


MODEL_NAME = "facebook/nllb-200-distilled-600M"
INPUT_FILE = "dataset.csv"
OUTPUT_FILE = "predictions_model_1.csv"
BATCH_SIZE = 32
MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SRC_LANG_SOUTH = "uzs_Arab"
TGT_LANG_STD   = "uzn_Latn"
TGT_LANG_ENG   = "eng_Latn"

GEN_CONFIG = {
    "max_length": MAX_LENGTH,
    "num_beams": 5,
    "no_repeat_ngram_size": 3,
    "length_penalty": 1.0,
    "early_stopping": True,
}


def clean_translation(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # "Hello   world   !" | "Hello world !"
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # "Hello world !" | "Hello world!"
    text = re.sub(r'([.,!?;:])([^\s\d])', r'\1 \2', text)  # "Hello world!How are you?" | "Hello world! How are you?"
    text = re.sub(r'([.,;:])(\1)+', r'\1', text)  # "Hello!!! How are you??" | "Hello! How are you?"
    text = text.strip() 
    words = text.split()  
    
    out = []
    for i, w in enumerate(words):
        if i == 0 or w.lower() != words[i-1].lower():
            out.append(w)  
    text = ' '.join(out)
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]
    return text


def clean_batch(translations: list[str]) -> list[str]:
    return [clean_translation(x) for x in translations]


def clean_cuda():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


def translate_batch(texts: list[str], tokenizer, model, src_lang: str, tgt_lang: str) -> list[str]:
    tokenizer.src_lang = src_lang 
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    with torch.no_grad():
        out = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, **GEN_CONFIG)

    return clean_batch(tokenizer.batch_decode(out, skip_special_tokens=True))


def translate_with_ensembling(texts: list[str], tokenizer, model, src_lang: str, tgt_lang: str, n_samples: int = 3):
    tokenizer.src_lang = src_lang 
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=MAX_LENGTH,
            num_beams=5,
            num_return_sequences=n_samples,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True,
            do_sample=True,
            temperature=0.8,
        )
    all_translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    best_translation = []
    n = len(all_translations)

    for i in range(0, n, n_samples):
        candidates = all_translations[i:i + n_samples]
        best = max(candidates, key=len)
        best_translation.append(best)
    
    return clean_batch(best_translation)


def run_inference():
    df = pl.read_csv(INPUT_FILE)

    tokenizer, model = load_model()
    sources = df["uz_south_source"].to_list()
    print(f"\nProcessing {len(sources)} samples..., {BATCH_SIZE=}")

    print('\n[1/4] Normalizing South -> North')
    preds_uzn = []
    for i in tqdm(range(0, len(sources), BATCH_SIZE), desc="Normalizing"):
        batch = sources[i:i + BATCH_SIZE]
        preds_uzn.extend(translate_batch(batch, tokenizer, model, SRC_LANG_SOUTH, TGT_LANG_STD))
    
    clean_cuda()
    
    print('\n[2/4] Direct translation South -> Eng')
    preds_eng_direct = []
    for i in tqdm(range(0, len(sources), BATCH_SIZE), desc="Direct translation"):
        batch = sources[i:i + BATCH_SIZE]
        preds_eng_direct.extend(translate_batch(batch, tokenizer, model, SRC_LANG_SOUTH, TGT_LANG_ENG))
    
    clean_cuda()

    print('\n[3/4] Pivot translation South -> North -> Eng')
    preds_eng_pivot = []
    for i in tqdm(range(0, len(preds_uzn), BATCH_SIZE), desc="Pivot"):
        batch = preds_uzn[i:i + BATCH_SIZE]
        preds_eng_pivot.extend(translate_batch(batch, tokenizer, model, TGT_LANG_STD, TGT_LANG_ENG))
    
    clean_cuda()

    print('\n[4/4] Ensembling pivot translations')
    preds_eng_ensemble = []
    for i in tqdm(range(0, len(preds_uzn), BATCH_SIZE), desc="Ensemble"):
        batch = preds_uzn[i:i + BATCH_SIZE]
        preds_eng_ensemble.extend(translate_with_ensembling(batch, tokenizer, model, TGT_LANG_STD, TGT_LANG_ENG))
    
    print("\nSaving results")
    result_df = df.with_columns([
        pl.Series(name="nllb_uz_to_uz", values=preds_uzn),            
        pl.Series(name="nllb_uz_to_en_direct", values=preds_eng_direct), 
        pl.Series(name="nllb_uz_to_en_pivot", values=preds_eng_pivot),    
        pl.Series(name="nllb_uz_to_en_ensemble", values=preds_eng_ensemble) 
    ])
    
    result_df.write_csv(OUTPUT_FILE)
    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_inference()