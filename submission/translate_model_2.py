import torch
import polars as pl
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm

MODEL_NAME = "facebook/m2m100_418M"
INPUT_FILE = "dataset.csv"
OUTPUT_FILE = "predictions_model_2.csv"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SRC_LANG = "uz"
TGT_LANG_ENG = "en"
TGT_LANG_UZN = "uz"

def load_model():
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return tokenizer, model

def translate_batch(texts, tokenizer, model, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
            max_length=128
        )

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

def run_inference():
    df = pl.read_csv(INPUT_FILE)
    sources = df["uz_south_source"].to_list()

    tokenizer, model = load_model()

    preds_eng = []
    for i in tqdm(range(0, len(sources), BATCH_SIZE)):
        batch = sources[i:i + BATCH_SIZE]
        preds_eng.extend(translate_batch(batch, tokenizer, model, SRC_LANG, TGT_LANG_ENG))

    preds_uzn = []
    for i in tqdm(range(0, len(sources), BATCH_SIZE)):
        batch = sources[i:i + BATCH_SIZE]
        preds_uzn.extend(translate_batch(batch, tokenizer, model, SRC_LANG, TGT_LANG_UZN))

    result_df = df.with_columns([
        pl.Series(name="pred_eng_m2m", values=preds_eng),
        pl.Series(name="pred_uzn_m2m", values=preds_uzn)
    ])

    result_df.write_csv(OUTPUT_FILE)

if __name__ == "__main__":
    run_inference()
