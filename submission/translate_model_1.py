import torch
import polars as pl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "facebook/nllb-200-distilled-600M"
INPUT_FILE = "dataset.csv"
OUTPUT_FILE = "predictions_model_1.csv"
BATCH_SIZE = 16
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SRC_LANG = "uzs_Arab"
TGT_LANG_ENG = "eng_Latn"
TGT_LANG_UZN = "uzn_Latn"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return tokenizer, model

def translate_batch(texts, tokenizer, model, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    ).to(DEVICE)

    try:
        tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    except:
        tgt_lang_id = tokenizer.lang_code_to_id.get(tgt_lang, tokenizer.eos_token_id)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=MAX_LENGTH,
            num_beams=5
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
        pl.Series(name="nllb_uz_to_en", values=preds_eng),
        pl.Series(name="nllb_uz_to_uz", values=preds_uzn)
    ])

    result_df.write_csv(OUTPUT_FILE)

if __name__ == "__main__":
    run_inference()
