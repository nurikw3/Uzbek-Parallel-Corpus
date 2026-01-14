import polars as pl
from datasets import load_dataset
from deep_translator import GoogleTranslator
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import time

DATASET_NAME = "tahrirchi/lutfiy"
OUTPUT_FILE = "dataset.csv"
RANDOM_SEED = 42
TEST_FRACTION = 0.8
FIGURES_DIR = "figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_standard_uzbek_with_progress(texts, max_workers=10):
    results = [None] * len(texts)
    def translate_single(idx, text):
        try:
            translator = GoogleTranslator(source='en', target='uz')
            return idx, translator.translate(text)
        except:
            return idx, None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(translate_single, i, text): i for i, text in enumerate(texts)}
        for future in tqdm(as_completed(futures), total=len(texts)):
            idx, translation = future.result()
            results[idx] = translation
            time.sleep(0.01)
    return results

def process_data():
    ds = load_dataset(DATASET_NAME, split="train")
    df = pl.from_arrow(ds.data.table)
    df = df.filter(pl.col("src_lang") == "eng_Latn")
    df = df.rename({"src_sent": "en_source", "tgt_sent": "uz_south_source"})
    df = df.select(["en_source", "uz_south_source"]).with_row_index("id")
    df = df.unique().drop_nulls()

    english_texts = df["en_source"].to_list()
    synthetic_translations = generate_standard_uzbek_with_progress(english_texts, max_workers=10)
    df = df.with_columns(pl.Series(name="uz_north_synthetic_ref", values=synthetic_translations))
    df = df.drop_nulls(subset=["uz_north_synthetic_ref"])

    df = df.sample(fraction=1.0, shuffle=True, seed=RANDOM_SEED)
    test_size = int(df.height * TEST_FRACTION)
    df_test = df.slice(0, test_size).with_columns(pl.lit("test").alias("split"))
    df_val = df.slice(test_size, df.height - test_size).with_columns(pl.lit("validation").alias("split"))
    final_df = pl.concat([df_val, df_test])
    final_df.write_csv(OUTPUT_FILE)
    return final_df

def calculate_similarity(row):
    return SequenceMatcher(None, str(row['uz_south_source']), str(row['uz_north_synthetic_ref'])).ratio()

def run_eda():
    pl_df = pl.read_csv(OUTPUT_FILE)
    pl_df = pl_df.with_columns([
        pl.col("en_source").str.split(" ").list.len().alias("wc_en"),
        pl.col("uz_south_source").str.split(" ").list.len().alias("wc_uz_south"),
        pl.col("uz_north_synthetic_ref").str.split(" ").list.len().alias("wc_uz_north"),
    ])
    df = pl_df.to_pandas()
    df['dialect_similarity'] = df.apply(calculate_similarity, axis=1)

    plt.figure(figsize=(12,6))
    sns.set_style("whitegrid")
    sns.kdeplot(data=df, x='wc_en', fill=True, alpha=0.3)
    sns.kdeplot(data=df, x='wc_uz_south', fill=True, alpha=0.3)
    sns.kdeplot(data=df, x='wc_uz_north', fill=True, alpha=0.3)
    plt.savefig(f"{FIGURES_DIR}/word_count_distribution.png", dpi=300)
    plt.close()

    fig = px.scatter(
        df,
        x="wc_en",
        y="wc_uz_south",
        color="dialect_similarity",
        hover_data=["en_source","uz_south_source","uz_north_synthetic_ref"],
        title="Sentence Length Correlation & Dialect Similarity",
        labels={"wc_en":"English Word Count","wc_uz_south":"South Uz Word Count"},
        color_continuous_scale="RdBu"
    )
    fig.write_html(f"{FIGURES_DIR}/interactive_analysis.html")

if __name__ == "__main__":
    process_data()
    run_eda()
