import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
import mlflow
import evaluate
from sacrebleu.metrics import BLEU, CHRF
from bert_score import score as bert_score

warnings.filterwarnings("ignore")

# ================== CONFIG ==================

FILES = {
    "NLLB-200": "predictions_model_1.csv",
    "M2M100": "predictions_model_2.csv"
}

EXPERIMENT_NAME = "South_Uzbek_Translation_Benchmark"
COMPUTE_BERT = True
BERT_BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== METRICS ==================

def compute_bleu(preds: list[str], refs: list[str]) -> float:
    refs = [[r] for r in refs]
    return BLEU().corpus_score(preds, refs).score


def compute_chrf(preds: list[str], refs: list[str]) -> float:
    refs = [[r] for r in refs]
    return CHRF().corpus_score(preds, refs).score


def compute_meteor(preds: list[str], refs: list[str]) -> float:
    meteor = evaluate.load("meteor")
    preds = [p or "" for p in preds]
    refs = [r or "" for r in refs]
    return meteor.compute(predictions=preds, references=refs)["meteor"] * 100


def compute_length_stats(preds: list[str], refs: list[str]) -> dict[str, float]:
    p = [len(str(x).split()) for x in preds]
    r = [len(str(x).split()) for x in refs]
    return {
        "avg_pred_length": np.mean(p),
        "avg_ref_length": np.mean(r),
        "length_ratio": np.mean(p) / max(np.mean(r), 1e-6),
        "std_pred_length": np.std(p),
        "std_ref_length": np.std(r),
    }


# ================== BERTSCORE ==================

def compute_bertscore(preds: list[str], refs: list[str], lang: str) -> dict[str, float]:
    if not COMPUTE_BERT:
        return {"bertscore_f1": 0.0, "bertscore_precision": 0.0, "bertscore_recall": 0.0}

    preds = [str(p or "") for p in preds]
    refs = [str(r or "") for r in refs]

    if lang == "en":
        model_type = "microsoft/deberta-xlarge-mnli"
    else:
        model_type = "xlm-roberta-base"

    P, R, F1 = bert_score(
        preds,
        refs,
        model_type=model_type,
        lang=lang if lang == "en" else None,
        device=DEVICE,
        batch_size=BERT_BATCH_SIZE,
        verbose=False
    )

    return {
        "bertscore_precision": P.mean().item() * 100,
        "bertscore_recall": R.mean().item() * 100,
        "bertscore_f1": F1.mean().item() * 100,
    }


# ================== CORE EVAL ==================

def evaluate_task(preds: list[str], refs: list[str], lang: str, name: str) -> dict[str, float]:
    print(f"Evaluating: {name}")

    results = {
        "bleu": compute_bleu(preds, refs),
        "chrf": compute_chrf(preds, refs),
        "meteor": compute_meteor(preds, refs),
    }

    results.update(compute_length_stats(preds, refs))
    results.update(compute_bertscore(preds, refs, lang))

    print(
        f"    BLEU={results['bleu']:.2f} | chrF={results['chrf']:.2f} | "
        f"METEOR={results['meteor']:.2f} | BERT-F1={results['bertscore_f1']:.2f}"
    )

    return results


# ================== MODEL CONFIGS ==================

@dataclass
class TaskConfig:
    pred_col: str
    ref_col: str
    lang: str
    name: str


NLLB_TASKS = [
    TaskConfig("nllb_uz_to_uz", "uz_north_synthetic_ref", "other", "Uzbek normalization"),
    TaskConfig("nllb_uz_to_en_direct", "en_source", "en", "Direct translation"),
    TaskConfig("nllb_uz_to_en_pivot", "en_source", "en", "Pivot translation"),
    TaskConfig("nllb_uz_to_en_ensemble", "en_source", "en", "Ensemble translation"),
]

M2M_TASKS = [
    TaskConfig("pred_uzn_m2m", "uz_north_synthetic_ref", "other", "Uzbek normalization"),
    TaskConfig("pred_eng_m2m", "en_source", "en", "English translation"),
]


# ================== EVALUATION ==================

def print_comparison_table(all_results: dict[str, dict]):
    print("\n" + "=" * 100)
    print("FINAL COMPARISON TABLE")
    print("=" * 100)

    for model_name, results in all_results.items():
        print(f"\nModel: {model_name}")

        for task_name, metrics in results.items():
            print(f"{task_name}")

            line = (
                f"    BLEU: {metrics['bleu']:.2f} | "
                f"chrF: {metrics['chrf']:.2f} | "
                f"METEOR: {metrics['meteor']:.2f}"
            )

            if "bertscore_f1" in metrics:
                line += f" | BERT-F1: {metrics['bertscore_f1']:.2f}"

            print(line)


def evaluate_model(model_name: str, path: str):
    print(f"\nLoading: {path}")
    df = pl.read_csv(path)

    if "split" in df.columns:
        df = df.filter(pl.col("split") == "validation") 
        print(f"Using validation split: {df.height} samples")

    tasks = NLLB_TASKS if "nllb" in model_name.lower() else M2M_TASKS

    results = {}

    for task in tasks:
        if task.pred_col not in df.columns:
            continue

        preds = df[task.pred_col].to_list()
        refs = df[task.ref_col].to_list()

        results[task.name] = evaluate_task(preds, refs, task.lang, task.name)

    return results, df


# ================== MLFLOW ==================

def log_to_mlflow(model_name: str, results: dict[str, dict], df: pl.DataFrame, filename: str):
    with mlflow.start_run(run_name=f"Eval_{model_name}"):
        mlflow.log_param("model", model_name)
        mlflow.log_param("samples", df.height)
        mlflow.log_param("file", filename)

        for task, metrics in results.items():
            prefix = task.replace(" ", "_").lower()
            for k, v in metrics.items():
                mlflow.log_metric(f"{prefix}_{k}", v)

        mlflow.log_artifact(filename)


# ================== RUN ==================

def main():
    print("Starting evaluation")
    print("Device:", DEVICE)
    print("BERTScore:", COMPUTE_BERT)

    mlflow.set_experiment(EXPERIMENT_NAME)

    all_results = {}

    for model_name, path in FILES.items():
        results, df = evaluate_model(model_name, path)
        log_to_mlflow(model_name, results, df, path)
        all_results[model_name] = results

    print("\nDONE\nwrite mlflow ui --port 5001 to view results")


if __name__ == "__main__":
    main()