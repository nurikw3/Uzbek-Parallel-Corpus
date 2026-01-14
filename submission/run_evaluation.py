import polars as pl
import mlflow
import evaluate
import os

FILES = {
    "NLLB-200": "predictions_model_1.csv",
    "M2M100": "predictions_model_2.csv"
}

EXPERIMENT_NAME = "South_Uzbek_Translation_Benchmark"

def calculate_metrics(predictions, references):
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    meteor = evaluate.load("meteor")

    clean_preds = [p if p else "" for p in predictions]
    clean_refs = [r if r else "" for r in references]

    res_bleu = bleu.compute(predictions=clean_preds, references=clean_refs)
    res_chrf = chrf.compute(predictions=clean_preds, references=clean_refs)
    res_meteor = meteor.compute(predictions=clean_preds, references=clean_refs)

    return {
        "bleu": res_bleu["score"],
        "chrf": res_chrf["score"],
        "meteor": res_meteor["meteor"]
    }

def run_evaluation():
    mlflow.set_experiment(EXPERIMENT_NAME)

    for model_name, filename in FILES.items():
        if not os.path.exists(filename):
            continue

        df = pl.read_csv(filename)
        test_df = df.filter(pl.col("split") == "test")

        refs_en = test_df["en_source"].to_list()
        refs_uz = test_df["uz_north_synthetic_ref"].to_list()

        if "nllb" in model_name.lower():
            preds_en = test_df["nllb_uz_to_en"].to_list()
            preds_uz = test_df["nllb_uz_to_uz"].to_list()
        else:
            preds_en = test_df["pred_eng_m2m"].to_list()
            preds_uz = test_df["pred_uzn_m2m"].to_list()

        metrics_en = calculate_metrics(preds_en, refs_en)
        metrics_uz = calculate_metrics(preds_uz, refs_uz)

        with mlflow.start_run(run_name=f"Eval_{model_name}"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("test_samples", test_df.height)

            mlflow.log_metric("en_bleu", metrics_en["bleu"])
            mlflow.log_metric("en_chrf", metrics_en["chrf"])
            mlflow.log_metric("en_meteor", metrics_en["meteor"])

            mlflow.log_metric("uz_bleu", metrics_uz["bleu"])
            mlflow.log_metric("uz_chrf", metrics_uz["chrf"])
            mlflow.log_metric("uz_meteor", metrics_uz["meteor"])

            mlflow.log_artifact(filename)

if __name__ == "__main__":
    run_evaluation()
