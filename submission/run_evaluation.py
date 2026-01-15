import polars as pl
import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from bert_score import score as bert_score
import mlflow
import evaluate
import os
import warnings
warnings.filterwarnings('ignore')


FILES = {
    "NLLB-200_Old": "predictions_model_1_old.csv",
    "NLLB-200": "predictions_model_1.csv",
    "M2M100": "predictions_model_2.csv"
}
EXPERIMENT_NAME = "South_Uzbek_Translation_Benchmark"


class TranslationEvaluator:
    def __init__(self, compute_bert=True):
        self.compute_bert = compute_bert

    def bleu(self, predictions, references):
        refs = [[r] for r in references]
        return BLEU().corpus_score(predictions, refs).score

    def chrf(self, predictions, references):
        refs = [[r] for r in references]
        return CHRF().corpus_score(predictions, refs).score

    def meteor(self, predictions, references):
        meteor = evaluate.load("meteor")
        clean_preds = [p or "" for p in predictions]
        clean_refs = [r or "" for r in references]
        return meteor.compute(predictions=clean_preds, references=clean_refs)["meteor"] * 100

    def bertscore(self, predictions, references, lang='en'):
        device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
        P, R, F1 = bert_score(predictions, references, lang=lang, verbose=False, device=device, batch_size=32)
        return {'precision': P.mean().item()*100, 'recall': R.mean().item()*100, 'f1': F1.mean().item()*100}

    def length_stats(self, predictions, references):
        pred_len = [len(p.split()) for p in predictions]
        ref_len = [len(r.split()) for r in references]
        return {
            'avg_pred_length': np.mean(pred_len),
            'avg_ref_length': np.mean(ref_len),
            'length_ratio': np.mean(pred_len) / np.mean(ref_len),
            'std_pred_length': np.std(pred_len),
            'std_ref_length': np.std(ref_len)
        }

    def all_metrics(self, predictions, references, lang='en'):
        results = {
            'bleu': self.bleu(predictions, references),
            'chrf': self.chrf(predictions, references),
            'meteor': self.meteor(predictions, references)
        }
        results.update(self.length_stats(predictions, references))
        if self.compute_bert:
            bert = self.bertscore(predictions, references, lang=lang)
            results.update({'bertscore_f1': bert['f1'], 'bertscore_precision': bert['precision'], 'bertscore_recall': bert['recall']})
        return results

    def evaluate_nllb_methods(self, df, references, lang='en'):
        methods = {
            'Direct': 'nllb_uz_to_en_direct',
            'Pivot': 'nllb_uz_to_en_pivot',
            'Ensemble': 'nllb_uz_to_en_ensemble',
            'Base': 'nllb_uz_to_en'
        }
        results = {}
        for name, col in methods.items():
            if col in df.columns:
                metrics = self.all_metrics(df[col].to_list(), references, lang=lang)
                results[name] = metrics
                print(f"📊 NLLB {name} Metrics:")
                print(f"  BLEU: {metrics['bleu']:.2f}, chrF: {metrics['chrf']:.2f}, METEOR: {metrics['meteor']:.2f}, AvgLen: {metrics['avg_pred_length']:.1f}")
        return results

    def evaluate_model(self, model_name, filename):
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return None

        df = pl.read_csv(filename)
        # test_df = df.filter(pl.col("split")=="test") if "split" in df.columns else df
        test_df = df
        refs_en = test_df["en_source"].to_list()
        refs_uz = test_df["uz_north_synthetic_ref"].to_list()
        is_nllb = "nllb" in model_name.lower()

        with mlflow.start_run(run_name=f"Eval_{model_name}"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("test_samples", test_df.height)
            mlflow.log_param("prediction_file", filename)
            all_results = {}

            if is_nllb:
                methods_results = self.evaluate_nllb_methods(test_df, refs_en, lang='en')
                all_results['english'] = methods_results
                for method, metrics in methods_results.items():
                    prefix = f"en_{method.lower()}_"
                    for k, v in metrics.items():
                        mlflow.log_metric(f"{prefix}{k}", v)
            else:
                pred_col = "pred_eng_m2m"
                if pred_col in test_df.columns:
                    metrics_en = self.all_metrics(test_df[pred_col].to_list(), refs_en, lang='en')
                    all_results['english'] = {'M2M': metrics_en}
                    print(f"📊 {model_name} Metrics:")
                    print(f"  BLEU: {metrics_en['bleu']:.2f}, chrF: {metrics_en['chrf']:.2f}, METEOR: {metrics_en['meteor']:.2f}, AvgLen: {metrics_en['avg_pred_length']:.1f}")
                    for k, v in metrics_en.items():
                        mlflow.log_metric(f"en_{k}", v)

            pred_uz_col = "nllb_uz_to_uz" if is_nllb else "pred_uzn_m2m"
            if pred_uz_col in test_df.columns:
                metrics_uz = self.all_metrics(test_df[pred_uz_col].to_list(), refs_uz, lang='other')
                all_results['uzbek'] = metrics_uz
                for k, v in metrics_uz.items():
                    mlflow.log_metric(f"uz_{k}", v)
            else:
                all_results['uzbek'] = None

            if os.path.exists(filename):
                mlflow.log_artifact(filename)


def run_evaluation(compute_bert=False):
    evaluator = TranslationEvaluator(compute_bert=compute_bert)
    mlflow.set_experiment(EXPERIMENT_NAME)
    for model_name, filename in FILES.items():
        evaluator.evaluate_model(model_name, filename)
    print("Evaluation completed. Use mlflow ui to view results!")


if __name__ == "__main__":
    run_evaluation()
