"""
evaluate_translations.py - Оценка качества переводов
"""
import polars as pl
import numpy as np
from sacrebleu.metrics import BLEU, CHRF, TER
from bert_score import score as bert_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# CONFIG
PREDICTIONS_FILE = "predictions_model_1.csv"
DATASET_FILE = "dataset.csv"

def calculate_bleu(predictions, references):
    """Вычисляет BLEU score"""
    bleu = BLEU()
    refs = [[ref] for ref in references]
    score = bleu.corpus_score(predictions, refs)
    return score.score

def calculate_chrf(predictions, references):
    """Вычисляет chrF score (character n-gram F-score)"""
    chrf = CHRF()
    refs = [[ref] for ref in references]
    score = chrf.corpus_score(predictions, refs)
    return score.score

def calculate_ter(predictions, references):
    """Вычисляет TER (Translation Error Rate) - чем меньше, тем лучше"""
    ter = TER()
    refs = [[ref] for ref in references]
    score = ter.corpus_score(predictions, refs)
    return score.score

def calculate_bertscore(predictions, references, lang='en'):
    """Вычисляет BERTScore (семантическое сходство)"""
    print(f"  Computing BERTScore for {lang}...")
    P, R, F1 = bert_score(
        predictions, 
        references, 
        lang=lang, 
        verbose=False,
        device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
        batch_size=32
    )
    return {
        'precision': P.mean().item() * 100,
        'recall': R.mean().item() * 100,
        'f1': F1.mean().item() * 100
    }

def calculate_length_stats(predictions, references):
    """Статистика по длинам"""
    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]
    
    return {
        'avg_pred_length': np.mean(pred_lengths),
        'avg_ref_length': np.mean(ref_lengths),
        'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths),
        'std_pred_length': np.std(pred_lengths),
        'std_ref_length': np.std(ref_lengths)
    }

def evaluate_task1_english():
    """Оценка Task 1: South Uzbek -> English"""
    print("="*70)
    print("TASK 1: EVALUATION - South Uzbek → English")
    print("="*70)
    
    # Загружаем данные
    df = pl.read_csv(PREDICTIONS_FILE)
    
    references = df["en_target"].to_list()
    
    # Проверяем наличие колонок
    methods = {
        'Direct': 'nllb_uz_to_en_direct',
        'Pivot': 'nllb_uz_to_en_pivot',
        'Ensemble': 'nllb_uz_to_en_ensemble'
    }
    
    results = {}
    
    for method_name, col_name in methods.items():
        if col_name not in df.columns:
            print(f"⚠️ Column {col_name} not found, skipping...")
            continue
            
        print(f"\n{'='*70}")
        print(f"Method: {method_name}")
        print(f"{'='*70}")
        
        predictions = df[col_name].to_list()
        
        # BLEU
        print("📊 Computing BLEU...")
        bleu_score = calculate_bleu(predictions, references)
        
        # chrF
        print("📊 Computing chrF...")
        chrf_score = calculate_chrf(predictions, references)
        
        # TER
        print("📊 Computing TER...")
        ter_score = calculate_ter(predictions, references)
        
        # BERTScore
        bert_scores = calculate_bertscore(predictions, references, lang='en')
        
        # Length stats
        length_stats = calculate_length_stats(predictions, references)
        
        results[method_name] = {
            'BLEU': bleu_score,
            'chrF': chrf_score,
            'TER': ter_score,
            'BERTScore-F1': bert_scores['f1'],
            'BERTScore-P': bert_scores['precision'],
            'BERTScore-R': bert_scores['recall'],
            **length_stats
        }
        
        # Вывод результатов
        print(f"\n📈 Results for {method_name}:")
        print(f"  BLEU:           {bleu_score:.2f}")
        print(f"  chrF:           {chrf_score:.2f}")
        print(f"  TER:            {ter_score:.2f} (lower is better)")
        print(f"  BERTScore-F1:   {bert_scores['f1']:.2f}")
        print(f"  BERTScore-P:    {bert_scores['precision']:.2f}")
        print(f"  BERTScore-R:    {bert_scores['recall']:.2f}")
        print(f"  Avg Length:     {length_stats['avg_pred_length']:.1f} words (ref: {length_stats['avg_ref_length']:.1f})")
        print(f"  Length Ratio:   {length_stats['length_ratio']:.2f}")
    
    # Сравнительная таблица
    print(f"\n{'='*70}")
    print("COMPARISON TABLE - Task 1 (English Translation)")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'BLEU':>8} {'chrF':>8} {'TER':>8} {'BERTScore':>10} {'Length':>8}")
    print("-"*70)
    for method_name, scores in results.items():
        print(f"{method_name:<15} {scores['BLEU']:>8.2f} {scores['chrF']:>8.2f} "
              f"{scores['TER']:>8.2f} {scores['BERTScore-F1']:>10.2f} {scores['avg_pred_length']:>8.1f}")
    
    return results

def evaluate_task2_uzbek():
    """Оценка Task 2: South Uzbek -> Standard Uzbek (Normalization)"""
    print(f"\n{'='*70}")
    print("TASK 2: EVALUATION - South Uzbek → Standard Uzbek (Normalization)")
    print(f"{'='*70}")
    
    # Загружаем данные
    df = pl.read_csv(PREDICTIONS_FILE)
    
    references = df["uz_standard_target"].to_list()
    predictions = df["nllb_uz_to_uz"].to_list()
    
    # BLEU
    print("📊 Computing BLEU...")
    bleu_score = calculate_bleu(predictions, references)
    
    # chrF (особенно важен для морфологически богатых языков)
    print("📊 Computing chrF...")
    chrf_score = calculate_chrf(predictions, references)
    
    # TER
    print("📊 Computing TER...")
    ter_score = calculate_ter(predictions, references)
    
    # BERTScore (используем multilingual model)
    # Для узбекского используем 'other' или 'en' как fallback
    print("📊 Computing BERTScore (multilingual)...")
    bert_scores = calculate_bertscore(predictions, references, lang='en')  # Fallback
    
    # Length stats
    length_stats = calculate_length_stats(predictions, references)
    
    # Вывод результатов
    print(f"\n📈 Results for Normalization (South → Standard Uzbek):")
    print(f"  BLEU:           {bleu_score:.2f}")
    print(f"  chrF:           {chrf_score:.2f}")
    print(f"  TER:            {ter_score:.2f} (lower is better)")
    print(f"  BERTScore-F1:   {bert_scores['f1']:.2f}")
    print(f"  Avg Length:     {length_stats['avg_pred_length']:.1f} words (ref: {length_stats['avg_ref_length']:.1f})")
    print(f"  Length Ratio:   {length_stats['length_ratio']:.2f}")
    
    return {
        'BLEU': bleu_score,
        'chrF': chrf_score,
        'TER': ter_score,
        'BERTScore-F1': bert_scores['f1'],
        **length_stats
    }

def show_examples(n=5):
    """Показывает примеры переводов"""
    print(f"\n{'='*70}")
    print(f"SAMPLE TRANSLATIONS (first {n} examples)")
    print(f"{'='*70}")
    
    df = pl.read_csv(PREDICTIONS_FILE)
    
    for i in range(min(n, len(df))):
        print(f"\n{'─'*70}")
        print(f"Example {i+1}:")
        print(f"{'─'*70}")
        print(f"Source (South Uz): {df['uz_south_source'][i]}")
        print(f"\n--- Task 1: English Translation ---")
        print(f"Reference:         {df['en_target'][i]}")
        if 'nllb_uz_to_en_direct' in df.columns:
            print(f"Direct:            {df['nllb_uz_to_en_direct'][i]}")
        if 'nllb_uz_to_en_pivot' in df.columns:
            print(f"Pivot:             {df['nllb_uz_to_en_pivot'][i]}")
        if 'nllb_uz_to_en_ensemble' in df.columns:
            print(f"Ensemble:          {df['nllb_uz_to_en_ensemble'][i]}")
        
        print(f"\n--- Task 2: Uzbek Normalization ---")
        print(f"Reference (Std):   {df['uz_standard_target'][i]}")
        if 'nllb_uz_to_uz' in df.columns:
            print(f"Normalized:        {df['nllb_uz_to_uz'][i]}")

def main():
    print("🚀 Starting Evaluation Pipeline...")
    print(f"Loading predictions from: {PREDICTIONS_FILE}")
    print(f"Loading references from: {DATASET_FILE}\n")
    
    # Task 1: English Translation
    task1_results = evaluate_task1_english()
    
    # Task 2: Uzbek Normalization
    task2_results = evaluate_task2_uzbek()
    
    # Show examples
    show_examples(n=5)
    
    # Final summary
    print(f"\n{'='*70}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*70}")
    print("\nTask 1 (English Translation) - Best Method:")
    if task1_results:
        best_method = max(task1_results.items(), key=lambda x: x[1]['BLEU'])
        print(f"  🏆 {best_method[0]}: BLEU={best_method[1]['BLEU']:.2f}, BERTScore={best_method[1]['BERTScore-F1']:.2f}")
    
    print("\nTask 2 (Uzbek Normalization):")
    print(f"  BLEU={task2_results['BLEU']:.2f}, chrF={task2_results['chrF']:.2f}")
    
    print(f"\n{'='*70}")
    print("✅ Evaluation Complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()