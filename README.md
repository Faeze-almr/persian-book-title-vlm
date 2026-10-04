# Persian Book Title Extraction with Qwen2.5-VL

This project fine-tunes `Qwen/Qwen2.5-VL-3B-Instruct` to extract the complete Persian book title from a cover image while excluding author, translator, publisher, and edition text.

## Experiment

- Dataset: `shenasa/bookroom-persian-book-covers-and-titles`
- Training set: 1,000 validated samples
- Test set: 200 held-out validated samples
- Method: 4-bit QLoRA with NF4 and double quantization
- Training: 2 epochs, 250 optimizer steps, batch size 1, gradient accumulation 8
- Environment: Kaggle Tesla T4
- Evaluation: Raw/Normalized Exact Match, Levenshtein Similarity, and Word F1

## Results

| Metric | Baseline | Fine-tuned | Change |
| --- | ---: | ---: | ---: |
| Raw Exact Match | 4.5% | 14.0% | +9.5 pp |
| Normalized Exact Match | 5.0% | 14.0% | +9.0 pp |
| Levenshtein Similarity | 0.480 | 0.602 | +0.122 |
| Word F1 | 0.340 | 0.446 | +0.105 |

Fine-tuning improved every recorded metric. Multi-line titles and separating the title from other cover text remained the main error sources.

## Repository contents

- `persian-book-title-vlm.ipynb`: complete data preparation, baseline evaluation, QLoRA fine-tuning, and final evaluation
- `Faeze_Ali-Mirzaee_Report.pdf`: Persian technical report
- `results/`: reproducible manifests, metrics, loss history, plots, and worst-case analysis

## Reproduction notes

The notebook was designed for a Kaggle GPU session. Core package versions are listed in `requirements.txt`. Dataset samples and model weights are downloaded from their upstream sources and are not committed to this repository.

The evaluation split is separated from training and recorded through `results/selection_manifest.json`. Generation is deterministic, with sampling disabled and a maximum of 64 new tokens.

## Limitations

- The experiment uses a modest 1,000/200 sample split and no independent validation set.
- Exact-match metrics penalize small orthographic and spacing differences heavily.
- No data augmentation or early stopping was used.
- Results are specific to the retained split and configuration.
