# Whisper Hindi Fine-tuning Project

This project fine-tunes OpenAI's Whisper-small model on a custom Hindi dataset and evaluates performance on the FLEURS Hindi benchmark.

## Results

| Model | Hindi WER |
|-------|-----------|
| Whisper-small (pretrained) | wer_base |
| Whisper-small (fine-tuned) | wer_ft |

> **Note:** Replace `wer_base` and `wer_ft` with actual WER values after running the pipeline.

## Features

- **URL Reconstruction**: Dynamically reconstructs valid GCS URLs from user_id and recording_id
- **Audio Resampling**: Converts audio to 16kHz (required by Whisper)
- **Hindi Text Normalization**: Unicode normalization for consistent WER calculation
- **FLEURS Evaluation**: Benchmarks on Google's FLEURS Hindi test set

## Preprocessing Steps

1. **URL Reconstruction** - Fixes outdated URLs in the dataset
2. **JSON Parsing** - Extracts transcriptions from JSON files
3. **Audio Resampling** - Converts to 16kHz using librosa
4. **Text Normalization** - NFC Unicode normalization for Hindi
5. **Filtering** - Removes invalid samples and audio > 30 seconds
6. **Mel Spectrogram Conversion** - 80 mel frequency bins
7. **Tokenization** - Using WhisperProcessor

## Usage

### Full Pipeline (Training + Evaluation)
```python
from whisper_hindi_finetuning import run_complete_pipeline

results = run_complete_pipeline(csv_path='path/to/your/data.csv')
```

### Evaluation Only
```python
from whisper_hindi_finetuning import evaluate_only

results = evaluate_only(model_path='./whisper-small-hindi-finetuned')
```

## Requirements

```
transformers
datasets
accelerate
evaluate
jiwer
librosa
soundfile
torch
pandas
numpy
tqdm
```

## Configuration

Key parameters in `Config` class:
- `model_name`: openai/whisper-small
- `language`: hi (Hindi)
- `target_sampling_rate`: 16000 Hz
- `num_train_epochs`: 3
- `learning_rate`: 1e-5
- `fp16`: True (mixed precision training)

## License

MIT License