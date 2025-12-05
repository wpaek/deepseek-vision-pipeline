# DeepSeek Vision Pipeline

Vision-language data synthesis using DeepSeek-OCR. Single model for OCR, object detection, and captioning to generate question-answer-reasoning datasets.

## Features

- DeepSeek-OCR model for all vision tasks
    - Comparative to github.com/shu4dev/DCVLR
- Smart binning into Text/Object/Commonsense categories
- LLM-based Q/A generation with reasoning
- Batch processing support

## Requirements

- Python 3.9+
- CUDA GPU with 12GB+ VRAM
- `transformers==4.46.3`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_pipeline.py \
    --images-dir ./data/images \
    --output-dir ./output \
    --num-images 100
```

Or via Python API:

```python
from pipeline import DeepSeekPipeline

pipeline = DeepSeekPipeline("config.yaml")
dataset = pipeline.run("./images", "./output", num_images=100)
```

## Configuration

Edit `config.yaml`:

```yaml
model:
  size: "tiny"  # tiny/small/base/large
  
binning:
  text_threshold: 50
  object_threshold: 5
  
synthesis:
  llm_model: "tiiuae/falcon-7b-instruct"
  temperature: 0.7
```

## Output Format

JSONL with Q/A pairs:

```json
{
  "image": "path/to/image.jpg",
  "bin": "B",
  "question": "What objects are visible?",
  "answer": "A car and person",
  "reasoning": "Based on object detection..."
}
```

## License

MIT


