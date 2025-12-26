# TIMELY-Bench Dataset

## Overview

TIMELY-Bench is a benchmark dataset for time-aligned fusion of clinical time-series and notes in MIMIC-IV.

## Dataset Structure

Each episode is a JSON object with the following structure:

```json
{
  "episode_id": "mimic4_12345",
  "patient_id": 10001,
  "stay_id": 12345,
  "hadm_id": 20001,
  
  "conditions": ["sepsis", "aki"],
  "n_conditions": 2,
  
  "condition_graph": {
    "nodes": [...],
    "edges": [...]
  },
  
  "timeseries": {
    "window_hours": 24,
    "hours_available": [0, 1, 2, ...],
    "features": {
      "heart_rate": [{"hour": 0, "value": 85}, ...],
      "creatinine": [{"hour": 2, "value": 1.2}, ...],
      ...
    }
  },
  
  "notes_spans": [
    {
      "note_id": "note_001",
      "hour_offset": 3.5,
      "category": "Radiology",
      "text": "Chest X-ray shows..."
    },
    ...
  ],
  
  "pattern_detections": {
    "tachycardia": {
      "n_detections": 5,
      "detections": [{"hour": 2, "value": 105, "severity": "mild"}, ...],
      "text_alignments": [{"relevant_text": "Patient tachycardic...", "annotation": null}, ...]
    },
    ...
  },
  
  "physiology_templates": {
    "tachycardia": {
      "disease": "Sepsis",
      "clinical_standard": "Sepsis-3",
      "threshold": 90,
      "direction": "above",
      ...
    },
    ...
  },
  
  "pattern_annotations": {
    "total_alignments": 15,
    "annotated": 0,
    "supportive": 0,
    "contradictory": 0,
    "ambiguous": 0,
    "unrelated": 0
  },
  
  "labels": {
    "mortality": 0,
    "prolonged_los_7d": 1,
    "readmission_30d": 0,
    "los_hours": 168.5,
    "aki_stage_max": 2
  },
  
  "metadata": {
    "window_hours": 24,
    "data_source": "MIMIC-IV v3.1",
    "created_at": "2025-01-15T10:30:00"
  }
}
```

## Statistics

| Metric | Value |
|--------|-------|
| Total Episodes | 74829 |
| With Notes | 45435 |
| With Patterns | 74812 |
| With Alignments | 27277 |
| Total Patterns | 678373 |
| Total Alignments | 137666 |

## Files

- `timely_bench_episodes.jsonl` - Complete dataset (one episode per line)
- `sample_episodes.json` - Sample of 100 episodes for inspection
- `dataset_stats.json` - Dataset statistics
- `condition_graph.json` - Disease relationship graph

## Usage

```python
import json

# Load episodes
episodes = []
with open('timely_bench_episodes.jsonl', 'r') as f:
    for line in f:
        episodes.append(json.loads(line))

# Filter by condition
sepsis_episodes = [e for e in episodes if 'sepsis' in e['conditions']]

# Get patterns for an episode
for pattern_name, pattern_data in episodes[0]['pattern_detections'].items():
    print(f"{pattern_name}: {pattern_data['n_detections']} detections")
```

## License

This dataset is derived from MIMIC-IV and requires PhysioNet credentialing.

## Citation

```bibtex
@misc{timely-bench,
  title={TIMELY-Bench: A Benchmark for Time-Aligned Fusion of Clinical Time-Series and Notes},
  author={Wang, Haoyu},
  year={2025},
  institution={King's College London}
}
```
