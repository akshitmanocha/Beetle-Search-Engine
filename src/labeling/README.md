# Labeling Tools

This directory contains tools for generating weak labels and manually labeling blog/not-blog samples.

## Files

- `weak_labeler.py` - Generates heuristic-based weak labels
- `label_ui.py` - Gradio UI for manual labeling
- `requirements.txt` - Python dependencies for labeling tools

## Usage

### 1. Generate Weak Labels

```bash
python src/labeling/weak_labeler.py
```

This creates `data/labels/weak.csv` with heuristic-based predictions for all parsed samples.

### 2. Manual Labeling

Install dependencies:
```bash
pip install -r src/labeling/requirements.txt
```

Launch the labeling UI:
```bash
python src/labeling/label_ui.py
```

The UI will:
- Prioritize uncertain samples first
- Show full content, metadata, and weak label predictions
- Save labels to `data/labels/manual.csv` (tracked by DVC)
- Display progress and statistics

### 3. Access Labels

Both label files are tracked by DVC:
- `data/labels/weak.csv.dvc`
- `data/labels/manual.csv.dvc`

To pull the latest labels:
```bash
dvc pull data/labels/weak.csv data/labels/manual.csv
```

## Labeling Guidelines

**Blog posts** typically have:
- Personal or opinionated writing style
- Author attribution
- Publication date
- 500-5000 words
- Clear narrative structure
- First-person pronouns

**Not blogs** include:
- Academic papers (with arXiv/DOI)
- API documentation
- Product pages
- Landing pages
- Reference documentation
- Very short content (<200 words)

When uncertain, check:
1. URL structure (`/blog/`, `/posts/`, dates)
2. Author bio presence
3. Writing tone (personal vs. technical)
4. Content purpose (narrative vs. reference)
