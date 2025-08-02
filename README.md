# arxivy

*Helping to keep up with the arXiv.*

This repository contains various scripts for discovering, scoring, and summarizing recent arXiv papers tailored to a particular user's research interests. It is a work in progress.

## Overview

This workflow processes arXiv papers through several stages:
1. **Fetch** recent papers by category
2. **Score** for interest/impact (via LLM)
3. **Download** top-scoring PDFs
4. **Summarize** paper contents (via LLM)
5. **Re-score** based on summaries (via LLM)
6. **Synthesize** comprehensive digest of top-scoring papers

There is also an option to convert the digest into a podcast-style audio file.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/joshspeagle/arxivy
cd arxivy
pip install -r requirements.txt

# Configure (edit config/config.yaml)
cp config/config.yaml.example config/config.yaml

# Run workflow
python ...
```

## Configuration

All configurations can be found in `config/config.yaml`. This is organized by options associated with various portions of the workflow described above.

## LLM API

Configurations, API settings, and aliases for LLMs can be found in `config/llm.yaml`.

## Project Status

- ✅ Paper fetching (Step 1)
- ⏳ LLM scoring (Step 2)
- ⏳ PDF processing (Step 3)
- ⏳ Summarization (Step 4)
- ⏳ Final scoring (Step 5)
- ⏳ Report generation (Step 6)
- ⏳ Optional: Audio synthesis (Step 7)

## Requirements

- Python 3.8+
- API keys for chosen LLM services
- ~2GB storage for paper caching

## License

MIT License