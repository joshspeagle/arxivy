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

# Edit configurations in config/

# Test workflow components
python src/llm_utils.py  # check LLM setup
python src/score_utils.py  # check scoring setup
python src/selection_utils.py  # check selection setup
python src/pdf_utils.py  # check PDF download setup
python src/extract_utils.py  # check PDF text extraction setup

# Run workflow (step-by-step)
python src/fetch_papers.py  # grab abstracts
python src/score_papers.py  # score abstracts
python src/download_papers.py  # subselect and download PDFs
python src/extract_papers.py  # extract text from PDFs
...

# Run workflow (integrated)
...
```

## Configuration

All configurations can be found in `config/config.yaml`. This is organized by options associated with various portions of the workflow described above.

## LLM API

Configurations, API settings, and aliases for LLMs can be found in `config/llm.yaml`. Currently, support is provided for external models through OpenAI, Anthropic, and Google APIs. Support for local models is provided through Ollama and LM Studio, as well as custom local APIs in either Ollama or OpenAI format. See the `config/llm.yaml` file for examples of how to specify these models.

## Project Status

- ✅ Paper fetching (Step 1)
- ✅ LLM scoring (Step 2)
- ✅ PDF processing (Step 3)
- ✅ Summarization (Step 4)
- ⏳ Final scoring (Step 5)
- ⏳ Report generation (Step 6)
- ⏳ Optional: Audio synthesis (Step 7)

## Requirements

- Python 3.8+
- API keys for chosen LLM services
- ~2GB storage for paper caching

## License

MIT License

## Acknowledgements

Made in collaboration with Claude Sonnet 4.
