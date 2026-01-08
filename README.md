# SC6117-Idea-Generation


# AI Startup Ideation System

This project analyzes user feedback from Xiaohongshu (RedNote) regarding Robot Vacuums to generate data-driven startup ideas using the DeepSeek LLM.

## Project Structure

- `data/`: Sample CSV files.
- `src/`: Backend logic modules.
  - `data_loader.py`: Handles CSV ingestion.
  - `analyzer.py`: Filters comments for pain points.
  - `generator.py`: Connects to DeepSeek API.
- `app.py`: The Streamlit frontend dashboard.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt