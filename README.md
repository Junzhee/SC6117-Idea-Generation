# SC6117 Capstone Project: InsightFoundry AI

**InsightFoundry AI** is an evidence-based startup ideation engine. It analyzes unstructured user feedback from social media platforms (Xiaohongshu / RedNote) to identify market gaps, visualize trends, and automatically generate investment-ready startup pitches using the **DeepSeek LLM**.

---

### Features

- **Data Ingestion**
  - Scrapes and loads user comments and note contents related to specific products (e.g., robot vacuums).

- **Market Analysis**
  - **Pain Point Detection**: Uses LLMs to extract negative feedback and critical product flaws.
  - **Hot Thread Analysis**: Summarizes controversial or high-engagement discussions.

- **Visual Intelligence**
  - Generates daily comment volume trends and word clouds (using Jieba for Chinese segmentation).

- **Founder AI Assistant**
  - Transforms analytical insights into a structured six-part startup pitch deck (Problem, Solution, Differentiation, etc.).

---

### Project Structure

```text
├── data/                    # CSV storage for raw comments and contents
├── output/                  # Generated reports (JSON) and charts (PNG)
├── src/                     # Core logic modules
│   ├── analyzer.py          # LLM analysis & chart generation
│   ├── data_loader.py       # Data cleaning & timestamp normalization
│   ├── generator.py         # DeepSeek prompt engineering & pitch generation
│   └── scraper.py           # XhsDataService wrapper for data fetching
├── app.py                   # Streamlit web dashboard entry point
├── requirements.txt         # Python dependencies
└── .env                     # API credentials
```

---

### Setup & Installation

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd SC6117-Idea-Generation
```

#### 2. Install Dependencies

Ensure you have **Python 3.1+** installed.

```bash
pip install -r requirements.txt
```

Key libraries used:

- streamlit
- pandas
- openai
- matplotlib
- seaborn
- wordcloud
- jieba
- python-dotenv

#### 3. Environment Configuration

Create a `.env` file in the root directory and add your API keys:

```ini
# Required for logic and analysis
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Required for data scraping (optional if using local data)
CRAWLING_KEY=your_crawling_service_key
```

---

### Usage

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

#### Workflow

- **Sidebar**
  - Search for a product keyword or select existing CSV datasets from the `data/` folder.

- **Market Insights Tab**
  - Click **Start Analyzer** to process the data.
  - Generates `analyzer.json` and visualization charts.

- **Founder AI Assistant Tab**
  - After analysis is complete, click **Run AI Ideation** to generate a startup pitch based on identified user pain points.

---

### Data Format

The system expects **two CSV files per dataset** in the `data/` folder. Please refer to the sample data.
