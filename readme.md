# InsightForge 🔍📊

InsightForge is an intelligent, conversational data analysis and visualization platform built with Flask, Pandas, and Google Gemini LLM. It allows users to upload structured datasets and explore, query, and visualize their data using natural language.

---

## ✨ Features

- **Intuitive Dataset Upload**: Support for CSV and Excel (`.xlsx`, `.xls`) files with automated schema detection and data previews.
- **Natural Language Data Querying**: Ask questions in plain English; queries are parsed via LLM into structured Pandas data operations.
- **Interactive Chart Recommendations**: Automatically suggests and renders dynamic charts (bar, line, scatter, pie, etc.) matching your query results.
- **Conversational Memory**: Multi-turn dialogue context so you can refine queries iteratively.
- **Custom Query Engine**: Safe execution pipeline handling filtering, grouping, aggregation, sorting, and top-N queries.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- A Google Gemini API key ([Get one here](https://aistudio.google.com/))

### 2. Clone the Repository
```bash
git clone https://github.com/Kavin86080/InsightForge.git
cd InsightForge
```

### 3. Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirement.txt
```

### 5. Configure API Key
Create a `.env` file in the root directory (or copy from `.env.example`):
```bash
cp .env.example .env
```
And add your Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 6. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`.

---

## 📁 Project Structure

```text
InsightForge/
├── app.py                # Main Flask application and API endpoints
├── llm_parser.py         # Gemini LLM prompt engineering and parsing logic
├── query_engine.py       # Safe Pandas query execution engine
├── requirement.txt       # Python package dependencies
├── sales_data.csv        # Sample dataset for testing
├── static/               # Frontend assets (CSS, JS, icons)
├── templates/            # HTML templates (Jinja2)
├── uploads/              # Uploaded user datasets (excluded from Git)
├── .env.example          # Sample environment configuration
└── README.md             # Project documentation
```

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
