# 🧪 Model Behavior Lab

A lightweight evaluation framework for analyzing the behavior of locally running LLMs using [Ollama](https://ollama.com). This project runs a set of prompts across different models and scores them on logic, emotion, hallucination, and code quality.

---

## 📌 Features

- 🔁 Run multiple prompts across multiple local models
- 🧠 Score responses using custom criteria
- 📊 Visualize results using bar plots and heatmaps
- ⚡️ Uses models via `ollama` — fast and offline
- ✅ Modular scoring and prompt evaluation system

---

## 🗂 Project Structure

```
model-behavior-lab/
│
├── main.py                      # Runs all prompts through selected models
├── analyze_results.py          # Converts JSON results into plots
├── score_results.py            # (Optional) Re-scores results.json
│
├── prompts/
│   └── sample_set.json         # List of prompts with ids and task text
│
├── data/
│   └── results.json            # Output after running all prompts
│
├── models/
│   └── ollama_runner.py        # Function to query Ollama models
│
├── analyzers/
│   ├── scoring.py              # Scoring logic for different task types
│   └── apply_scoring.py        # Routing logic to apply correct scoring
│
└── visualize.ipynb             # Jupyter Notebook for analysis
```

---

## 🛠 Installation

1. **Clone the repo**

```bash
git clone https://github.com/Chunduri-Aditya/Model-Behavior-Lab.git
cd Model-Behavior-Lab
```

2. **Set up environment**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Install Ollama and required models**

Make sure you have [Ollama](https://ollama.com) installed and then pull the models:

```bash
ollama pull phi3:3.8b
ollama pull mistral:7b
ollama pull samantha-mistral:7b
```

---

## ▶️ Usage

### 1. Run the prompt evaluation:

```bash
python main.py
```

This will run all prompts through all models and generate `data/results.json`.

### 2. Visualize results:

You can either use:

- The Python script:

  ```bash
  python analyze_results.py
  ```

- Or the notebook:

  ```bash
  jupyter notebook visualize.ipynb
  ```

---

## 🧪 Scoring Criteria

| Type         | Scoring Logic                                                                 |
|--------------|--------------------------------------------------------------------------------|
| Logic        | 1 if output includes "yes" or "no"                                             |
| Emotion      | 1 if output contains emotion-related words like "hope", "feel", etc.          |
| Hallucination| 1 if output correctly says nobody is president of Mars                        |
| Code         | 1 if output contains both `def` and `return`                                  |

Scoring logic can be customized via `analyzers/scoring.py`.

---

## 📈 Sample Output

(You can generate plots by running the notebook or script.)

---

## 🧠 Models Used

- `phi3:3.8b`
- `mistral:7b`
- `samantha-mistral:7b`

You can modify `main.py` to add or remove models.

---

## 🙌 Future Ideas

- Add GPT-4 or Claude via API (optionally)
- Extend scoring types: safety, reasoning depth, creativity
- Add prompt category tagging
- Benchmark speed vs quality

---

## 📄 License

MIT License

---

## 👤 Author

**Aditya Chunduri**  
🔗 [github.com/Chunduri-Aditya](https://github.com/Chunduri-Aditya)  
🎓 M.S. Applied Data Science, USC  
💡 Building AI projects with love & caffeine ☕️
