# 🧪 Model Behavior Lab

An automated evaluation framework for benchmarking LLMs across **reasoning accuracy**, **hallucination propensity**, **emotional alignment consistency**, and **code correctness**. Designed for reproducible experiments and deployment-oriented model selection.

This framework runs entirely **offline** using local Ollama models, making it perfect for private evaluation and reproducible research.

---

## What This Project Evaluates

The framework evaluates LLMs across **four key axes**:

1. **Reasoning Accuracy**: Logical reasoning, math problems, syllogisms
   - Uses exact match, numeric tolerance, and optional LLM-judge rubrics

2. **Hallucination Propensity**: Tendency to invent facts, names, or unsupported claims
   - Tests unknown facts, fictional entities, and citation requirements
   - Uses LLM-judge rubrics with failure tags like `hallucinated_entity`, `unsupported_claim`, `overconfident`

3. **Emotional Alignment Consistency**: Empathy, validation, and consistency across prompt paraphrases
   - Uses LLM-judge rubrics per response
   - Measures **consistency** across prompt variant groups (3-5 paraphrases) and across multiple repeats

4. **Code Correctness**: Executable Python code with unit test assertions
   - Extracts code blocks, executes in subprocess with timeout
   - Runs provided test cases and scores partial credit

**Additional Analyses:**
- **Consistency gaps**: Variance and standard deviation across repeats and prompt variants
- **Failure modes**: Systematic vs sporadic failures, categorized by failure tags
- **Behavioral tradeoffs**: Cross-axis correlations (e.g., reasoning vs hallucination)

---

## Requirements

- **Python**: 3.8 or higher
- **Ollama**: Installed and running locally ([ollama.com](https://ollama.com))
- **Models**: Pull the models you want to evaluate (see Setup below)

---

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama Models

Pull the models you want to evaluate:

```bash
ollama pull phi3:3.8b
ollama pull mistral:7b
ollama pull samantha-mistral:7b
```

You can evaluate **any Ollama model**—just pull it and add it to your config (see "Evaluate Any LLM" below).

---

## Quickstart

### 1. Run an Experiment

```bash
python main.py --config configs/run_config.json
```

This will:
- Load the test suite from `prompts/suites/core_suite.json` (40+ tests)
- Run each test across all models in the config with configurable repeats
- Evaluate responses using appropriate methods (exact_match, numeric_tolerance, llm_judge, python_exec)
- Generate results in `data/runs/<run_id>/`

**Options:**
- `--config`: Path to config file (default: `configs/run_config.json`)
- `--run_id`: Custom run ID (default: timestamp-based)
- `--out_dir`: Custom output directory
- `--models`: Comma-separated list to override models (e.g., `--models "llama3:8b,custom-model:latest"`)
- `--suite`: Override suite path (e.g., `--suite prompts/suites/core_suite.json`)

**Example:**
```bash
python main.py --config configs/run_config.json --run_id my_experiment_001
```

### 2. Launch Interactive Dashboard

```bash
streamlit run dashboard.py
```

The dashboard provides:
- **Overview**: Per-model and per-category performance metrics
- **Heatmap**: Model × test score visualization
- **Tradeoffs**: Cross-axis correlations and scatter plots
- **Failure Modes**: Failure tag analysis and worst tests
- **Consistency**: Variance analysis and consistency scores

Select your `run_id` from the sidebar to explore results.

### 3. Generate Model Recommendations

```bash
python select_model.py --run_id <run_id>
```

This creates `data/runs/<run_id>/recommendations.json` with:
- Overall model ranking (weighted scores)
- Best model per category
- Deployment-oriented use case recommendations
- Consistency-based recommendations

**Custom weights:**
```bash
python select_model.py --run_id <run_id> --weights '{"reasoning": 2.0, "hallucination": 1.5}'
```

### 4. Generate Evaluation Report

```bash
python reports/generate_report.py --run_id <run_id>
```

This generates `data/runs/<run_id>/REPORT.md` with:
- Topline metrics
- Tradeoffs narrative
- Failure mode insights
- Consistency gaps
- Deployment recommendations
- Prompt strategy suggestions

---

## Evaluate Any LLM

To evaluate a **different LLM**, you don't need to change any code:

1. **Pull the model in Ollama:**
   ```bash
   ollama pull my-model:latest
   ```

2. **Update the config** (`configs/run_config.json`):
   ```json
   {
     "models": ["my-model:latest", "other-model:7b"],
     ...
   }
   ```
   
   **OR** use the CLI override:
   ```bash
   python main.py --config configs/run_config.json --models "my-model:latest,other-model:7b"
   ```

3. **Run the experiment** as usual—no code changes needed!

---

## Test Suite Schema

Each test case in `prompts/suites/core_suite.json` follows this schema:

```json
{
  "id": "reasoning-001",
  "category": "reasoning|hallucination|emotion|code",
  "prompt": "Your test prompt here...",
  "expected": { ... } or "..." or number,
  "rubric": { ... },
  "eval": {
    "method": "exact_match|numeric_tolerance|contains|llm_judge|python_exec",
    "params": { ... }
  },
  "meta": {
    "difficulty": "easy|med|hard",
    "tags": ["..."],
    "prompt_variant_group": "emotion-set-01" or null
  }
}
```

**Evaluation Methods:**
- `exact_match`: Normalized string comparison
- `numeric_tolerance`: Numeric comparison within tolerance
- `contains`: Checks for required/forbidden terms
- `llm_judge`: LLM-based evaluation with rubric
- `python_exec`: Executes code and runs unit tests

**For code tests**, `expected` must include:
```json
{
  "entrypoint": "function_name",
  "tests": [
    {"input": {"args": [1, 2]}, "output": 3},
    {"input": {"args": [5, 7]}, "output": 12}
  ]
}
```

---

## Outputs and Results

Each experiment run creates `data/runs/<run_id>/` with:

- **`results.jsonl`**: Incremental results (JSONL format, one line per run)
- **`results.json`**: Full results (JSON array)
- **`results.csv`**: CSV export with `output_preview` (truncated to 200 chars)
- **`summary.json`**: Aggregated statistics including:
  - Per-model and per-category averages
  - Consistency metrics (variance, std dev, consistency scores)
  - Failure mode analysis (systematic/sporadic failures, top tags)
  - Tradeoff analysis (correlations, strengths/weaknesses)
- **`recommendations.json`**: Model selection recommendations (if `select_model.py` was run)
- **`REPORT.md`**: Human-readable evaluation report (if `generate_report.py` was run)
- **Aggregated CSVs**: `aggregated_model_category.csv`, `aggregated_model.csv`, `aggregated_category.csv`

**Note**: `data/runs/` is gitignored—your experiment outputs stay local.

---

## Example Snapshot (Committed)

The `examples/sample_run/` folder contains a **tiny, committed example** from a real run:

- **`summary.json`**: Aggregated metrics from run `20260106_025634`
- **`example_results.csv`**: Results for test `reasoning-001` only (2 models, truncated outputs)
- **`recommendations.json`**: Model selection output

This example is **small (~7KB)** and safe to commit. It demonstrates the output format without including large model outputs. Real runs are stored in `data/runs/` and are **not committed**.

To export your own example snapshot:
```bash
python reports/export_example.py --run_id <run_id> --test_id reasoning-001 --out_dir examples/sample_run
```

---

## Prompt Strategy Templates

Based on evaluation results, the framework suggests prompt engineering strategies. Here are the canonical templates:

### Hallucination Prevention

**Template 1: Explicit Uncertainty Acknowledgment**

```
[Your task here]

Important: If you are unsure about any fact, name, date, or detail, explicitly state that you don't know. Do not invent information. If the information is not available or you cannot verify it, say so clearly.
```

**Use when:** Hallucination failure rate > 30%

**Example:**
```
Who is the current president of Mars?

Important: If you are unsure about any fact, name, date, or detail, explicitly state that you don't know. Do not invent information. If the information is not available or you cannot verify it, say so clearly.
```

**Template 2: Citation Required Format**

```
[Your task here]

Format your response as JSON with the following structure:
{
  "answer": "...",
  "claims": ["claim1", "claim2"],
  "evidence": ["source1", "source2"],
  "confidence": "high|medium|low"
}

If you cannot provide evidence for a claim, set confidence to "low" and note uncertainty in the evidence field.
```

**Use when:** Need verifiable outputs with source tracking

### Code Quality

**Template 1: Mental Tracing Instruction**

```
[Your coding task here]

Before writing code, mentally trace through the logic:
1. What are the edge cases?
2. What happens with empty inputs?
3. What happens with invalid inputs?
4. Does the function handle all expected cases?

Write the code, then verify it handles all cases above.
```

**Use when:** Code execution failure rate > 30%

**Template 2: Unit Test First**

```
[Your coding task here]

First, think about what test cases your function should pass:
- Test case 1: [describe]
- Test case 2: [describe]
- Edge case: [describe]

Now write the function to pass all these tests.
```

**Use when:** Partial test pass rate is low

### Consistency Improvement

**Template 1: Deterministic Instructions**

```
[Your task here]

Provide a clear, direct answer. Be consistent in your reasoning approach.
```

**Use when:** Consistency score < 0.7

**Note:** Also ensure `temperature=0.0` and `seed` is set in sampling config.

**Template 2: Structured Output**

```
[Your task here]

Format your response as:
1. [First element]
2. [Second element]
3. [Conclusion]

This structure helps ensure consistent formatting across runs.
```

**Use when:** Variant group consistency is low

### Emotional Alignment

**Template 1: Validation-First Approach**

```
[User's emotional concern]

I understand this is difficult. [Acknowledge the emotion first]

[Then provide helpful response]
```

**Use when:** Emotion alignment scores are low

**Template 2: Non-Judgmental Tone**

```
[User's situation]

There's no right or wrong way to feel about this. [Acknowledge without judgment]

[Provide supportive response]
```

**Use when:** Emotion tests show judgmental language

### Reasoning Accuracy

**Template 1: Step-by-Step Reasoning**

```
[Your reasoning task]

Think through this step by step:
1. [First step]
2. [Second step]
3. [Conclusion]

Show your work.
```

**Use when:** Reasoning accuracy is low

**Template 2: Explicit Answer Format**

```
[Your reasoning task]

Answer with: "Yes" or "No" followed by your reasoning.
```

**Use when:** Exact match failures are high

### Combining Strategies

For best results, combine multiple strategies:

```
[Task]

Important guidelines:
- If uncertain, say so explicitly (hallucination prevention)
- Show your reasoning step-by-step (reasoning accuracy)
- Format as JSON with evidence field (verifiability)
- Consider edge cases (code quality)
```

---

## Repository Structure

```
Model-Behavior-Lab/
├── README.md                    # This file (single source of truth)
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
│
├── main.py                      # Experiment runner
├── analyze_results.py           # CSV export and aggregation
├── select_model.py             # Model ranking and recommendations
├── dashboard.py                 # Interactive Streamlit dashboard
│
├── configs/
│   └── run_config.json         # Experiment configuration
│
├── prompts/
│   └── suites/
│       └── core_suite.json     # Test suite (40+ tests)
│
├── models/
│   └── ollama_runner.py        # Model execution (Ollama CLI)
│
├── analyzers/
│   ├── evaluator.py            # Evaluation methods
│   ├── consistency.py          # Consistency analysis
│   ├── failure_modes.py        # Failure categorization
│   └── tradeoff_analysis.py   # Cross-axis tradeoffs
│
├── reports/
│   ├── generate_report.py      # Auto-generated report
│   └── export_example.py      # Export example snapshot
│
├── examples/
│   └── sample_run/             # Committed example (CSV/JSON only)
│       ├── summary.json
│       ├── example_results.csv
│       └── recommendations.json
│
└── tests/
    ├── test_config_loading.py
    ├── test_consistency.py
    └── test_evaluator.py
```

---

## Troubleshooting

### Ollama Not Found

**Error:** `ollama: command not found`

**Solution:** Install Ollama from [ollama.com](https://ollama.com) and ensure it's in your PATH.

### Model Not Pulled

**Error:** `model not found` or `model does not exist`

**Solution:** Pull the model first:
```bash
ollama pull <model-name>
```

### Timeout Errors

**Error:** `Timeout after X seconds`

**Solution:** Increase `timeout_s` in `configs/run_config.json` or use `--timeout_s` if supported.

### Sampling Flags Not Supported

**Warning:** `sampling_supported: false` in results

**Solution:** Your Ollama version may not support all sampling flags. The framework will fall back to basic execution. Update Ollama or ignore this warning if results are still reproducible.

### Import Errors

**Error:** `ModuleNotFoundError`

**Solution:** Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

---

## Testing

Run the test suite:

```bash
python -m pytest
```

Tests cover:
- JSON schema loading
- Evaluation methods on known outputs
- Code executor safety and timeout behavior
- Consistency analysis functions

---

## License

MIT License

---

## Author

**Aditya Chunduri**  
🔗 [github.com/Chunduri-Aditya](https://github.com/Chunduri-Aditya)  
🎓 M.S. Applied Data Science, USC
