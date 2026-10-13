# Federated Learning for Personalized LLM Agents

This repository accompanies the accepted research paper on federated learning and personalized LLM agents. It contains the experimental implementation, dataset, and evaluation notebook.

## Repository structure

```text
.
├── preference-learning.ipynb   # Model training and experimentation
├── evaluation.ipynb            # Evaluation and testing workflows
├── prompt_template.py          # Prompt templates and structured output schemas
├── data/
│   └── preference_cleaned.csv  # Preference and question dataset
└── README.md
```

## Requirements and setup

The notebooks record Python 3.11 environments. Install the imported libraries, JupyterLab for local notebook access, and Accelerate for automatic model placement:

```bash
python -m pip install numpy pandas torch transformers peft accelerate sentence-transformers datasets pydantic langchain-core langchain-groq nltk networkx matplotlib plotly librosa praat-parselmouth tqdm scikit-learn python-dotenv pinecone jupyterlab
python -m pip install git+https://github.com/openai/whisper.git
```

The Whisper installation requires Git; audio transcription also requires FFmpeg. Dependency versions are not pinned in this release.

Configure credentials privately; never commit keys or a populated `.env` file:

| Setting | Used by |
| --- | --- |
| `GROQ_API_KEY`, `GROQ_API_KEY_2` | Training notebook's two Groq clients |
| `GROQ_API_KEY` | Evaluation notebook's Groq client |
| `LLM_MODEL` | Groq model identifier in both notebooks |
| `HUGGING_FACE_ACCESS_TOKEN` | Hugging Face model access in both notebooks |
| `PINECONE_API_KEY` | Pinecone evaluation sections |

Training reads its settings through Kaggle Secrets (`kaggle_secrets.UserSecretsClient`), which requires the Kaggle notebook environment. Evaluation reads environment variables and calls `load_dotenv()`. Pinecone evaluation expects an existing index named `preference-prompts` compatible with the BERT embeddings; those sections delete index contents during evaluation, so use a dedicated index.

Model downloads require internet access and permission to access `meta-llama/Llama-2-7b-chat-hf` for training and `meta-llama/Llama-3.2-1B-Instruct` for LLM evaluation. Supporting models are `bert-base-uncased`, `all-MiniLM-L6-v2`, `cardiffnlp/twitter-roberta-base-sentiment-latest`, and Whisper `base`.

## Execution

**Training and experimentation:** Open `preference-learning.ipynb` in Kaggle, enable internet access, configure the Secrets listed above, and attach the CSV at `/kaggle/input/pref-data/preference_cleaned.csv` (the Runner cell's configured path). Review the model-loading device and memory settings for your hardware, then execute cells in order through Runner. Adapter and result paths use `./kaggle/working/`; check these paths for your environment. The Runner's `federate` setting selects aggregation instead of batch processing.

**Evaluation:** From the repository root, launch the separate notebook:

```bash
jupyter lab evaluation.ipynb
```

Run the desired evaluation section and its setup cells. Dataset-based sections use `./data/preference_cleaned.csv`, while the cross-validation section still uses the Kaggle input path. Graph evaluations require saved graphs under `./graphs`; result-analysis sections require previously generated CSVs. Check input paths and create required output directories before running.

Several sections import local modules absent from this release: `CPL`, `adapter_management`, `query_router`, `Implicit`, `utils.LLM_judge`, and `utils.federation`. These must be supplied before those sections can run. Full notebook execution has not been validated for this release.

The provided implementation may contain experimental modifications or hyperparameter settings that differ from those used in the published study. Please refer to the original paper for the reported experimental configurations and hyperparameters. Exact reproduction of the published results using this release has not been independently verified.
