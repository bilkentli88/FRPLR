# Security

Never commit Hugging Face tokens, Colab secrets, API keys, or other credentials.

Use Colab's secret store for `HF_TOKEN`; never place a token in a notebook cell,
configuration file, output row, shell history, or Git remote URL. Before every
release, scan both tracked files and notebook metadata for credentials, personal
paths, email addresses, and identifying comments.

Model weights are intentionally excluded and must be obtained from the official
model repositories under their respective licenses.
