- [X] Create a script that uses [steering-vectors](https://steering-vectors.github.io/steering-vectors/) to derive steering vector, starting with basic difference in means between two groups of data (e.g., high vs. low Danish language quality)
- [X] As a starting dataset to derive steering vectors, make a loader for the file "data/Linguistic_quality_preference_20260326.tsv"
- [X] Models: Initially https://huggingface.co/google/gemma-3-4b-it, but then also enable https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503
- [X] Then apply steering vector
- [ ] Evals: https://github.com/danish-foundation-models/dfm-evals

Additional instructions:
- Keep it simple. A single script is fine for now.
- Dependencies should be managed by `uv`
