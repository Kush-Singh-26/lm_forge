# Minimal Forge Example

This example demonstrates how to use Forge to orchestrate a GPT-2 fine-tuning job with automated state management on Hugging Face Hub.

## Setup

1. **Install Forge**:
   Make sure you have installed the `lm-forge` package.
   ```bash
   pip install -e .
   ```

2. **Configure Authentication**:
   Ensure your Hugging Face token is set in your environment:
   ```bash
   export HF_TOKEN=your_huggingface_token_here
   ```

3. **Update Config**:
   Edit `forge.yaml` and set `repo_id` to a repository you have write access to (e.g., `your-username/gpt2-forge-checkpoints`).

## Running

### Local Run
Test the integration locally first:
```bash
python train.py
```

### Remote Run (Modal)
To ship this training job to a remote A10G GPU:
```bash
forge run modal --script train.py
```

Forge will automatically:
1. Detect `train.py` and `forge.yaml`.
2. Provision an A10G on Modal.
3. Install necessary dependencies.
4. Sync checkpoints and data state to Hugging Face Hub every 100 steps.
5. Handle resumption automatically if you stop and restart the command.

### Check Status
Monitor your Nomad Training progress:
```bash
forge status
```
Or launch the TUI dashboard:
```bash
forge dash
```
