# Setting Up Weights & Biases (wandb) for Cartoonify

This guide will help you install and configure Weights & Biases (wandb) for experiment tracking in the Cartoonify project.

## Installation

1. Install wandb using pip:
   ```bash
   pip install wandb
   ```

2. Create a free account on [Weights & Biases](https://wandb.ai/site) if you don't have one already.

3. Get your API key from the [wandb settings page](https://wandb.ai/settings).

## Configuration

### Option 1: Login via Command Line

Run the following command and paste your API key when prompted:
```bash
wandb login
```

### Option 2: Use Environment Variable

Add your API key as an environment variable:
```bash
export WANDB_API_KEY=your_api_key_here
```

### Option 3: Pass Key Directly to the Training Script

Use the `--wandbkey` argument when running the training script:
```bash
python train.py --wandbkey your_api_key_here --projectname Cartoonify --wandbentity your_username
```

## Usage with Cartoonify

When running the training script, you can customize wandb parameters:

```bash
python train.py \
  --wandbkey your_api_key_here \
  --projectname YourProjectName \
  --wandbentity your_username \
  --batch_size 16 \
  --epoch 10
```

Where:
- `--wandbkey`: Your wandb API key
- `--projectname`: The name of your wandb project (default is "Cartoonify")
- `--wandbentity`: Your wandb username or team name

## Viewing Results

1. After starting training, the console will display a URL to your wandb dashboard.
2. Visit the URL to see real-time updates on:
   - Training metrics (Generator and Discriminator losses)
   - Generated images throughout training
   - Model checkpoints

## Resuming Training

To resume training with previously saved weights:

```bash
python train.py --wandbkey your_api_key_here --load_checkpoints True
```

## Troubleshooting

- If you encounter issues with wandb, you can run without it by omitting the `--wandbkey` parameter.
- For network connectivity issues, try setting offline mode: `export WANDB_MODE=offline`
- For more help, visit the [wandb documentation](https://docs.wandb.ai/).
