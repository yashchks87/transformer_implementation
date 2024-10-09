# Transformer Model Training

This project demonstrates how to train a Transformer model using PyTorch. The code includes data loading, model training, and loss computation.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/transformer-training.git
    cd transformer-training
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your dataset and ensure it is compatible with the DataLoader.
2. Modify the `main.ipynb` file to include your dataset and model parameters.
3. Run the Jupyter Notebook to start training the model.

## Code Explanation

### Training Loop

The training loop is defined in the `main.ipynb` file. Here is an excerpt of the main training loop:

```python
model.to(device)
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    
    for batch in tqdm(dataloader):
        src, tgt = batch
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt[:, :-1])
        
        # Reshape output and target for loss computation
        output = output.permute(0, 2, 1)  # [batch_size, vocab_size, seq_len]
        tgt = tgt[:, 1:]  # Shift target to the right
        
        # Compute loss
        loss = criterion(output, tgt)
        total_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')