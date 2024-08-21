# momentum_former: Simple Encoder-Only Momentum Transformer

This repository contains the implementation of a transformer-based model for futures prediction using momentum-based features. The model is trained and evaluated across multiple years, and the performance is measured using metrics like Sharpe Ratio, cumulative return, and overall volatility.

## Project Structure

- `datasets/`: Contains the custom dataset class for loading and processing futures time series data.
- `models/`: Contains the transformer model and the custom transformer layer used in the model.
- `losses/`: Contains the custom loss function for optimizing the Sharpe ratio.
- `utils/`: Contains utility functions for weight decay and trading strategy calculations.
- `train.py`: The main script for training and evaluating the model across different years.
- `requirements.txt`: Lists the dependencies required to run the project.
- `README.md`: This file, providing an overview of the project.

## Getting Started

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/simple-encoder-momentum-transformer.git
   cd simple-encoder-momentum-transformer
   ```

2. Install the dependencies:
   ```sh
   pip install -r requirements.txt
   ```

