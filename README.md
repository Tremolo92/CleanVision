# CleanVision

CleanVision is a project that uses CycleGAN to generate synthetic images for raising awareness about plastic pollution on Cartagena’s beaches.

## Structure

- `src/`: Model training and utility scripts
- `data/`: Raw and processed image datasets
- `notebooks/`: Exploratory notebooks
- `requirements.txt`: Required Python libraries

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python src/train_cyclegan.py
   ```

3. Generate images:
   ```bash
   python src/generate_images.py
   ```
