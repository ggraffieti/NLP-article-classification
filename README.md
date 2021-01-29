# Fake articles classification using BERT

## How to launch the training
1. Download real and fake news dataset from [here](https://www.kaggle.com/nopdev/real-and-fake-news-dataset) and put in into the `data` folder. 
2. Launch the preprocess script with `python -m utils.preprocess_data` to obtain the train-valid-test datasets.
3. Build the conda environemnt with `conda env create -f environment.yml`.
4. Launch the training script with `python train.py`. 

## Notes
- A GPU is highly recommended. 
- Training time (on a NVidia Titan X GPU) ~ 5m30s
- Accuracy: ~95.7%
