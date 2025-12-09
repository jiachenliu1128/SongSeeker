# SongSeeker
Team 15: Jiachen Liu (jl315), Joey Cai (qcai20), Tong Tong (tong37), Jiachen Liang (liang88)

# Environment Setup
If you are using Conda (recommended), create a new environment and install the required packages:

```bash
conda create -n SongSeeker -f environment.yml
conda activate SongSeeker
```
Alternatively, you can install the required packages using pip:

```bash
pip install -r requirements.txt
```

# Run SongSeeker
The `SongSeeker.ipynb` notebook contains the complete workflow for model training, evaluation and actual use. You can run the notebook in Jupyter or any compatible environment.

# Data
We have already preprocessed the dataset and created two subsets for the notebook:
- `sample_data/processed/genius-clean-with-title-artist-10000.csv`: A subset of 10,000 rows for search evaluation.
- `sample_data/processed/genius-clean-with-title-artist-5000.csv`: A labeled subset of 5,000 rows for training and evaluating the Learn to Rank model.
The original dataset can be found at `https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information` and all data processing scripts are located in the `src` directory.



