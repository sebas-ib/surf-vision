# SurfVision

## About the Project

As surfers (and soon to be ones), we want to understand wave patterns to improve our & others' surfing experience.

The initial idea was to use Convolutional Neural Networks to classify images of waves into different categories like "plunging" or "surging", hence the name SurfVision.

This turned out to be a hard task, since there are no well-maintained and big datasets on wave images.

That's why we pivoted to a more feasable method for this scope: we still wanted to improve the overall surfing experience: wave classification based on sensory data turned out to be a feasable project. We use random forest classification or logistic regression to classify waves into "worth" or "not worth" to surf.

Once installed, you can train a machine learning model with random forest classification or logistic regression based on southern california swell & weather data.

## Getting Started

You can train and evaluate surf condition prediction models using the `main.py` script.

### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/sebas-ib/surf-vision.git
   ```

2. Change directory, then create a virtual environment & activate it

   ```sh
   cd surf-vision && python3 -m venv venv
   ```

   ```sh
   source venv/bin/activate
   ```

3. Install packages from requirements

   ```sh
   pip3 install -r requirements.txt
   ```

## Usage

You can train the model based on the included data and the default included models:

```sh
python3 main.py --model=random_forest --spot=blacks_beach
```

### Command Line Argument `--model`

Model to use for prediction (default: `random_forest`).

Defined in `config.py`:

```python
models = {
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'logistic_regression': LogisticRegression(alpha=0.01, num_iters=1000),
    # Add more models here if needed
}
```

To add a model, instantiate it in the `models` dictionary. (Note: a model class has to have a `.fit()` method to train the model and a `.predict()` method to give predictions)

### Command Line Argument `--spot`

Surf spot to use to train model on (required).

Defined in `config.py`:

```python
spot_files = {
    'blacks_beach': 'swell-data/Blacks_Beach_North_San_Diego_County_swell_data.csv',
    'cardiff_reef': 'swell-data/Cardiff_Reef_North_San_Diego_County_swell_data.csv',
    'malibu_north': 'swell-data/Malibu_North_Los_Angeles_County_swell_data.csv',
    'oceanside_harbour': 'swell-data/Oceanside_Harbor_North_San_Diego_County_swell_data.csv',
    'oceanside_pier': 'swell-data/Oceanside_Pier_North_San_Diego_County_swell_data.csv',
    'san_onofre': 'swell-data/San_Onofre_South_Orange_County_swell_data.csv'
    # Add more spots here if needed
}
```

To add a spot, add the its relative file path to the `spot_files` dictionary. (Note: the csv data file has to be of the same format as the default spots)

### Example Commands

Train a Random Forest model on data from Blacks Beach:

```sh
python3 main.py --model=random_forest --spot=blacks_beach
```

### Results

The trained model will be saved as a .joblib file in the /results directory. It will also contain an accuracy and classification report in csv format. The same report will also be printed to the console. Example:

```console
$ python3 main.py --model=random_forest --spot=san_onofre

Data loaded
Data preprocessed
Features selected
Training RandomForestClassifier...
Model trained

Model Accuracy: 0.92

Classification Report:

                   precision  recall  f1-score  support
Not Worth Surfing       0.87    1.00      0.93    13.00
Worth Surfing           1.00    0.85      0.92    13.00
accuracy                0.92    0.92      0.92     0.92
macro avg               0.93    0.92      0.92    26.00
weighted avg            0.93    0.92      0.92    26.00

Feature importance plot saved as 'results/random_forest/blacks_beach/feature_importance_blacks_beach.png'

Model saved as 'results/random_forest/blacks_beach/surf_prediction_model_blacks_beach.joblib'
```

## Dependencies

This project uses the following external libraries:

- [NumPy](https://numpy.org/): Used for numerical computations and array operations.
- [Pandas](https://pandas.pydata.org/): Used for data manipulation and preprocessing in `data_preprocessing.py`.
- [scikit-learn](https://scikit-learn.org/stable/): Used for machine learning models and evaluation metrics in `logistic_regression.py`, `random_forest.py`, and `model_training.py`.
- [Matplotlib](https://matplotlib.org/): Used for data visualization, such as plotting feature importance.
- [Joblib](https://joblib.readthedocs.io/en/latest/): Used for saving and loading trained models in `model_utils.py`.

## Contributing

We are a team of five, all studying at San Diego State. Our GitHub profiles are:

Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
