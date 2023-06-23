## RecGen
Code repository from the publication "Prediction of designer-recombinases for DNA editing with generative deep learning"

RecGen is a conditional variational autoencoder for the generation of tyrosine site-specific recombinases selective for the defined DNA target site. The repository contains the code that was used to train the RecGen models.

You can find the publication [here](https://www.nature.com/articles/s41467-022-35614-6) and the recombinase sequences [here](https://www.ebi.ac.uk/ena/browser/view/PRJEB57361)

### Content:
- vae_train_loocv.py: perform leave-one-out cross-validation on the training data.
- vae_train_save.py: train models with all libraries and save models.
- vae_load_predict.py: load trained models and predict for target site of interest.

### Example Data:
- training_data_masked.csv: an example how the input data should look like and can be used to test the software. The data is not useful, all mutations have been replaced with stars.
- predict_ts.csv: contains the target sites for which predictions were made for in the publication, can be used for testing.

### Requiremnts:
- Python3.9 with pandas, numpy, pytorch (compiled for cuda), argparse
- A cuda capable GPU

The application has been tested on Arch Linux v5.16.5.arch1-1 with Python 3.9.9, pytorch-gpu 1.10.1, pandas 1.4.0, numpy 1.22.1.
To train the models a Nvidia Geforce RTX 3060 was used.

### Installation:
I recommend installing pytorch over conda, which shouldn't take more than a couple of minutes:
```
conda create -n "pytorch" python=3.9
conda activate pytorch
conda install -c conda-forge pytorch-gpu
conda install -c anaconda pandas
conda install -c anaconda numpy
```
To download the repository for use:
```
git clone https://github.com/ltschmitt/RecGen
```

### Usage Demo:
#### Leave-one-out cross-validation:
```
python vae_train_loocv.py -i example_input/training_data_masked.csv
```
Expected output in output_loocv/: 
- loss.csv: the loss values observed over the course of training
- parameters.txt: the parameters used for training
- prediction_freqs.csv: the frequency of the predicted amino acids for each position
- prediction_hamming.csv: the hamming distances of for example the left out libraries and the predictions
- prediction_strings.csv: the predicted target site + amino acid sequences in a comma seperated format

#### Prediction of novel recombinases:
```
python vae_train_save.py -i example_input/training_data_masked.csv
python vae_load_predict.py -m saved_models -t example_input/predict_ts.csv -d example_input/training_data_masked.csv 
``` 
Expected output in saved_models/: 
- parameters.txt: the parameters used for training
- CVAE_0.pt: the model file

Expected output in output_prediction/:
- parameters.txt: the parameters used for prediction
- prediction_str.csv: the predicted recombinase sequences with their target sites

All of these processes are not very demanding, so they should be done within a few minutes.

### Further Usage:
In case you want to test the application with custom data I recommend to use the --help flag on the scripts to learn about how the parameters can be adapted for your needs.
