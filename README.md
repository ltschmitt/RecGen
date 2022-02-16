## RecGen
Code repository from the publication "Prediction of designer-recombinases for DNA editing with generative deep learning"

RecGen is a conditional variational autoencoder for the generation of tyrosine site-specific recombinases selective for the defined DNA target site. The repository contains the scripts that were used to train the RecGen models. The also allow for testing of other types of VAEs with different parameters.

### Content:

- vae_train_loocv.py: perform leave-one-out cross-validation on the data. Either define the libraries you want to leave out and predict or do it for all libraries.
- vae_train_save.py: train models with all libraries and save models.
- vae_load_predict.py: load trained models and predict for target site of interest.

### Data:
- training_data_encoded.csv contains a csv file that serves as an example how the input data should look like. The data contained in there is not suitable for training the model.
- predict_ts.csv contains the target sites that where predictions were made for in the publication

### Requiremnts:
Python3, pandas, numpy, pytorch (compiled for cuda), argparse
