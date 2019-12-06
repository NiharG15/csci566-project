## CSCI566 Deep Learning Class Project

This is the code for VAE based approach to translating Image to Music and Music to Image through a shared network. The files are taken from [MusicVAE](https://github.com/tensorflow/magenta/tree/master/magenta/models/music_vae) and modified for this project.

### To Execute:
1. Clone [Magenta Repo](https://github.com/tensorflow/magenta/)
2. Replace the files in the models/music_vae directory with the files in this repository.
3. Install using `pip install -e .`.
4. The model can be trained using `music_vae_train.py` and trained models can be used for prediction using the class in `trained_model.py`.