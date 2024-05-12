from midetection.Utils import echo_config
from midetection.Utils import mri_config

batch_size = 16
valid_size = 0.15
epochs = 65
margin = 2

### paths BELOW IS FOR MRI
class MRIPaths:
    training_dir = mri_config.partial_paths[0]
    training_csv = "../datasets/mri/train_data.csv"
    validating_dir = mri_config.partial_paths[4]
    validating_csv = "../datasets/mri/valid_data.csv"
    testing_dir = mri_config.partial_paths[2]
    testing_csv = "../datasets/mri/test_data.csv"

### paths BELOW IS FOR ECHO
class EchoPaths:
    training_dir = echo_config.trainInput_data
    training_csv = "../datasets/echo/train_data.csv"
    validating_dir = echo_config.validInput_data
    validating_csv = "../datasets/echo/valid_data.csv"
    testing_dir = echo_config.testInput_data
    testing_csv = "../datasets/echo/test_data.csv"