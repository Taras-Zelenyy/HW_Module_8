# HW_Module_8 (Basics MLE)
This repository contains a template for a well-structured machine learning project. This project works with the [Iris flower data set - Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set), where the result is a classification of flowers based on the length and width of the `sepals` and `petals`, in centimeters. 

The project covers the following aspects:
- Primitive testing of the main parts of the existing code
- Creating data for models (Splitting data into training and inference sets)
- Data preparation and NN training
- Testing on inference data

## Cloning project from GitHub
To start working with this project, you should clone it to your local machine.

To do this, use the following command via Git Bash or another method convenient for you:

```bash
git clone https://github.com/Taras-Zelenyy/HW_Module_8.git
```

## Settings:
Project configurations are managed using the `settings.json` file. It stores important variables that control the behavior of the project, namely: 
- paths to certain resource files
- constant values
- hyperparameters for the ML model

To pass the path to the configurator to the scripts, you need to create the `.env` file and manually initialize the environment variable as `CONF_PATH=settings.json`. (This step is required for the correct construction of the Docker image)

## Unittests
Before you start working with the main code, you can try to test the main blocks of the project. 
You can test the following three project blocks:
- testing data generation
- testing some aspects of model training
- testing inference results

### Note
There is one bug in the test that I don't know how to solve correctly and reasonably. 
It occurs under the following conditions: after cloning the project, you decide to test individual project blocks. If you immediately decide to start by testing the `inference file`, you will get an error because you don't have any trained model beforehand and the tests will fail. 

Therefore, to avoid this, testing should be performed in the correct order:
1. run test_data_generation.py
2. run test_train.py
3. run test_inference.py

Then everything should work and all tests will pass 

### Data generation tests
Let's take a look at the testing process performed in the `data_generation.py` file. 

The main tests performed in the file include:

- **Dataset availability check**: Ensures that the dataset object was successfully created.
- **Dataset Type**: Validates that the training and test datasets are objects of type `pandas.DataFrame`.
- **Number of rows**: Validates that both datasets contain the expected number of rows, 120 for the training dataset and 30 for the test dataset, and also validates that the training and test datasets contain more than zero rows.
- **Dataset Content**: Verifies that the datasets include all the required columns.

To run data generation tests, paste the following command into the terminal:
```bash
python unittests/test_data_generation.py
```

### Train tests
During the training process, since there is no dedicated data for testing, test data is generated from the same Iris dataset used for model training. To avoid using an excessive amount of data and to ensure test efficiency, only 10% of the full dataset is selected.

After the test is completed, the temporarily created test data is deleted so that it does not affect the further use of the dataset or the training process.

The main tests performed in the file include:

**Model initialization test (test_model_initialization)**:
   - Checks whether the `IrisClassifier` model is initialized without errors.

**Test of the training process (test_default_training)**:
   - Validates the ability of the model to train with default parameters.
   - Uses `DataProcessor` to prepare datasets.
   - Creates DataLoaders for training and test datasets.
   - Initializes the `IrisClassifier` model with the specified parameters.
   - Starts training the model using the `Training` class.
   - Checks whether the model parameters have been updated after training.

To run data generation tests, paste the following command into the terminal:
```bash
python unittests/test_train.py
```

### Inference Tests
During the inference process, similar to training tests, I create test data from the same Iris dataset used for model training.

After the test is completed, the temporarily created test data is deleted so that it does not affect the further use of the dataset or the training process.

The main tests performed in the file include:
1. **Test model loading (test_model_loading)**:
   - Checks if the path to the last saved model can be found.
   - Loads the model using this path and checks whether the model was successfully loaded.

2. **Test of the inference process (test_inference)**:
   - Loads data for inference.
   - Uses the model to perform an inference on the data.
   - Verifies that the inference runs without errors and that the results are successfully generated.

3. **Test of saving results (test_result_storage)**:
   - Creates an application DataFrame that contains the imaginary results of the inferencing for testing.
   - Saves these results to the specified location on the local file system.
   - Verifies that the result file was actually created.

To run data generation tests, paste the following command into the terminal:
```bash
python unittests/test_inference.py
```

## Data:
For generating the data, use the script located at `data_process/data_generation.py`. The generated data is used to train the model and to test the inference. Following the approach of separating concerns, the responsibility of data generation lies with this script.

According to the task condition, the data has already been pre-generated, but to start the data generation process, use the following command:
```bash
python data_process/data_generation.py
```

## Training:
The training phase of the ML pipeline includes preprocessing of data, the actual training of the model, and the evaluation and validation of the model's performance. All of these steps are performed by the script `training/train.py`.

1. To train the model using Docker: 

- Build the training Docker image. If the built is successfully done, it will automatically train the model:
```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```
- You may run the container with the following parameters to ensure that the trained model is here:

```bash
docker run -dit training_image
```
```You can see the logs in the Logs section of the Docker Desktop```

Note: If you do not have a `models` folder at this step, you should create one, otherwise the trained model will not be saved correctly. 

So create a `models` folder if you don't have one.

Then, move the trained model from the directory inside the Docker container `/app/models` to the local machine using:
```bash
docker cp <container_id>:/app/models/<model_name>.pickle ./models
```
Replace `<container_id>` with your running Docker container ID and `<model_name>.pickle` with your model's name .

1. Alternatively, the `train.py` script can also be run locally as follows:

```bash
python training/train.py
```

## Inference:
Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in `inference/run.py`.

1. To run the inference using Docker, use the following commands:

- Build the inference Docker image:
```bash
docker build -f ./inference/Dockerfile --build-arg model_name=<model_name>.pickle --build-arg settings_name=settings.json -t inference_image .
```
- Run the inference Docker container:
```bash
docker run -v /path_to_your_local_model_directory:/app/models -v /path_to_your_input_folder:/app/input -v /path_to_your_output_folder:/app/output inference_image
```
- Or you may run it with the attached terminal using the following command:
```bash
docker run -dit inference_image
```
After that ensure that you have your results in the `results` directory in your inference container.

2. Alternatively, you can also run the inference script locally:

```bash
python inference/run.py
```

Replace `/path_to_your_local_model_directory`, `/path_to_your_input_folder`, and `/path_to_your_output_folder` with actual paths on your local machine or network where your models, input, and output are stored.