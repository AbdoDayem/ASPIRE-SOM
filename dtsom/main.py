# Utility packages
import os
import pickle
import argparse

# Regular SOM packages
from minisom import MiniSom
from classifiers import classify_BMU
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# DTSOM packages
import cupy as cp
from dt_som import DTSOM_CPU, DTSOM_GPU

# Default settings for SOM training
MSIZE = [32, 32]  # SOM grid size
NRADIUS = 3       # Neighborhood radius (sigma)
ITERATIONS = 10    # Number of training iterations
LEARN_RATE = 0.7   # Learning rate
DATASET = "16000_pcm_speeches"  # Dataset name
PCA_COMPONENTS = 80
AUDIO_TIME = 0.99  # Duration of audio in seconds

# Method to handle acceleration (use GPU or CPU based on flag)
def acceleration(use_gpu=False):
    if use_gpu:
        print("Using DTSOM_GPU for training.")
        return DTSOM_GPU  # Return DTSOM_GPU class
    else:
        print("Using DTSOM_CPU for training.")
        return DTSOM_CPU  # Return DTSOM_CPU class

# Method to handle iterations
def set_iterations(num_iterations):
    global ITERATIONS
    ITERATIONS = num_iterations
    print(f"Setting iterations to {ITERATIONS}")

# Main function to parse arguments and run the SOM code
def main():
    global ITERATIONS, SOM_NAME

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate SOMs with optional acceleration, iterations, and user name.")
    
    # Define arguments
    parser.add_argument('--acc', action='store_true', help="Use GPU for training (DTSOM_GPU)")
    parser.add_argument('--it', type=int, help="Set the number of iterations", default=10)
    parser.add_argument('--name', type=str, help="Specify SOM name for saving", default="dt_test2")
    
    # Parse the arguments
    args = parser.parse_args()

    # Handle acceleration (GPU or CPU) based on --acc
    dtsom_class = acceleration(args.acc)

    # Set the number of iterations
    set_iterations(args.it)

    # Set the SOM name
    SOM_NAME = args.name
    print(f"Using SOM name: {SOM_NAME}\n")

    # Load dataset
    with open(f'../data_files/{DATASET}/{AUDIO_TIME}sec_{PCA_COMPONENTS}PCA_data.p', 'rb') as infile:
        data = pickle.load(infile)
    with open(f'../data_files/{DATASET}/{AUDIO_TIME}sec_labels.p', 'rb') as infile:
        labels = pickle.load(infile)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, train_size=0.9, stratify=labels)

    # Initialize the selected DTSOM class (CPU or GPU)
    print(f"Initializing {dtsom_class.__name__}")
    dtsom = dtsom_class(
        x=MSIZE[0],
        y=MSIZE[1],
        input_len=data[0].shape[0],  # Length of each input vector
        sigma=NRADIUS,               # Neighborhood radius (sigma)
        learning_rate=LEARN_RATE,    # Learning rate
        density_threshold=0.1        # Density threshold (optional parameter)
    )

    print("Initializing Weights for DTSOM")
    dtsom.random_weights_init(X_train)

    # Train the selected DTSOM (CPU or GPU)
    print("Training DTSOM")
    dtsom.train(X_train, ITERATIONS, verbose=True)  # Show progress during training

    # Classification with DTSOM
    print("\nClassification with DTSOM")
    print(classification_report(y_test, classify_BMU(dtsom, cp.array(X_test), cp.array(X_train), y_train)))

    # Initialize the standard MiniSom
    print("Initializing MiniSom")
    som = MiniSom(
        x=MSIZE[0],
        y=MSIZE[1],
        input_len=data[0].shape[0],
        sigma=NRADIUS,
        learning_rate=LEARN_RATE,
        random_seed=None
    )

    print("Initializing Weights for MiniSom")
    som.random_weights_init(X_train)

    # Train MiniSom
    print("Training MiniSom")
    som.train_random(X_train, ITERATIONS, verbose=True)

    # Classification with MiniSom
    print("\nClassification with MiniSom")
    print(classification_report(y_test, classify_BMU(som, X_test, X_train, y_train)))

    # Save both SOMs
    if not os.path.exists(f'../som_files/{SOM_NAME}'):
        os.makedirs(f'../som_files/{SOM_NAME}')

    # Save the trained SOMs
    with open(f'../som_files/{SOM_NAME}/som.p', 'wb') as outfile:
        pickle.dump(som, outfile)
    with open(f'../som_files/{SOM_NAME}/dt_som.p', 'wb') as outfile:
        pickle.dump(dtsom, outfile)

if __name__ == "__main__":
    main()
