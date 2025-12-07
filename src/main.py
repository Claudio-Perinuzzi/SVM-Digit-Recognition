from SVM_Model import SVM_Model
import sys
import os

##########################################################################################
# Claudio Perinuzzi, 5/15/25
# Main File Includes:
#   - The overall pipeline for training and getting predictions for a SVM model
#   - Please see the SVM_Model.py file for implementation details
##########################################################################################

def main():

    # Defined file paths for train and test data
    train_file_path = 'data/train-data.txt'
    test_file_path = 'data/test-data.txt'


    # If the there is no saved model, then train a new one
    if not os.path.exists('svm_model.pkl'):

        print('Model does not exist, training a new one...')

        # Create a SVM model object to manage loading data extracting 
        # features, training and performing predictions
        model = SVM_Model()

        # Load training and testing data
        train_images, train_labels = model.load_data(train_file_path)
        test_images, test_labels = model.load_data(test_file_path)

        # Extract features for the model to be trained on
        train_features = model.extract_features(train_images)
        test_features = model.extract_features(test_images)

        # Train the SVM model
        model.train_model(train_features, train_labels)

        # Predict and evaluate on the test set
        predictions, accuracy = model.predict_and_evaluate(test_features, test_labels, show_confusion=True)

        print('Predictions:', predictions)
        print('Accuracy:', accuracy)

        model.save_model('svm_model.pkl')
    
    else:
        model = SVM_Model()
        model.load_model('svm_model.pkl')

        if model is None:
            print('Please delete the svm_model.pkl file and re-run the program')    
            sys.exit(1)

        # The logic here can be used for inference for given data, 
        # but for now just evaluate the test set
        test_images, test_labels = model.load_data(test_file_path)
        test_features = model.extract_features(test_images)
        predictions, accuracy = model.predict_and_evaluate(test_features, test_labels, show_confusion=True)

        print('Predictions:', predictions)
        print('Accuracy:', accuracy)

def test_current_features():

    # Defined file paths for train and test data
    train_file_path = 'data/train-data.txt'
    test_file_path = 'data/test-data.txt'

    print('Training a new Model and testing...')

    # Create a SVM model object to manage loading data extracting 
    # features, training and performing predictions
    model = SVM_Model()

    # Load training and testing data
    train_images, train_labels = model.load_data(train_file_path)
    test_images, test_labels = model.load_data(test_file_path)

    # Extract features for the model to be trained on
    train_features = model.extract_features(train_images)
    test_features = model.extract_features(test_images)

    # Train the SVM model
    model.train_model(train_features, train_labels)

    # Predict and evaluate on the test set
    predictions, accuracy = model.predict_and_evaluate(test_features, test_labels, show_confusion=True)

    print('Predictions:', predictions)
    print('Accuracy:', accuracy)


def test_no_features():

    print("I am the test for evaluating just the normalized values")

    # Defined file paths for train and test data
    train_file_path = 'data/train-data.txt'
    test_file_path = 'data/test-data.txt'

    # Create a SVM model object to manage loading data extracting 
    # features, training and performing predictions
    model = SVM_Model()

    # Load training and testing data
    train_images, train_labels = model.load_data(train_file_path)
    test_images, test_labels = model.load_data(test_file_path)

    # Extract features for the model to be trained on
    # train_features = model.extract_features(train_images, kernel_size=2)
    # test_features = model.extract_features(test_images, kernel_size=2)

    # Train the SVM model
    model.train_model(train_images, train_labels)

    # Predict and evaluate on the test set
    predictions, accuracy = model.predict_and_evaluate(test_images, test_labels)

    print('Predictions:', predictions)
    print('Accuracy:', accuracy)

if __name__ == '__main__':
    main()
    # test_no_features()
    # test_current_features()
