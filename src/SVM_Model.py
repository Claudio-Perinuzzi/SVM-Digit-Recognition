from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from skimage import exposure
from sklearn.svm import SVC
import numpy as np
import pickle

##########################################################################################
# Claudio Perinuzzi, 5/6/25
#
# Credits:
# The algorithm used/modified came from:
#   https://stackoverflow.com/questions/72229909/handwritten-digit-recognition-without-deep-learning-techniques
#   I used a similar approach here as well as included edge features for the SVM which can be found here:
#       https://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm
##########################################################################################

class SVM_Model:
    '''
    SVM Model Class Includes:
        - Initializing the SVM model
        - Loading data from the MNIST data set
        - Extracting features
            - Average Pooling
            - Sobol Edge Detection
        - Training the Model
        - Testing the Model
        - Saving/Loading the Model
        - Performing Predictions
    '''


    def __init__(self):
        self.model = SVC() # Init the SVM model


    def load_data(self, data_file_path):
        '''
        Loads the data from the text file and separates the labels and features
        Converts the features to an np array
        '''

        # Load the data
        with open(data_file_path, 'r') as f:
            data = f.readlines()

        # Define a feature and labels array
        features = []
        labels = []

        # Iterate through each line of the data file
        for line in data:
            
            # Split the line by spaces.
            # EX: ['6.0000', '-1.0000', '-1.0000' ... -> 256 + 1 for label (first index)]
            # Total is 257 including the label        
            values = line.split()

            # Convert to floats (they are strings initially)
            numeric_values = [float(num) for num in values]

            # Pull out the label and define a vector of just the features
            label = int(numeric_values[0])
            feature_vector = np.array(numeric_values[1:])

            # Append to each corresponding list
            labels.append(label)
            features.append(feature_vector)

        # Convert to numpy arrays and return
        return np.array(features), np.array(labels)


    def extract_features(self, img_data):
        '''
        Extracts the features for each image (averaging and edge)
        and returns it as a flattened 1D np array
        '''

        features = []
        for image in img_data:

            # Reshape the 1D data into a 16x16 array (16, 16)
            img_2d = image.reshape(16, 16)

            # Get the features from kernel averaging and edge detection
            avg_feat = self._extract_avg_features(img_2d, kernel_size=2)   # Does best with kernel size = 2
            edge_feat = self._extract_edge_features(img_2d, kernel_size=3) # Does best with kernel size = 3
            
            # Combine these features and append to our features where each index belongs to a specific img
            combined_features = np.concatenate([avg_feat, edge_feat])
            features.append(combined_features)
        
        # Convert back to a np array
        return np.array(features)  


    def _extract_avg_features(self, img, kernel_size=2):
        '''
        Performs a kernel averging convolution, then flatten back to a normalized 1D   
        feature vector which will be fed into the SVM model

        https://stackoverflow.com/questions/72229909/handwritten-digit-recognition-without-deep-learning-techniques
        '''

        # Get the output dimensions (8x8 if the kernel size is 2)
        output_size = 16 // kernel_size

        # Initialize the output feature map with the output size
        feature_map = np.zeros((output_size, output_size))

        # Perform kernel averaging by iterating through the dimensions
        for row in range(output_size):
            for col in range(output_size):

                '''
                Define the boundaries of the current kernel region
                Where we get the starting and ending indices of
                both the row and columns to be used for averaging
                '''
                start_row   = row * kernel_size
                end_row     = start_row + kernel_size
                start_col   = col * kernel_size
                end_col     = start_col + kernel_size
                
                # Extract this specific kernel region from the image
                kernel_region = img[start_row:end_row, start_col:end_col]
                
                # Calculate the average value of the kernel region and store in the feature map
                feature_map[row, col] = np.mean(kernel_region)

        # Flatten the feature map back into a 1D vector
        avg_feat = feature_map.flatten()
    
        # Normalize the vector
        avg_feat = (avg_feat - np.min(avg_feat)) / (np.max(avg_feat) - np.min(avg_feat) + 1e-6)

        return avg_feat


    def _extract_edge_features(self, img, kernel_size=3):
        '''
        Uses two masks (3x3 each) for detecting horizontal and vertical edges and
        then performs max pooling on the edge values. 

        https://www.geeksforgeeks.org/sobel-edge-detection-vs-canny-edge-detection-in-computer-vision/
        https://en.wikipedia.org/wiki/Sobel_operator
        '''

        # Get the output dimensions (5x5 if the kernel size is 3)
        output_size = 16 // kernel_size

        # Horizontal edge detection mask
        Gx = np.array([[ -1,  0,  1],
                       [ -2,  0,  2],
                       [ -1,  0,  1]])
        
        # Vertical edge detection mask
        Gy = np.array([[ 1,  2,  1],
                       [ 0,  0,  0],
                       [-1, -2, -1]])

        # 45 degree edge detection mask
        G45 = np.array([[ 0,  1, 2], 
                        [-1,  0, 1], 
                        [-2, -1, 0]])  

        # Preprocess the image by improving the contrast to enhance edge detection
        img = exposure.equalize_hist(img)  

        # First pad the image with a boarder of width 1 and constant 0's
        # Necessary for handling edges of the image during the convolution
        padded_img = np.pad(img, pad_width=1, mode='constant')

        # Init a array to store gradient magnitudes for each pixel (of same shape of orig img)
        gradient_magnitude = np.zeros_like(img)

        # Iterate over the padded image (1 to 17 because the padded image is 18x18)
        # We need a 3x3 neighborhood around each pixel in the orig 16x16 img
        for i in range(1, 17):
            for j in range(1, 17):
                
                # Get the 3x3 neighborhood around this (i,j) pixel
                region = padded_img[i-1:i+2, j-1:j+2]

                # Calc the gradient in the x and y direction by using the kernels defined above
                # Element wise mult and summation is used here
                gx = np.sum(Gx * region)
                gy = np.sum(Gy * region)
                g45 = np.sum(G45 * region)

                # Calc the magnitude of the gradient using pythagorean theorem 
                # This is used to represent the edge strenght of the curr pixel
                gradient_magnitude[i-1, j-1] = np.sqrt(gx**2 + gy**2 + g45**2)

        # Once we have the gradients, now we can do average pooling

        # Init an empty pool array of (5x5 if kernel size = 3)
        edge_pooled = np.zeros((output_size, output_size))

        # Iterate through the output size dimensions
        for row in range(output_size):
            for col in range(output_size):

                # Define the boundaries of the current kernel region
                start_row   = row * kernel_size
                end_row     = start_row + kernel_size
                start_col   = col * kernel_size
                end_col     = start_col + kernel_size

                # Get this specific region from the gradient mag
                region = gradient_magnitude[start_row:end_row, start_col:end_col]

                # Get the pooled mean of this region and place into the pooled array
                edge_pooled[row, col] = np.mean(region)

        # Flatten the feature back into a 1D vector
        edge_feat = edge_pooled.flatten()

        # Normalize the vector
        edge_feat = (edge_feat - np.min(edge_feat)) / (np.max(edge_feat) - np.min(edge_feat) + 1e-6)

        return edge_feat


    def train_model(self, train_features, train_labels):
        '''
        Trains a SVM classifer on the features and labels passed in
        '''

        self.model.fit(train_features, train_labels)
        

    def predict_and_evaluate(self, test_features, test_labels, show_confusion=False):
        '''
        Makes a prediction on the test set, evaluates overall accuracy, class accuracy
        and optionally shows a confusion matrix
        '''
        
        predictions = self.model.predict(test_features)
        accuracy = np.mean(predictions == test_labels)

        # Only show the confusion matrix if this is true
        if show_confusion:
            conf_m = confusion_matrix(test_labels, predictions)
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_m)
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.show()

            # Also calc the per class accuracy with the diagnoals / row sums
            class_counts = conf_m.sum(axis=1)  # Total true instances per class
            correct_counts = np.diag(conf_m)   # Correct predictions per class
            per_class_accuracy = correct_counts / class_counts

            # Iterate through the classes and their accuracies
            print("\nPer-Class Accuracy:")
            for i, acc in enumerate(per_class_accuracy):
                print(f"Class {i}: {acc:.2f}")

        return predictions, accuracy

    
    def predict(self, data):
        return self.model.predict(data)


    def save_model(self, filename):
        '''
        Saves the trained SVM model to a file 
        '''
        try:
            with open(filename, 'wb') as f: 
                pickle.dump(self.model, f)
            print(f"Model saved to {filename}")
        except Exception as e:
            print(f"ERROR saving the model: {e}")

    def load_model(self, filename):
        '''
        Loads the trained SVM model from a file 
        '''
        try:
            with open(filename, 'rb') as f: 
                self.model = pickle.load(f)
            print(f"Model loaded from {filename}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None