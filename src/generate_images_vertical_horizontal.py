import pandas as pd
import numpy as np
from PIL import Image
import os

# Define a function for dataset processing and image generation
def generate_images(data_path):
    # Load the dataset from CSV into a DataFrame
    df = pd.read_csv(data_path)

    # Drop 'dst' and 'src' columns
    df = df.drop(['dst', 'src'], axis=1)

    # Encode 'protocol' column using one-hot encoding
    df_encoded = pd.get_dummies(df, columns=['protocol'])

    # Create output folders if they don't exist
    output_folder_benign = 'data/BENIGN'
    output_folder_malicious = 'data/MALICIOUS'
    os.makedirs(output_folder_benign, exist_ok=True)
    os.makedirs(output_folder_malicious, exist_ok=True)

    # Iterate over each row in the DataFrame
    for index, row in df_encoded.iterrows():
        try:
            # Extract the label from the 'label' column
            label = row['label']

            # Remove the label column from the row
            row = row.drop('label')

            # Convert the row to a numpy array
            row_array = np.array(row)

            # Reshape the row array to a 1x22 numpy array
            row_array = row_array.reshape((1, 22))

            # Repeat the row array vertically and horizontally
            image_array = np.tile(row_array, (22, 1))

            # Create a PIL Image object from the image array
            image = Image.fromarray(image_array)

            # Convert the image to RGB mode
            image = image.convert('RGB')

            # Determine the output folder based on the label
            output_folder = output_folder_benign if label == 0 else output_folder_malicious

            # Generate the output file path
            output_file_path = os.path.join(output_folder, f'image_{index}.png')

            # Save the image as a file
            image.save(output_file_path)

        except Exception as e:
            print(f"Error processing row {index}: {e}")

# If executed as a script
if __name__ == "__main__":
    data_file_path = os.getenv('DATA_FILE_PATH', 'path/to/your/dataset.csv')  # Set the path to your dataset
    generate_images(data_file_path)
