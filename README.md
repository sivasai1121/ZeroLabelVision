# ZeroLabel Vision

## AI-Based Image Recognition and Similarity Analysis Platform

ZeroLabel Vision is an AI-powered computer vision project developed using Python and Streamlit. The main purpose of this project is to identify and retrieve visually similar images using deep learning-based feature extraction and similarity analysis. Instead of manually labeling every input image, the system extracts numerical features from an image and compares them with images stored in a reference gallery.

## Project Description

The application allows users to upload images and compare them with a collection of reference images. A deep learning model extracts features from the input image, and cosine similarity is used to compare its feature vector with the feature vectors of gallery images. Based on the similarity scores, the system ranks the most relevant matches and displays the results through an interactive interface.

ZeroLabel Vision also provides tools for image comparison, gallery management, viewing-angle analysis, interactive image rotation, top-K predictions, unknown-image detection, similarity visualization, and model evaluation. It includes a live camera recognition interface and dataset analysis features where supported by the application.

## Technologies Used

The project uses Python as its primary programming language and Streamlit to create the interactive web interface. PyTorch supports deep learning operations and feature extraction, while NumPy handles numerical computations. Pandas supports tabular data processing, Scikit-learn provides similarity calculations and machine learning utilities, and Pillow is used for image loading and preprocessing.

## Computer Vision Concepts

The main computer vision concepts involved in this project are image preprocessing, deep-learning-based feature extraction, image similarity, image retrieval, cosine similarity, image comparison, top-K prediction, viewpoint variation analysis, image rotation, and model evaluation. These techniques help the system compare images and analyze recognition results.

## Working Methodology

First, the user provides an input image through the application. The image is preprocessed according to the model's requirements, and the deep learning model extracts its feature vector. The system then compares this vector with the feature vectors stored in the reference gallery using cosine similarity. Finally, the candidate matches are ranked according to their similarity scores and displayed to the user.

## Algorithm

The main algorithm combines deep-learning-based feature extraction with cosine similarity. Cosine similarity measures the similarity between two feature vectors using the formula: Cosine Similarity = (A · B) / (||A|| × ||B||), where A represents the input image feature vector and B represents a reference image feature vector. Higher scores generally indicate more similar feature representations, but they do not guarantee that two images contain the same object.

## Installation and Execution

To run the project, install Python and the required dependencies. Clone the repository, open the project directory, create and activate a virtual environment, and install the packages listed in `requirements.txt`. Start the application using the command `python -m streamlit run app.py`. Open the local URL displayed in the terminal to access the interface. The required model files and reference image gallery must be available for the relevant features to work.

## Advantages

ZeroLabel Vision reduces the manual effort involved in image retrieval and provides ranked image matches based on feature similarity. Its interactive interface makes it easier to compare images, inspect similarity scores, and explore how changes in viewing angle may affect recognition. The application also provides tools for analyzing gallery quality and evaluating model behavior.

## Limitations

The performance of the system depends on the selected deep learning model, the quality of the input image, and the contents of the reference gallery. Changes in lighting, background, and viewing angle may affect the similarity scores. Images of visually similar objects can receive high scores even when they belong to different classes, so the results should not be treated as guaranteed predictions.

## Future Enhancements

Future improvements may include expanding the reference gallery, improving feature extraction, optimizing recognition speed, adding more evaluation metrics, improving robustness to lighting and viewpoint changes, and deploying the application on a cloud platform.

## Conclusion

ZeroLabel Vision demonstrates how artificial intelligence and computer vision can be used for image recognition, similarity analysis, and visual retrieval. By extracting image features and comparing them with a reference gallery, the system identifies potential matches and presents ranked results through an interactive web application.

**Project Name:** ZeroLabel Vision

**Domain:** Artificial Intelligence, Machine Learning, and Computer Vision

**Programming Language:** Python

**Framework:** Streamlit

**Core Technique:** Deep Learning-Based Feature Extraction and Cosine Similarity
