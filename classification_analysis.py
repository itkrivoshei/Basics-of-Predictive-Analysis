import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def run_classification_analysis():
    # Step 1: Create a synthetic dataset for Berlin apartment classification
    np.random.seed(42)  # For reproducibility

    # Number of samples
    n_samples = 1000

    # Features
    location = np.random.choice(
        ['Mitte', 'Friedrichshain', 'Kreuzberg', 'Neukölln', 'Charlottenburg'],
        n_samples)
    size_sqm = np.random.uniform(20, 150, n_samples)  # Size in square meters
    num_rooms = np.random.randint(1, 6, n_samples)  # Number of rooms
    distance_to_transport = np.random.uniform(
        0.1, 5, n_samples)  # Distance to nearest public transport in km
    age_of_building = np.random.randint(
        1, 100, n_samples)  # Age of the building in years

    # Create a DataFrame
    df = pd.DataFrame({
        'location': location,
        'size_sqm': size_sqm,
        'num_rooms': num_rooms,
        'distance_to_transport': distance_to_transport,
        'age_of_building': age_of_building
    })

    # Target variable: 'luxury' apartment (1 if rental price is above a threshold, 0 otherwise)
    rental_price = (
        10 * df['size_sqm'] + 100 * df['num_rooms'] +
        -50 * df['distance_to_transport'] + -2 * df['age_of_building'] +
        np.random.normal(0, 50, n_samples)  # Adding some noise
    )

    # Adjust the threshold to ensure a balanced dataset
    threshold = np.percentile(rental_price, 70)  # Top 30% as luxury
    df['luxury'] = (rental_price > threshold).astype(int)

    # Display the first few rows of the synthetic dataset
    print("First few rows of the synthetic dataset:")
    print(df.head())
    print("Class distribution:", df['luxury'].value_counts())

    # Step 2: Handle missing values (no missing values in synthetic data)

    # Step 3: Encode categorical variables
    # Using OneHotEncoder for the 'location' categorical variable
    categorical_features = ['location']
    numerical_features = df.drop(
        columns=['location', 'luxury']).columns.tolist()

    # Step 4: Split the dataset into features (X) and target (y), then into training and testing sets
    X = df.drop('luxury', axis=1)  # Features
    y = df['luxury']  # Target variable

    # Use train_test_split to split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Step 5: Create a pipeline to preprocess the data and train the Logistic Regression model
    # ColumnTransformer is used to encode categorical features and scale numerical features
    preprocessor = ColumnTransformer(transformers=[(
        'cat', OneHotEncoder(),
        categorical_features), ('num', StandardScaler(), numerical_features)])

    # Pipeline to combine the preprocessing step with the Logistic Regression model
    pipeline = Pipeline(
        steps=[('preprocessor',
                preprocessor), ('classifier', LogisticRegression())])

    # Train the model using the training data
    pipeline.fit(X_train, y_train)

    # Step 6: Predict the target values for the testing set and evaluate the model's performance
    y_pred = pipeline.predict(X_test)

    # Calculate evaluation metrics to understand the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    # Print the evaluation metrics
    print("Model Performance:")
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)

    # Step 7: Visualizations
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Plotting the distribution of the target variable
    axs[0].hist(df['luxury'], bins=3, edgecolor='k', alpha=0.7)
    axs[0].set_title('Distribution of the Target Variable (luxury)')
    axs[0].set_xlabel('Luxury (0 = No, 1 = Yes)')
    axs[0].set_ylabel('Frequency')
    axs[0].grid(True)

    # Scatter plot of actual vs. predicted values
    axs[1].scatter(y_test, y_pred, alpha=0.7, color='b')
    axs[1].plot([min(y_test), max(y_test)],
                [min(y_test), max(y_test)],
                color='k',
                linestyle='--',
                linewidth=2)
    axs[1].set_title('Actual vs. Predicted Values')
    axs[1].set_xlabel('Actual Values')
    axs[1].set_ylabel('Predicted Values')
    axs[1].grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_classification_analysis()
