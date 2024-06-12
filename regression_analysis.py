import pandas as pd  # Importing pandas for data manipulation
import numpy as np  # Importing numpy for numerical operations
import matplotlib.pyplot as plt  # Importing matplotlib for data visualization
from sklearn.model_selection import train_test_split  # Importing train_test_split for splitting the dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Importing StandardScaler and OneHotEncoder for data preprocessing
from sklearn.compose import ColumnTransformer  # Importing ColumnTransformer to manage preprocessing pipelines
from sklearn.pipeline import Pipeline  # Importing Pipeline to streamline preprocessing and model training
from sklearn.linear_model import LinearRegression  # Importing LinearRegression for regression analysis
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Importing metrics to evaluate model performance


def run_regression_analysis():
    # Step 1: Create a synthetic dataset for Berlin apartment rental prices
    np.random.seed(42)  # Setting the seed for reproducibility

    # Number of samples in the dataset
    n_samples = 1000

    # Generate synthetic features
    location = np.random.choice(
        ['Mitte', 'Friedrichshain', 'Kreuzberg', 'Neukölln', 'Charlottenburg'],
        n_samples)  # Randomly assign a location from the given options
    size_sqm = np.random.uniform(
        20, 150, n_samples)  # Generate random sizes in square meters
    num_rooms = np.random.randint(1, 6,
                                  n_samples)  # Generate random number of rooms
    distance_to_transport = np.random.uniform(
        0.1, 5, n_samples)  # Generate random distances to transport in km
    age_of_building = np.random.randint(
        1, 100, n_samples)  # Generate random ages of the building in years

    # Create a DataFrame to hold the features
    df = pd.DataFrame({
        'location': location,
        'size_sqm': size_sqm,
        'num_rooms': num_rooms,
        'distance_to_transport': distance_to_transport,
        'age_of_building': age_of_building
    })

    # Generate the target variable (rental price) using a synthetic formula
    df['rental_price'] = (
        10 * df['size_sqm'] + 100 * df['num_rooms'] +
        -50 * df['distance_to_transport'] + -2 * df['age_of_building'] +
        np.random.normal(0, 50, n_samples)  # Add some random noise
    )

    # Display the first few rows of the synthetic dataset for verification
    print("First few rows of the synthetic dataset:")
    print(df.head())

    # Step 2: Handle missing values (no missing values in synthetic data)

    # Step 3: Encode categorical variables
    # Using OneHotEncoder to transform the 'location' categorical variable into binary columns
    categorical_features = ['location']
    numerical_features = df.drop(
        columns=['location', 'rental_price'
                 ]).columns.tolist()  # Identify numerical features

    # Step 4: Split the dataset into features (X) and target (y), then into training and testing sets
    X = df.drop('rental_price',
                axis=1)  # Separate features from target variable
    y = df['rental_price']  # Define target variable

    # Use train_test_split to create training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Step 5: Create a pipeline to preprocess the data and train the Linear Regression model
    # ColumnTransformer to apply different transformers to categorical and numerical features
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(),
         categorical_features),  # One-hot encode categorical features
        ('num', StandardScaler(),
         numerical_features)  # Standardize numerical features
    ])

    # Create a pipeline that includes preprocessing and regression model training
    pipeline = Pipeline(steps=[('preprocessor',
                                preprocessor), ('regressor',
                                                LinearRegression())])

    # Train the model using the training data
    pipeline.fit(X_train, y_train)

    # Step 6: Predict the target values for the testing set and evaluate the model's performance
    y_pred = pipeline.predict(X_test)  # Generate predictions on the test set

    # Calculate evaluation metrics to understand the model's performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Print the evaluation metrics
    print("Model Performance:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R2): {r2}")

    # Step 7: Visualizations to assess model performance
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    # Plotting the distribution of the target variable (rental prices)
    axs[0].hist(df['rental_price'], bins=30, edgecolor='k', alpha=0.7)
    axs[0].set_title('Distribution of the Target Variable (rental_price)')
    axs[0].set_xlabel('Rental Price in Euros')
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

    # Residuals plot to diagnose model performance
    residuals = y_test - y_pred
    axs[2].scatter(y_pred, residuals, alpha=0.7, color='r')
    axs[2].hlines(y=0,
                  xmin=min(y_pred),
                  xmax=max(y_pred),
                  color='k',
                  linestyle='--',
                  linewidth=2)
    axs[2].set_title('Residuals Plot')
    axs[2].set_xlabel('Predicted Values')
    axs[2].set_ylabel('Residuals')
    axs[2].grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


# Run the regression analysis function if this script is executed directly
if __name__ == "__main__":
    run_regression_analysis()
