import tkinter as tk  # Import the Tkinter library for creating GUI applications
from tkinter import messagebox  # Import the messagebox module for displaying message boxes
from regression_analysis import run_regression_analysis  # Import the function to run regression analysis
from classification_analysis import run_classification_analysis  # Import the function to run classification analysis


# Function to determine which analysis to run based on user's selection
def choose_analysis():
    # Get the selected analysis type from the radio buttons
    analysis_type = analysis_choice.get()

    # Check if the user selected "Regression" and run the corresponding function
    if analysis_type == "Regression":
        run_regression_analysis()
    # Check if the user selected "Classification" and run the corresponding function
    elif analysis_type == "Classification":
        run_classification_analysis()
    # If no option is selected, show an error message
    else:
        messagebox.showerror("Error", "Please choose an analysis type.")


# Create the main window for the application
root = tk.Tk()
root.title("Choose Analysis Type")

# Add a label to instruct the user to select an analysis type
label = tk.Label(root, text="Select the type of analysis to run:")
label.pack(pady=10)  # Add some padding for better spacing

# Create a Tkinter variable to store the user's analysis choice
analysis_choice = tk.StringVar(value="")

# Add a radio button for selecting "Regression" analysis
regression_rb = tk.Radiobutton(root,
                               text="Regression",
                               variable=analysis_choice,
                               value="Regression")

# Add a radio button for selecting "Classification" analysis
classification_rb = tk.Radiobutton(root,
                                   text="Classification",
                                   variable=analysis_choice,
                                   value="Classification")

# Pack (display) the radio buttons in the window
regression_rb.pack(anchor=tk.W)  # Align to the west (left) side
classification_rb.pack(anchor=tk.W)  # Align to the west (left) side

# Add a button to submit the choice and run the selected analysis
submit_button = tk.Button(root, text="Run Analysis", command=choose_analysis)
submit_button.pack(pady=20)  # Add some padding for better spacing

# Start the Tkinter main event loop to keep the window open and responsive
root.mainloop()
