import tkinter as tk
from tkinter import messagebox
from regression_analysis import run_regression_analysis
from classification_analysis import run_classification_analysis


def choose_analysis():
    analysis_type = analysis_choice.get()

    if analysis_type == "Regression":
        run_regression_analysis()
    elif analysis_type == "Classification":
        run_classification_analysis()
    else:
        messagebox.showerror("Error", "Please choose an analysis type.")


# Create the main window
root = tk.Tk()
root.title("Choose Analysis Type")

# Add a label
label = tk.Label(root, text="Select the type of analysis to run:")
label.pack(pady=10)

# Create a variable to store the analysis choice
analysis_choice = tk.StringVar(value="")

# Add radio buttons for choosing the analysis type
regression_rb = tk.Radiobutton(root,
                               text="Regression",
                               variable=analysis_choice,
                               value="Regression")
classification_rb = tk.Radiobutton(root,
                                   text="Classification",
                                   variable=analysis_choice,
                                   value="Classification")

regression_rb.pack(anchor=tk.W)
classification_rb.pack(anchor=tk.W)

# Add a button to submit the choice
submit_button = tk.Button(root, text="Run Analysis", command=choose_analysis)
submit_button.pack(pady=20)

# Start the main event loop
root.mainloop()
