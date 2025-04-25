import joblib
import numpy as np
import pandas as pd

# Load the trained model
# Make sure 'predictive_maintenance_model.pkl' is in the same directory
try:
    model = joblib.load('predictive_maintenance_model.pkl')
    print("Model 'predictive_maintenance_model.pkl' loaded successfully.")
except FileNotFoundError:
    print("--- ERROR ---")
    print("Model file 'predictive_maintenance_model.pkl' not found.")
    print("Please make sure the model file is in the same directory as this script.")
    exit() # Exit if model can't be loaded
except Exception as e:
    print(f"An error occurred loading the model: {e}")
    exit()

def predict_failure(temperature, process_temperature, rotational_speed, torque, type_val, tool_wear):
    """
    Takes individual measurements, preprocesses them to match model expectations,
    and returns a failure risk prediction.
    """
    # Create a DataFrame with the input values - Use the ORIGINAL names here initially
    input_data = pd.DataFrame({
        'Air temperature [K]': [temperature],
        'Process temperature [K]': [process_temperature],
        'Rotational speed [rpm]': [rotational_speed],
        'Torque [Nm]': [torque],
        'Type': [type_val], # Temporary column for preprocessing
        'Tool wear [min]': [tool_wear]
    })

    # Preprocess the categorical data (One-Hot Encoding for 'Type')
    # Create columns WITHOUT prefixes first
    input_data['Type_H'] = 0
    input_data['Type_L'] = 0
    input_data['Type_M'] = 0
    if type_val == 'H':
        input_data['Type_H'] = 1
    elif type_val == 'L':
        input_data['Type_L'] = 1
    elif type_val == 'M':
        input_data['Type_M'] = 1

    # --- Correction for Feature Names ---
    # Define the feature column names EXACTLY as the trained model expects them
    # (Based on the previous error message)
    feature_columns_expected_by_model = [
        'num__Air temperature [K]',
        'num__Process temperature [K]',
        'num__Rotational speed [rpm]',
        'num__Torque [Nm]',
        'num__Tool wear [min]',
        'cat__Type_H',
        'cat__Type_L',
        'cat__Type_M'
    ]

    # Rename the columns in the DataFrame to match the expected names
    # Map original/intermediate names to the final expected names
    rename_map = {
        'Air temperature [K]': 'num__Air temperature [K]',
        'Process temperature [K]': 'num__Process temperature [K]',
        'Rotational speed [rpm]': 'num__Rotational speed [rpm]',
        'Torque [Nm]': 'num__Torque [Nm]',
        'Tool wear [min]': 'num__Tool wear [min]',
        'Type_H': 'cat__Type_H',
        'Type_L': 'cat__Type_L',
        'Type_M': 'cat__Type_M'
        # 'Type' column is no longer needed after one-hot encoding and renaming
    }
    input_data_renamed = input_data.rename(columns=rename_map)

    # Select the final features using the CORRECTED names in the correct order
    # Make sure all expected columns exist after renaming before selecting
    try:
        # Only keep columns that the model expects
        input_features = input_data_renamed[feature_columns_expected_by_model]
        print(f"\nColumns being passed to model.predict(): {input_features.columns.tolist()}") # Debug print
    except KeyError as e:
        print(f"--- ERROR ---")
        print(f"Could not select all required feature columns after renaming. Missing key: {e}")
        print(f"Columns available after renaming: {input_data_renamed.columns.tolist()}")
        return "Error preparing features"
    except Exception as e:
         print(f"An unexpected error occurred preparing features: {e}")
         return "Error preparing features"


    # Make prediction using the loaded model
    try:
        print("Attempting model.predict()...")
        prediction = model.predict(input_features)[0] # Get the prediction for the single input row
        print("model.predict() successful.")
        # Return the result as a string
        return "High failure risk" if prediction == 1 else "Low failure risk"
    except Exception as e:
        # Catch errors during prediction itself, including potential further feature mismatches
        print(f"--- Error during model prediction ---")
        print(f"{e}") # Print the specific error from the model
        return "Error in prediction"

# --- Main execution block ---
if __name__ == "__main__":
    print("Starting predictive maintenance tool...")
    while True:
        # Initialize variables used in try/except scope tracking
        temp = proc_temp = speed = torque = wear = None
        temp_str = proc_temp_str = speed_str = torque_str = wear_str = type_val = None

        try:
            # --- Get all inputs first with debugging prints ---
            print("\nAttempting to read Air temperature...")
            temp_str = input("Enter Air temperature [K]: ")
            temp = float(temp_str) # Attempt conversion
            print(f"Successfully read Air temperature: {temp}")

            print("\nAttempting to read Process temperature...")
            proc_temp_str = input("Enter Process temperature [K]: ")
            proc_temp = float(proc_temp_str) # Attempt conversion
            print(f"Successfully read Process temperature: {proc_temp}")

            print("\nAttempting to read Rotational speed...")
            speed_str = input("Enter Rotational speed [rpm]: ")
            speed = float(speed_str) # Attempt conversion
            print(f"Successfully read Rotational speed: {speed}")

            print("\nAttempting to read Torque...")
            torque_str = input("Enter Torque [Nm]: ")
            torque = float(torque_str) # Attempt conversion
            print(f"Successfully read Torque: {torque}")

            print("\nAttempting to read Type...")
            type_val = input("Enter Type (L, M, or H): ").strip().upper()
            if type_val not in ['L', 'M', 'H']:
                print("Invalid Type entered. Please enter L, M, or H.")
                continue # Ask for input again from the start of the loop
            print(f"Successfully read Type: {type_val}")

            print("\nAttempting to read Tool wear...")
            wear_str = input("Enter Tool wear [min]: ")
            wear = float(wear_str) # Attempt conversion
            print(f"Successfully read Tool wear: {wear}")

            # --- Call the prediction function only AFTER getting all valid inputs ---
            print("\nAll inputs read successfully, preparing features and making prediction...")
            result = predict_failure(temp, proc_temp, speed, torque, type_val, wear)

            # --- Print the final result ---
            print(f"\nPrediction: {result}")

        except ValueError as e:
            # --- Enhanced error reporting for ValueError ---
            print(f"\n--- INPUT ERROR ---")
            print(f"Failed to convert one of the measurement inputs to a number.")

            # Try to identify which input caused the error
            failing_input_description = "unknown measurement"
            if temp is None and temp_str is not None: failing_input_description = f"Air Temperature ('{temp_str}')"
            elif proc_temp is None and proc_temp_str is not None: failing_input_description = f"Process Temperature ('{proc_temp_str}')"
            elif speed is None and speed_str is not None: failing_input_description = f"Rotational Speed ('{speed_str}')"
            elif torque is None and torque_str is not None: failing_input_description = f"Torque ('{torque_str}')"
            elif wear is None and wear_str is not None: failing_input_description = f"Tool Wear ('{wear_str}')"

            print(f"The error likely occurred with: {failing_input_description}")
            print(f"Original Python error message: {e}")
            print("Please ensure you enter only valid numbers (digits and potentially a decimal point '.') for measurements.")
            print("---------------------")

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nExiting prediction script.")
            break # Exit the while loop

        except Exception as e:
            # Catch any other unexpected errors during the input/prediction process
            print(f"\nAn unexpected error occurred in the main loop: {e}")
            # Consider logging the error here
            # traceback.print_exc() # Uncomment this line and 'import traceback' at top for full debug info
            break # Exit on other errors

        # Optional: Ask if user wants to continue
        try:
            another = input("\nMake another prediction? (yes/no): ").strip().lower()
            if another != 'yes':
                print("Exiting.")
                break # Exit the while loop
        except KeyboardInterrupt:
             print("\nExiting prediction script.")
             break # Exit the loop if user presses Ctrl+C during this input