import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(csv_file_path):
    """Loads and preprocesses the data.

    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        tuple: X_train, X_test, y_train, y_test, input_dim
    """
    try:
        data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        exit()

    data = data.replace(['U3Gate', 'CnotGate', 'Measure', 'BLANK'], [1, 2, 3, 4])

    dense_features = []
    for i in range(41):
        n = str(i).zfill(2)
        prefix = f"gate_{n}_"
        dense_features.extend([
            prefix + suffix 
            for suffix in ["Gate_Type", "Gate_Number", "Control", "Target", "Angle_1", "Angle_2", "Angle_3"]
        ])

    mms = MinMaxScaler()
    data[dense_features] = mms.fit_transform(data[dense_features])

    n_qubits = 5
    target_columns = [f"statevector_{bin(i)[2:].zfill(n_qubits)}" for i in range(2**n_qubits)]

    X = data[dense_features].values
    y = data[target_columns].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_dim = len(dense_features)
    
    return X_train, X_test, y_train, y_test, input_dim