import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from src import config

def load_and_preprocess_data(csv_file_path, target_type='single', n_qubits=5):
    """Loads and preprocesses the data.

    Args:
        csv_file_path (str): Path to the CSV file.
        target_type (str): 'single' for statevector_00000, 'multi' for all statevectors.
        n_qubits (int): Number of qubits, used if target_type is 'multi'.

    Returns:
        tuple: X_train, X_test, y_train, y_test, input_dim
    """
    try:
        data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        exit()

    # Replace categorical labels with numeric codes
    data = data.replace(['U3Gate', 'CnotGate', 'Measure', 'BLANK'], [1, 2, 3, 4])

    dense_features = []
    for i in range(41):
        n = str(i).zfill(2)
        prefix = f"gate_{n}_"
        dense_features.extend([
            prefix + suffix 
            for suffix in ["Gate_Type", "Gate_Number", "Control", "Target", "Angle_1", "Angle_2", "Angle_3"]
        ])

    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    if target_type == 'single':
        target = ['statevector_00000']
    elif target_type == 'multi':
        target = [f"statevector_{bin(i)[2:].zfill(n_qubits)}" for i in range(2**n_qubits)]
    else:
        raise ValueError("Invalid target_type. Choose 'single' or 'multi'.")

    X = data[dense_features].values
    y = data[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_dim = len(dense_features)
    
    return X_train, X_test, y_train, y_test, input_dim


def dataframe_to_dataset(dataframe, batch_size, shuffle=True):
    """Convert Pandas DataFrame to tf.data.Dataset.
    
    Specifically designed for the DCN model which expects a single target 
    'statevector_00000' and features as a dictionary.
    """
    target = ['statevector_00000'] # Specific target for DCN
    dataframe = dataframe.copy()
    labels = dataframe.pop(target[0])
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds
