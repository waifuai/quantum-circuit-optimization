import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(csv_file_path: str, n_qubits: int):
    """Load and preprocess CSV data for statevector regression."""
    data = pd.read_csv(csv_file_path)
    data = data.replace(['U3Gate', 'CnotGate', 'Measure', 'BLANK'], [1, 2, 3, 4])
    
    dense_features = []
    for i in range(41):
        prefix = f"gate_{str(i).zfill(2)}_"
        dense_features.extend([f"{prefix}{suffix}" for suffix in 
                               ["Gate_Type", "Gate_Number", "Control", "Target", "Angle_1", "Angle_2", "Angle_3"]])
    
    scaler = MinMaxScaler()
    data[dense_features] = scaler.fit_transform(data[dense_features])
    
    statevectors = [bin(i)[2:].zfill(n_qubits) for i in range(2**n_qubits)]
    target = [f"statevector_{sv}" for sv in statevectors]
    return data[dense_features].values, data[target].values, target