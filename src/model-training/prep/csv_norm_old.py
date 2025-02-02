"""
CSV preprocessing

Normalizes numeric features in a CSV file and saves the result.
This older script loads the entire CSV into memory and uses MinMaxScaler 
for normalization. It is kept for reference and is not recommended for very large datasets.
"""

import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

GATE_MAPPING = {'U3Gate': 1, 'CnotGate': 2, 'Measure': 3, 'BLANK': 4}
NUM_GATES = 41

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Normalize quantum circuit data')
    parser.add_argument('-i', '--input', default='./qc5f.csv',
                        help='Input CSV file path')
    parser.add_argument('-o', '--output', default='qc5f_normalized.csv',
                        help='Output CSV file path')
    return parser.parse_args()

def generate_feature_names(num_gates: int) -> list:
    """Generate list of feature names for normalization."""
    features = []
    for gate_num in range(num_gates):
        prefix = f"gate_{gate_num:02}_"
        features.extend([
            f"{prefix}Gate_Type",
            f"{prefix}Gate_Number",
            f"{prefix}Control",
            f"{prefix}Target",
            f"{prefix}Angle_1",
            f"{prefix}Angle_2",
            f"{prefix}Angle_3"
        ])
    return features

def main():
    args = parse_args()
    
    # Load data and replace gate types using the predefined mapping
    data = pd.read_csv(args.input)
    data = data.replace(GATE_MAPPING)
    
    # Generate feature list and apply MinMax scaling
    dense_features = generate_feature_names(NUM_GATES)
    scaler = MinMaxScaler()
    data[dense_features] = scaler.fit_transform(data[dense_features])
    
    # Save the processed data
    data.to_csv(args.output, index=False)
    print(f"Successfully processed {len(data)} records. Output saved to {args.output}")

if __name__ == "__main__":
    main()
