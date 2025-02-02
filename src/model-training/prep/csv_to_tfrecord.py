import argparse
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Convert CSV to TFRecord')
    parser.add_argument('-i', '--input', default='qc5f_1k_2.csv',
                        help='Input CSV file path')
    parser.add_argument('-o', '--output', default='qc5f_1k_2.tfrecords',
                        help='Output TFRecord file path')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of records (for testing)')
    return parser.parse_args()

def convert_to_tfrecord(input_csv: str, output_tfrecord: str, limit: int = None):
    # Read CSV data
    df = pd.read_csv(input_csv)
    if limit:
        df = df.head(limit)
    
    # Create TFRecord writer
    with tf.io.TFRecordWriter(output_tfrecord) as writer:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating TFRecords"):
            # Assume first column is label and the rest are features
            label = int(row.iloc[0])
            features = row.iloc[1:].tolist()
            
            # Create a TF Example
            example = tf.train.Example(features=tf.train.Features(feature={
                'features': tf.train.Feature(float_list=tf.train.FloatList(value=features)),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            writer.write(example.SerializeToString())

def main():
    args = parse_args()
    convert_to_tfrecord(args.input, args.output, args.limit)
    print(f"TFRecord file created at {args.output}")

if __name__ == "__main__":
    main()
