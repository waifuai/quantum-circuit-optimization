import argparse
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from typing import List

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Convert CSV to TFRecord')
    parser.add_argument('-i', '--input', default='qc5f_1k_2.csv',
                        help='Input CSV file path')
    parser.add_argument('-o', '--output', default='qc5f_1k_2.tfrecords',
                        help='Output TFRecord file path')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of records (for testing)')
    return parser.parse_args()


def convert_to_tfrecord(input_csv: str, output_tfrecord: str, limit: int = None) -> None:
    """
    Converts a CSV file to TFRecord format.

    Args:
        input_csv: Path to the input CSV file.
        output_tfrecord: Path to the output TFRecord file.
        limit: Optional limit on the number of records to convert.
    """
    # Read CSV data using pandas
    df: pd.DataFrame = pd.read_csv(input_csv)
    if limit:
        df = df.head(limit)
    
    # Create TFRecord writer
    with tf.io.TFRecordWriter(output_tfrecord) as writer:
        # Iterate over each row in the DataFrame
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating TFRecords"):
            # Assume first column is label and the rest are features
            label: int = int(row.iloc[0])
            features: List[float] = row.iloc[1:].tolist()
            
            # Create a TF Example
            example: tf.train.Example = tf.train.Example(features=tf.train.Features(feature={
                'features': tf.train.Feature(float_list=tf.train.FloatList(value=features)),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            writer.write(example.SerializeToString())


def main() -> None:
    """Main function to execute the CSV to TFRecord conversion."""
    args: argparse.Namespace = parse_args()
    convert_to_tfrecord(args.input, args.output, args.limit)
    print(f"TFRecord file created at {args.output}")


if __name__ == "__main__":
    main()
