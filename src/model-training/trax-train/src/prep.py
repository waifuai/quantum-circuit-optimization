import os

def preprocess_data(input_file: str, input_processed_file: str, output_processed_file: str) -> None:
    """
    Preprocesses the input file by stripping leading whitespace,
    then creates processed files by removing the last line for input and the first line for output.
    """
    with open(input_file, 'r') as f:
        lines = [line.lstrip() for line in f.readlines()]

    with open(input_processed_file, 'w') as f_in_proc:
        f_in_proc.writelines(lines[:-1])

    with open(output_processed_file, 'w') as f_out_proc:
        f_out_proc.writelines(lines[1:])

    print(f"Data preparation complete. Processed files: {input_processed_file}, {output_processed_file}")

def main():
    data_dir = "data"
    input_file = os.path.join(data_dir, "input.txt")
    input_processed_file = os.path.join(data_dir, "input_processed.txt")
    output_processed_file = os.path.join(data_dir, "output_processed.txt")
    preprocess_data(input_file, input_processed_file, output_processed_file)

if __name__ == "__main__":
    main()
