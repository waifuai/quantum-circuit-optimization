# SQL

This directory contains historical SQL queries that were used during the cloud-based phase of the project. Since refactoring to a local, Cirq-based implementation, these queries have been superseded but are maintained here for reference.

## Files

- **blank_gates.sql**  
  Replaces empty gate types with 'BLANK' to handle missing values.
- **combine_tables.sql**  
  Merges data from multiple tables into a single result set.
- **fail_replace.sql**  
  Attempts (without validation) to replace 'BLANK' with '1' in the circuit data.
- **flatten_1s.sql**  
  Dynamically flattens the dataset schema (level-1 circuits) to extract detailed gate properties.
- **flatten_32s.sql**  
  Dynamically flattens the dataset schema (level-32 circuits) with dynamic column generation.
- **normalize.sql**  
  Normalizes state vector columns by dividing each by 1024.
