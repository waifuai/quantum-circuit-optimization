# SQL

This directory contains SQL queries.

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
