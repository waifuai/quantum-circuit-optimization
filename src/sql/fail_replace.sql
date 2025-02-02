-- Replace occurrences of 'BLANK' with '1' in the gate_40_Gate_Type column.
-- (Note: This assumes the resulting value is valid; add validation as needed.)
UPDATE qc.qc5b_rand1k_2
SET gate_40_Gate_Type = REPLACE(gate_40_Gate_Type, 'BLANK', '1');
