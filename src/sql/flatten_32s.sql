-- Step 1: Build dynamic SELECT for statevectors (numbers 0 through 31)
WITH RECURSIVE numbers AS (
  SELECT 0 AS n
  UNION ALL
  SELECT n + 1 FROM numbers WHERE n < 31
)
SELECT @statevector_select := GROUP_CONCAT(
  CONCAT('statevectors.statevector_', LPAD(n, 5, '0'))
  SEPARATOR ', '
)
FROM numbers;

-- Step 2: Build dynamic SELECT for gate properties (gate numbers 0 through 40)
WITH RECURSIVE gate_numbers AS (
  SELECT 0 AS gate_num
  UNION ALL
  SELECT gate_num + 1 FROM gate_numbers WHERE gate_num < 40
)
SELECT @gates_select := GROUP_CONCAT(
  CONCAT(
    'gates.gate_', gate_num, '.Gate_Number AS gate_', LPAD(gate_num, 2, '0'), '_Gate_Number, ',
    'gates.gate_', gate_num, '.Gate_Type AS gate_', LPAD(gate_num, 2, '0'), '_Gate_Type, ',
    'gates.gate_', gate_num, '.Control AS gate_', LPAD(gate_num, 2, '0'), '_Control, ',
    'gates.gate_', gate_num, '.Target AS gate_', LPAD(gate_num, 2, '0'), '_Target, ',
    'gates.gate_', gate_num, '.Angle_1 AS gate_', LPAD(gate_num, 2, '0'), '_Angle_1, ',
    'gates.gate_', gate_num, '.Angle_2 AS gate_', LPAD(gate_num, 2, '0'), '_Angle_2, ',
    'gates.gate_', gate_num, '.Angle_3 AS gate_', LPAD(gate_num, 2, '0'), '_Angle_3'
  )
  SEPARATOR ', '
)
FROM gate_numbers;

-- Step 3: Combine the dynamic SELECT parts and execute the final query
SET @sql = CONCAT(
  'SELECT ', @statevector_select, ', ', @gates_select,
  ' FROM (',
  '  SELECT * FROM qc.qc5 ',
  '  UNION ALL SELECT * FROM qc.qc5e ',
  '  UNION ALL SELECT * FROM qc.qc5g',
  ') AS statevectors ',
  'JOIN qc.gates AS gates ON statevectors.id = gates.id'
);

PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
