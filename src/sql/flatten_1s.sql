-- Flatten the level-1 circuit schema by dynamically generating columns for each gate property.
WITH dynamic_columns AS (
  SELECT GROUP_CONCAT(
         'gates.gate_' || idx ||
         '.Gate_Number AS gate_' || LPAD(idx::text, 2, '0') || '_Gate_Number, ' ||
         'gates.gate_' || idx ||
         '.Gate_Type AS gate_' || LPAD(idx::text, 2, '0') || '_Gate_Type, ' ||
         'gates.gate_' || idx ||
         '.Control AS gate_' || LPAD(idx::text, 2, '0') || '_Control, ' ||
         'gates.gate_' || idx ||
         '.Target AS gate_' || LPAD(idx::text, 2, '0') || '_Target, ' ||
         'gates.gate_' || idx ||
         '.Angle_1 AS gate_' || LPAD(idx::text, 2, '0') || '_Angle_1, ' ||
         'gates.gate_' || idx ||
         '.Angle_2 AS gate_' || LPAD(idx::text, 2, '0') || '_Angle_2, ' ||
         'gates.gate_' || idx ||
         '.Angle_3 AS gate_' || LPAD(idx::text, 2, '0') || '_Angle_3'
         ) AS columns
  FROM generate_series(0, 40) AS idx
)
SELECT statevectors.statevector_00000, dynamic_columns.columns
FROM qc.qc2 AS statevectors
CROSS JOIN dynamic_columns
GROUP BY statevectors.statevector_00000, dynamic_columns.columns;
