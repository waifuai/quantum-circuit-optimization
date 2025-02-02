-- Combine all rows from qc.qc8 and qc.qc5i32f using UNION ALL.
SELECT *
FROM qc.qc8
UNION ALL
SELECT *
FROM qc.qc5i32f;
