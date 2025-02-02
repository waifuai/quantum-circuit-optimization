-- Replace empty gate types with 'BLANK' in the qc.qc2c table for Gate_ID values between 7 and 40.
UPDATE qc.qc2c
SET Gate_Type = 'BLANK'
WHERE Gate_Type = ''
  AND Gate_ID BETWEEN 7 AND 40;
