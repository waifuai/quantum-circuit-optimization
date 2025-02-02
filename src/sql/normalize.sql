-- Build a dynamic SELECT statement to normalize each statevector column by dividing by 1024.
DECLARE @sql NVARCHAR(MAX) = N'SELECT';

;WITH binaries AS (
  SELECT RIGHT('00000' + CAST(number AS VARCHAR(5)), 5) AS binary
  FROM master..spt_values
  WHERE type = 'P' AND number BETWEEN 0 AND 31
)
SELECT @sql += N' statevector_' + binary + N' / 1024 AS statevector_' + binary + N','
FROM binaries;

-- Remove the trailing comma and complete the query.
SET @sql = LEFT(@sql, LEN(@sql) - 1) + N';';

EXEC sp_executesql @sql;
