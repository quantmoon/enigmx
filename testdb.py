from enigmx.dbgenerator import databundle_instance
import pyodbc
#databundle_instance("digital-maker-308101:us-central1:pip-sql",'BARS_FEATURES',False,True)
cnxn = pyodbc.connect("Driver={ODBC Driver 17 for SQL Server};Server=34.67.4.196;Database=ETFTRICK;Uid=sqlserver;Pwd=quantmoon2021")
print(cnxn)
