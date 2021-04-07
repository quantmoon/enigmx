from enigmx.dbgenerator import databundle_instance
import pyodbc
#databundle_instance("digital-maker-308101:us-central1:pip-sql",'BARS_FEATURES',False,True)
cnxn = pyodbc.connect("Driver={ODBC Driver 17 for SQL Server};Server=34.67.233.155;Database=BARS;Uid=sqlserver;Pwd=J7JA4L0pwz0K56oa;")
print(cnxn)
