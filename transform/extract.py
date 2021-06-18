from enigmx.transform.extractor import Extractor

list_stocks=['AMC']

Extractor(
    list_stocks = list_stocks,
    start_date = "2021-02-01",
    end_date= "2021-05-31",
    path = '/var/data/data/', 
    api_key='c04f66748v6u76cjieag',
    threads=1
    )	


