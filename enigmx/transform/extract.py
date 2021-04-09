from enigmx.transform.extractor import Extractor

list_stocks=['AFL', 'AGCO', 'AGI', 'AGIO']

Extractor(
    list_stocks = list_stocks,
    start_date = "2020-07-01",
    end_date= "2020-09-10",
    path = ".", 
    api_key='c04f66748v6u76cjieag',
    threads=2,
    )	



