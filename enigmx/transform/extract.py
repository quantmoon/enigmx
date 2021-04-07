from enigmx.transform.extractor import Extractor

list_stocks=['AAPL']

Extractor(
    list_stocks = list_stocks,
    start_date = "2020-12-04",
    end_date= "2021-03-30",
    path = ".", 
    api_key='c04f66748v6u76cjieag',
    threads=1,
    )	



