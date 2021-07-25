from enigmx.transform.extractor import Extractor

list_stocks=[
 'NMIH', 'SHIP', 'ITRM'
]

Extractor(
    list_stocks = list_stocks,
    start_date = "2020-12-01",
    end_date= "2021-07-22",
    path = '/var/data/data/',
    api_key='c04f66748v6u76cjieag',
    tupleTimeZone = (9,10),
    threads=1
    )
