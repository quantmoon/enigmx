from enigmx.transform.extractor import Extractor

list_stocks=[
'AEZS', 'EEFT', 'FND', 'LPRO', 'SNCY', 
'PCT', 'EXEL', 'CIEN', 'PAAS', 'SPKE', 'OSG', 'LHCG', 'NUS', 'RMR', 'YNDX', 'PLNT', 'CLGX', 'GWRS', 
'EXTN', 'REYN',
'FIZZ', 'SENS', 'LAUR', 'HCSG', 'VZ', 'SVM', 'SPPI', 'BRSP', 'KZR', 'IMTX', 'STMP', 'PINC', 'BMEA', 'JELD',
'ACCD', 'INDT', 'RIGL', 'DTEA', 'PDCO', 'CHEK', 'HOOK', 'CWT', 'UNF', 'ALX', 'TPHS', 'USWS', 'ENR',
'RETA', 'BLX', 'SWIM', 'EIX', 'GTE', 'SVRA', 'ACA', 'STSA', 'BIO', 'AHT', 'ACIU', 'WING', 'AEIS', 'AES',
'IBP', 'CNI', 'LE', 'AGFS', 'FLWS', 'FTCI', 'MDGL', 'DXCM', 'MCHX', 'CSPR', 'TFX'

]

Extractor(
    list_stocks = list_stocks,
    start_date = "2020-12-01",
    end_date= "2021-07-22",
    path = '/var/data/data/',
    api_key='c04f66748v6u76cjieag',
    tupleTimeZone=(9,10),
    threads=1
    )
