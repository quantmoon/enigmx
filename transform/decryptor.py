import zarr 
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import datetime as dt

class Decryptor():

    def __init__(self,stock,path,dates,total = False):
        self.stock = stock
        self.path = path
        self.dates = dates
        self.total = total
        
    def decrypt(self):
        zarrds = self.open_zarr_ga(self.stock,self.path)
        dates = self.extract_date_available_market(self.dates[0],self.dates[-1])
        if self.total == True:
            df = self.load_data(self.stock,self.path,dates)
            return df
        elif self.total == False:
            days = [self.run(zarrds,i) for i in dates]
            return days
        


    
    def trading_calendar():
        """
        Gets trading days based on NYSE Calendar.
        """
        nyse = mcal.get_calendar('NYSE')
        early = nyse.schedule(start_date='2015-01-01', end_date='2021-04-28')
        dts = list(early.index.date)

    #transform as datetime.date() each string date
        return dts

    #Extract date available markets
    def extract_date_available_market(self, 
                                        start_, 
                                        end_, 
                                        trd_cal_= trading_calendar()):
        """
        Match input days with NYSE trading calendar days.
        """

        startDate=dt.datetime.strptime(start_,'%Y-%m-%d')
        endDate=dt.datetime.strptime(end_,'%Y-%m-%d')

        if startDate == endDate:

            list_pre = [startDate.date()]
            date = min(
            trd_cal_, key= lambda x: abs(x - list_pre[0])
                        )

            if date == list_pre[0]:
                idx = [trd_cal_.index(date)]
                return [trd_cal_[idx[0]].strftime('%Y-%m-%d')]         
            else:
                print("No trading days at {}".format(
                startDate.date())
                    )
                sys.exit()

        else:

            date = min(
            trd_cal_, key=lambda x: abs(x - startDate.date())
            )
            idx_1 = trd_cal_.index(date)

            date = min(
            trd_cal_, key=lambda x: abs(x - endDate.date())
            )
            idx_2 = trd_cal_.index(date)

            resulted_dates_range = trd_cal_[idx_1:idx_2+1]

        if len(resulted_dates_range)<1:
            print("No trading days in {} to {}".format(
            startDate.date(), 
            endDate.date())
            )
            sys.exit()
        else:
            return [result_date_.strftime('%Y-%m-%d') for result_date_ in resulted_dates_range]  
    
    def open_zarr_ga(self,stock,path):         
        new_path = path+stock+".zarr"
        zarrds = zarr.open_group(new_path)
        return zarrds

    def run(self,zarrds,date_,drop_dup = False):
        arr = np.array(zarrds.date)
        idx = np.where(arr == date_)[0][0]
        prices =  zarrds.value[idx]
        prices = prices[prices>0]
        volume = zarrds.vol[idx]
        volume = volume[:len(prices)]
        timestamp = zarrds.timestamp[idx]
        timestamp = timestamp[:len(prices)]
        df = pd.DataFrame({
        'ts':timestamp,
        'price':prices,
        'vol':volume,
        })
        if drop_dup == True:
            df = df.drop_duplicates()
        else:
            df = df.copy()
        return df

    def load_data(self,symbol,path,dates,drop_dup = False):
        lista = [] 
        zarrds = self.open_zarr_ga(symbol,path)
        for date in dates:
            X = self.run(zarrds,date,drop_dup = drop_dup)
            new_ts = [dt.datetime.fromtimestamp(i) for i in X['ts']/1000]
            X['ts'] = new_ts#ts_idx
            X.set_index('ts',inplace=True)
            lista.append(X)
        result = pd.concat(lista)
        return result

