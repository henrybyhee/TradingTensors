from datetime import datetime, timedelta
from math import asin

import numpy as np
import pandas as pd
import numba
try:
    import swisseph as swe

    EARTH_RADIUS=6378136.6
    AUNIT=1.49597870691e+11       #au in meters, AA 2006 K6
    DEGTORAD=0.0174532925199433


    #planet order for columns in csv
    # p = '2314567'    
    planet_idx = {
            
                "Mercury" : 2,
                "Venus"   : 3,
                "Moon"    : 1,
                "Mars"    : 4,
                "Jupiter" : 5,
                "Saturn"  : 6,
                "Uranus"  : 7,        
            
            }

    def get_planet_coordinates(dates, input_planets_data, N_PERIODS=0):

        def _swisseph(t, ipl):
            t = t[0]
            jd = swe.julday(t.year,t.month,t.day,t.hour+t.minute/60)
            rslt = swe.calc_ut(jd, ipl , swe.FLG_EQUATORIAL | swe.FLG_SWIEPH | swe.FLG_SPEED)
        
            if ipl == 1: #Moon
                rm = swe.calc_ut(jd, ipl , swe.FLG_EQUATORIAL | swe.FLG_SWIEPH | swe.FLG_SPEED |swe.FLG_TRUEPOS | swe.FLG_RADIANS)
                sinhp = EARTH_RADIUS / rm[2] / AUNIT
                moondist=asin(sinhp) / DEGTORAD *3600
                return rslt[0], rslt[1], moondist

            return rslt[0], rslt[1], rslt[2]

        column_names = ['Date_Time']

        column_names.extend([x for x in planet_idx if x in input_planets_data])

        interval = dates[1] - dates[0]
        
        extended_dates = dates.copy()

        extended_dates.extend(
            [dates[-1] + (i+1)*interval for i in range(0, N_PERIODS)]
            )
        # Extend the dates N_PERIODS forward, so we can predict the future coordinates 

        final_df = pd.DataFrame(index=extended_dates)
        extended_dates = np.vstack(np.asarray(extended_dates))
        
        for planet in column_names[1:]:

            result = np.apply_along_axis( _swisseph, 1, extended_dates, planet_idx[planet]) #Apply Swisseph func on every rows

            if "right asc." in input_planets_data[planet]:
                final_df.loc[:, planet+'_'+"right asc."] = result[:, 0] 
                
            if "declination" in input_planets_data[planet]:
                final_df.loc[:, planet+'_'+"declination"] = result[:, 1]
            
            if "distance" in input_planets_data[planet]:
                final_df.loc[:, planet+'_'+"distance"] = result[:, 2]
            
        # Shift data forward N_PERIODS times
        original_df = final_df.copy()
        for i in range(0, N_PERIODS):
            df_2_join = original_df.shift(-(i+1))
            final_df = final_df.join(df_2_join, rsuffix="_t+{}".format(i+1))

        return final_df.loc[dates, :]

except ImportError:
    print ("Can not use swisseph package!")
    def get_planet_coordinates(dates, input_planets_data, n_periods=1):
        data_frame = pd.DataFrame({"Date_Time":dates})
        data_frame.set_index("Date_Time", inplace=True)
        return data_frame    



if __name__ == "__main__":
    df = pd.read_csv("./GBPUSD60.csv", parse_dates=[[0,1]], header=None, names=['Date','Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
    dates = [date_time.to_pydatetime() for date_time in df["Date_Time"]] 



    in_ = {
            #"Mercury" : ["right asc.", "declination", "distance"],
            #"Venus"   : ["right asc.", "declination", "distance"],
            #"Moon"    : ["right asc.", "declination", "distance"],
            #"Mars"    : ["right asc.", "declination", "distance"],
            #"Jupiter" : ["right asc.", "declination", "distance"],
            #"Saturn"  : ["right asc.", "declination", "distance"],
            "Uranus"  : ["right asc.", "declination", "distance"],
    }
    print (get_planet_coordinates(dates ,in_ , N_PERIODS=2))
    
