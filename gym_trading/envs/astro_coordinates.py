#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Astro
"""
# import numpy as np
import pandas as pd

from math import asin
from datetime import datetime, timedelta

try:
    import swisseph as swe
    def get_planet_coordinates(dates, input_planets_data, delta, n_periods=1):
        #to use current date and time uncomment
        #t=datetime.utcnow()
        
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
        
    
        colomn_names = ["Date"]
        
        
        for key, value in sorted(planet_idx.items(), key=lambda item: item[1]):
            
            if key in input_planets_data.keys():
                value = input_planets_data[key]
            else:
                continue
            for val in sorted(value):
                if val in input_planets_data[key]:
                    colomn_names.append(key + "_" + val)
                else:
                    continue
        
        
        
        
        #or to use a specific date and time
        
        tmp_dict = {"Date_Time" : []}
        for t in dates: 
            tmp_dict["Date_Time"].append(t)
            
            for q in range(n_periods):
          
                for key, ipl in planet_idx.items():
                    
                    if not (key in input_planets_data.keys()):
                        continue
                    
                        
                    jd = swe.julday(t.year,t.month,t.day,t.hour+t.minute/60)
                    rslt = swe.calc_ut(jd, ipl , swe.FLG_EQUATORIAL | swe.FLG_SWIEPH | swe.FLG_SPEED)
                    
                    if ipl == 1: #moon
                        rm = swe.calc_ut(jd, ipl , swe.FLG_EQUATORIAL | swe.FLG_SWIEPH | swe.FLG_SPEED |swe.FLG_TRUEPOS | swe.FLG_RADIANS)
                        sinhp = EARTH_RADIUS / rm[2] / AUNIT
                        moondist=asin(sinhp) / DEGTORAD *3600
                        rslt=(rslt[0], rslt[1], moondist)
                    if (q > 0):
                        ouput_key = str(q) + "_" + key
                    else: 
                        ouput_key = key
                        
                    if ("right asc." in input_planets_data[key]):
                        try:
                            tmp_dict[ouput_key + "_" + "right asc."].append( rslt[0] )
                        except KeyError:
                            tmp_dict[ouput_key + "_" + "right asc."] = [rslt[0]]
                            
                    if ("declination" in input_planets_data[key]):
                        try:
                            tmp_dict[ouput_key + "_" + "declination"].append( rslt[1] )
                        except KeyError:
                            tmp_dict[ouput_key + "_" + "declination"] = [rslt[1]]
                            
                    if ("distance" in input_planets_data[key]):
                        try:
                            tmp_dict[ouput_key + "_" + "distance"].append( rslt[2] )
                        except KeyError:
                            tmp_dict[ouput_key + "_" + "distance"] = [ rslt[2] ]
                    
                
                t = t + delta
        
        
        data_frame = pd.DataFrame(tmp_dict)
        data_frame.set_index("Date_Time", inplace=True)
        
        # data_frame = (data_frame - data_frame.mean()) / data_frame.std()
        return data_frame
    
except ImportError:
    print ("Can not use swisseph package!")
    def get_planet_coordinates(dates, input_planets_data, delta, n_periods=1):
        data_frame = pd.DataFrame({"Date_Time":dates})
        data_frame.set_index("Date_Time", inplace=True)
        return data_frame
     




if __name__ == "__main__":

    input_planets_data = {
        "Mercury" : ["right asc.", "declination", "distance"],
        "Venus"   : ["right asc.", "declination", "distance"],
        "Moon"    : ["right asc.", "declination", "distance"],
        "Mars"    : ["right asc.", "declination", "distance"],
        "Jupiter" : ["right asc.", "declination", "distance"],
        "Saturn"  : ["right asc.", "declination", "distance"],
        "Uranus"  : ["right asc.", "declination", "distance"],
    }
    
    df = pd.read_csv("../../GBPUSD60.csv", parse_dates=[[0,1]], header=None, names=['Date','Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    
    # dates = datetime(2009,9,27,20,6)
    
    
    dates = [date_time.to_pydatetime() for date_time in df["Date_Time"]]  #[df["Date_Time"][0].to_pydatetime()] # 
    delta = (df["Date_Time"][1] - df["Date_Time"][0]).to_pytimedelta()
    # timedelta(weeks=0, days=0, hours=0, minutes=1, seconds=1)
    
    
    n_periods = 3
    data_frame = get_planet_coordinates(dates, input_planets_data, delta, n_periods)
    
    
    for date in data_frame.index:
        print (date)
