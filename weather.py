from settings import *

class OutdoorTemp():
    def __init__(self): 

        # summer 
        self.o_temp_summer_wo_noise = MIN_TEMP_SUMMER + (MAX_TEMP_SUMMER - MIN_TEMP_SUMMER) * np.sin(np.pi * t /96)**2
        self.o_temp_summer = [(MIN_TEMP_SUMMER + (MAX_TEMP_SUMMER - MIN_TEMP_SUMMER) * np.sin(np.pi * x /96)**2) + random.uniform(-0.5,0.5) for x in t]

        # winter 
