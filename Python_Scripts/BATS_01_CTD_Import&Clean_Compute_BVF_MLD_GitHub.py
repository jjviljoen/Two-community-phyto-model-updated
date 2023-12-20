"""
BATS: CTD Import & Clean Original CTD data

@author: Johan Viljoen - j.j.viljoen@exeter.ac.uk
"""

#%%

### LOAD PACKAGES ###
#General Python Packages
import pandas as pd # data analysis and manipulation tool
import numpy as np # used to work with data arrays
import seawater as sw # Import CSIRO seawater package
import holteandtalley # Import MLD code available here https://github.com/garrettdreyfus/python-holteandtalley
from datetime import date
from dateutil import relativedelta
from PyAstronomy import pyasl # used to compute decimal year from DateTime & back. Info: https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/decimalYear.html
from holteandtalley import HolteAndTalley
from math import nan
from matplotlib import pyplot as plt
from scipy import interpolate # used to interpolate profiles for contour plots and computing BVF stratification index

#%%

### DEFINE ALL FUNCTIONS ###

def date_span( start, end ):
    """
    Calculate the difference in years, months, and days between two dates.
    """
    # Calculate the relative delta between the start and end dates
    rd = relativedelta.relativedelta(pd.to_datetime(end), pd.to_datetime(start))
    # Construct the string representing the duration in years, months, and days
    date_len = '{}y{}m{}d'.format(rd.years, rd.months, rd.days)
    # Return the formatted date length string
    return date_len

#%%
### READ & CLEAN - original BATS CTD data ###

# Original csv data from the BCO-DMO project site on BATS at https://www.bco-dmo.org/dataset/3918
# Data downloaded on 9 April 2023 and converted to CSV

# name file location and csv file name
filename1 = 'data/bats_ctd_BCO_DMO.csv' # change according to file location


#Read csv as dataframe
ctd = pd.read_csv(filename1, index_col = 0) # 3044923 rows
ctd.drop('index', axis=1, inplace = True)
#Inspect CTD data
ctd.info()

# Rename Columns
ctd.rename(columns={"CruiseCastID": "cruise_ID",
                                "Latitude_N": "lat",
                                "Decimal_Year": "Dec_Year",
                                "Longitude_W": "lon",
                                "Pressure_dbar": "pressure",
                                "Depth_m": "depth",
                                "Temperature_ITS90": "temperature",
                                "Salinity_psu": "salinity",
                                "Oxygen_umol_Kg": "dissolved_oxygen",
                                "Flu_rfu": "fluorescence",
                                "PAR_uE_m2": "PAR"},inplace=True)

ctd.head()

# Remove rows with depths below 500m
ctd = ctd[ctd["depth"]<501] #

# Replace empty cells of "-999" with python NaNs
ctd.replace(-999, np.nan, inplace = True)
 
# Remove rows with NaN temperature & salinity values
columns_to_check = ['temperature', 'salinity']
ctd.dropna(subset=columns_to_check, inplace=True)

# Convert Decimal year to DateTime using pyasl.decimalYearGregorianDate function from the PyAstronomy package
ctd['time'] = pd.to_datetime(np.vectorize(pyasl.decimalYearGregorianDate)(ctd['Dec_Year'], "datetime")) #np.vectorize similar to for loop

ctd['time'] = pd.DatetimeIndex(ctd['time'])

#Extract and reorder required columns
required_columns = ["cruise_ID", "Dec_Year", 'time', 'lat', 'lon', 'pressure',
                    'depth', 'temperature', 'salinity', 'dissolved_oxygen',
                    'fluorescence', 'PAR']
ctd = ctd[required_columns]

#Convert LonW to LonE
ctd['lon'] *= -1
ctd['lon'].head()

### Convert and separate datetime data into new additional columns ###

# convert to datetime format & extract Date yyyy-mm-dd into new column called "Date"
ctd['Date'] = pd.to_datetime(ctd['time']).dt.date
#print(ctd['Date'])

# convert to datetime format & extract year into new column called "yyyy"
ctd['yyyy'] = pd.to_datetime(ctd['time']).dt.year 
#print(ctd['yyyy'])

# convert to datetime format & extract month into new column called "mm"
ctd['mm'] = pd.to_datetime(ctd['time']).dt.month
#print(ctd['mm'])

# Sort Dataframe
ctd.sort_values(by=['time','depth'], inplace=True)
# Reset bottle df index removing old index with missing numbers after slice
ctd.reset_index(drop=True, inplace = True)

# Store DateTime as numpy array
ctd_DateTime = pd.DatetimeIndex(ctd['time'])

ctd_date = ctd['Date']
#ctd_date = pd.to_datetime(ctd_DateTime).date

# Print start and end dates of bottle data
print("CTD Dates: "+str(min(ctd_date))+" to "+str(max(ctd_date)))

# Print period timespan of bottle data using base date subtraction - only days
print("CTD Date Length: "+str(max(ctd_date)-min(ctd_date)))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
ctd_date_length = date_span(min(ctd_date), max(ctd_date))
print("Timespan: "+str(ctd_date_length))

print(np.max(ctd['Dec_Year']))

#%%

### EXTRACT PROFILES IN BOX AROUND BATS & AFTER 1990 ###

ctd.info()
ctd.head()

#Create copy of df to calculate measurements lost/removed
ctd_x = ctd.copy()

#date(1989,12,31)
#'1989-12-31'
# Remove profiles before 1989
ctd = ctd[ctd["Date"]>date(1989,12,31)]

# Filter stations for 0.25 margin around BATS
bats_lat = 31.67
bats_lon = -64.17
margin = 0.25
lat_min, lat_max = bats_lat - margin, bats_lat + margin
lon_min, lon_max = bats_lon - margin, bats_lon + margin
# Slice df for 0.25deg box aorund BATS
ctd = ctd[(ctd["lat"] >= lat_min) & (ctd["lat"] <= lat_max)
                           & (ctd["lon"] >= lon_min) & (ctd["lon"] <= lon_max)]
# Sort df again
ctd = ctd.sort_values(by=['time','depth'])

# Reset bottle df index replacing old index column
ctd = ctd.reset_index(drop=True)

#Calculate measurements lost
prof_lost = len(ctd_x) - len(ctd)
print("CTD Measurements Lost = "+str(prof_lost))

#%%

### REMOVE CTD PROFILES < 6 TEMP MESUREMENTS ###

# Remove CTD profiles with no surface measurements
grouped = ctd.groupby('cruise_ID')
ctd = grouped.filter(lambda x: (x['depth'].min() <= 11) and (len(x) >= 11))

# Re-sort and reset the index
ctd = ctd.sort_values(by=['time', 'depth']).reset_index(drop=True)


# Test and inspect new df 
ctd[["cruise_ID","temperature"]].groupby("cruise_ID").count()

#ctd.to_csv('data/BATS_CTD_01.csv')

#%%

### Extract required data from CTD dataframe into numpy arrays ###
ctd_time      = ctd.loc[:,'time'].to_numpy()
ctd_Decimal_year = ctd.loc[:,'Dec_Year'].to_numpy()
ctd_date      = ctd.loc[:,'Date'].to_numpy()
time_year     = ctd.loc[:,'yyyy'].to_numpy()
lat           = ctd.loc[:,'lat'].to_numpy()
lon           = ctd.loc[:,'lon'].to_numpy()
pressure      = ctd.loc[:,'pressure'].to_numpy()
depth         = ctd.loc[:,'depth'].to_numpy()
temperature   = ctd.loc[:,'temperature'].to_numpy()
salinity      = ctd.loc[:,'salinity'].to_numpy()
fluorescence  = ctd.loc[:,'fluorescence'].to_numpy()
PAR           = ctd.loc[:,'PAR'].to_numpy()
#beam          = ctd.loc[:,'beam_attenuation_coefficient'].to_numpy()
ID_ctd        = ctd.loc[:,'cruise_ID'].to_numpy()
time_2        = pd.to_datetime(ctd['time']) # panda series
ctd_DateTime  = pd.DatetimeIndex(time_2)

# Extract cruise_ID
ID_list_ctd = ctd['cruise_ID'].values
# Removes Duplicates
ID_list_ctd = pd.unique(pd.Series(ID_list_ctd)) # ID_list_ctd now = x1 ID cruise number per CTD profile

# Compare length of ID_list to all cells containing cruise/profile IDs
print(len(ID_list_ctd))
print(len(ID_ctd))   

### COUNT CTD PROFILES PER YEAR ###

#Create new df with number of CTD profiles (cruises) per year
ctd_y = ctd[["cruise_ID", "yyyy"]].groupby("yyyy").nunique()
print(ctd_y)

# Nice Table for Notebook
print(ctd_y.to_markdown())
#print(tabulate(ctd_y,headers="keys"))

#%%

### COMPUTE VARIABLES NEEDED FROM CTD DATA ###
# Variables: MLD, density, BVF and Kd

# Assuming 'sw.dens0' operates element-wise on arrays
density = sw.dens0(salinity, temperature)

# Add density to the existing CTD DataFrame
ctd["density"] = density

#%%
### COMPUTE Mixed Layer Depth & CTD Prof Meta Data###

### Compute MLD for each CTD profile ###
#Compute Mixed Layer Depth using Holt and Tally method

# Create emty arrays to store calculated MLD and single time/date per profile
MLD              = np.empty(len(ID_list_ctd))+nan
ctd_time_prof    = np.empty(len(ID_list_ctd), dtype='datetime64[ns]')
ctd_DecYear_prof = np.empty(len(ID_list_ctd))+nan
ctd_lat_prof     = np.empty(len(ID_list_ctd))+nan
ctd_lon_prof     = np.empty(len(ID_list_ctd))+nan

# Loop for MLD, time, Decimal year and lat/lon per profile
count = 0
for i in ID_list_ctd:
    asx = np.where(ID_ctd == i)
    A = depth[asx]
    B = temperature[asx]
    #S = salinity[asx]
    #D = density[asx]
    C = ctd_time[asx]
    C = C[0]
    E = ctd_Decimal_year[asx]
    E = E[0]
    lat_1 = lat[asx]
    lat_1 = lat_1[0]
    lon_1 = lon[asx]
    lon_1 = lon_1[0]
    ctd_time_prof[count]     = C
    ctd_DecYear_prof[count]  = E
    ctd_lat_prof[count] = lat_1
    ctd_lon_prof[count] = lon_1  
    if len(A) >= 10: 
        if np.min(A) <= 20:       # This part prevents calculating MLD for profiles where surface data not available as below.
            h = HolteAndTalley(A,B) #h = HolteAndTalley(pressures,temperaturess,salinities,densities)
            MLD[count] = h.tempMLD
            #h = HolteAndTalley(A,B,S,D)
            #MLD[count] = h.densityMLD
    count = count + 1
    
plt.plot(ctd_time_prof,MLD)
plt.show()

ctd_DateTime_prof = pd.DatetimeIndex(ctd_time_prof, dtype='datetime64[ns]', name='date_time', freq=None)

#Plot rough scatter map of profile locations
plt.scatter(ctd_lon_prof,ctd_lat_prof)
plt.show()

#%%
### COMPUTE BVF ###
# Compute Brunt–Väisälä (buoyancy) frequency (BVF)
bvf =  np.empty(len(temperature))+nan #Consider renaming to BVF
for i in ID_list_ctd:
    A = pressure[ctd.cruise_ID == i]
    B = temperature[ctd.cruise_ID == i]
    C = salinity[ctd.cruise_ID == i]
    D = lat[ctd.cruise_ID == i]
    if len(A) >= 10: 
        if np.min(A) <= 20:    
            BRUNT_T = sw.bfrq(C, B, A, D[0])
            BRUNT_T1 = BRUNT_T[0]
            BRUNT_T1 = np.resize(BRUNT_T1, len(BRUNT_T1))
            BRUNT_T3 = BRUNT_T[2]
            BRUNT_T3 = np.resize(BRUNT_T3, len(BRUNT_T3))
            interpfunc = interpolate.interp1d(BRUNT_T3, BRUNT_T1, kind='linear', fill_value="extrapolate")
            xxx = interpfunc(A)
            bvf[ctd.cruise_ID == i] = xxx

# Add BVF to existing CTD data frame
ctd["BVF"] = bvf
#ctd.head()

#%%

### SAVE DFs TO CSV ###

# Save CTD data df to csv
ctd.to_csv('data/BATS_CTD_Cleaned.csv')


#%%
### SAVE MLD AND single CTD values to DF ###

# df with all single CTD values
CTDpd = pd.DataFrame()
CTDpd["ID_list"] = ID_list_ctd
CTDpd["Time"]    = ctd_DateTime_prof
CTDpd["lat"]     = ctd_lat_prof
CTDpd["lon"]     = ctd_lon_prof
CTDpd["MLD"]     = MLD
CTDpd['Date']    = ctd_DateTime_prof.date # extract Date yyyy-mm-dd from DatetimeIndex
CTDpd['DecYear'] = ctd_DecYear_prof

CTDpd.info()

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
ctd_date_length = date_span(min(CTDpd['Date']), max(CTDpd['Date']))
print("Timespan: "+str(ctd_date_length))
print("Min Date: "+str(min(CTDpd['Date'])))
print("Max Date: "+str(max(CTDpd['Date'])))

# Save CTD prof data
CTDpd.to_csv('data/BATS_CTD_profData.csv')


#%%

### READ CLEANED CTD DATA FROM CSV for Example profile plots ###

# CSV filename
filename_1 = 'data/BATS_CTD_Cleaned.csv'
# Load data from csv. "index_col = 0" make first column the index.
ctd        = pd.read_csv(filename_1, index_col = 0)

ctd.info()

### Extract required data from CTD dataframe into numpy arrays ###
ctd_time      = ctd.loc[:,'time'].to_numpy()
ctd_date      = ctd.loc[:,'Date'].to_numpy()
time_year     = ctd.loc[:,'yyyy'].to_numpy()
lat           = ctd.loc[:,'lat'].to_numpy()
lon           = ctd.loc[:,'lon'].to_numpy()
depth         = ctd.loc[:,'depth'].to_numpy()
temperature   = ctd.loc[:,'temperature'].to_numpy()
salinity      = ctd.loc[:,'salinity'].to_numpy()
density       = ctd.loc[:,'density'].to_numpy()
bvf           = ctd.loc[:,'BVF'].to_numpy()
fluorescence  = ctd.loc[:,'fluorescence'].to_numpy()
doxy          = ctd.loc[:,'dissolved_oxygen'].to_numpy()
ID_ctd        = ctd.loc[:,'cruise_ID'].to_numpy()
time_2        = pd.to_datetime(ctd['time']) # panda series
ctd_Decimal_year = ctd.loc[:,'Dec_Year'].to_numpy()
ctd_DateTime  = pd.DatetimeIndex(time_2)

### Cruise ID list for CTD ###
# Extract cruise_ID
ID_list_ctd = ctd['cruise_ID'].values

# Removes Duplicates
ID_list_ctd = pd.unique(pd.Series(ID_list_ctd)) # ID_list_ctd now = x1 ID cruise number per CTD profile

# Compare length of ID_list to all cells containing cruise/profile IDs
print(len(ID_list_ctd))
print(len(ID_ctd))

### Read CTD prof data ###

# CSV filename
filename_2 = 'data/BATS_CTD_profData.csv'
# Load data from csv. "index_col = 0" make first column the index.
ctd_prof = pd.read_csv(filename_2, index_col = 0)
# Inspect ctd_prof df
ctd_prof.info()
#ctd_prof.head()

# Extract required data from df
ctd_DateTime_prof = pd.DatetimeIndex(ctd_prof['Time'])
ctd_date_prof     = ctd_prof.loc[:,'Date'].to_numpy()
MLD               = ctd_prof.loc[:,'MLD'].to_numpy()

#%% 

### PLOT EXAMPLE PROFILE CTD DATA ###

PROFILE_ID = 10195003#10195003#10309008#30042002#10016004#10015002 #10017006#10070004

# Supplemental Example Profile: 10195003

# Extract all CTD rows of this profile into new df to print
ctd_prof = ctd[ctd['cruise_ID'] == PROFILE_ID]
print(ctd_prof)

# find indexes of ctd ID for the selected profile
AS = np.where(ID_ctd == PROFILE_ID)
# Extract variables of specific profile
depth_1        = depth[AS]
temperature_1  = temperature[AS]
salinity_1     = salinity[AS]
fluorescence_1 = fluorescence[AS]
PAR_1          = PAR[AS]
density_1      = density[AS]
bvf_1          = bvf[AS]
time_date_1    = ctd_DateTime[AS]
doxy_1         = doxy[AS]

# Find index of single value variables of specific variables
ASD = np.where(ID_list_ctd == PROFILE_ID)
# Extract single value variables
MLD_TEMP = MLD[ASD]

#Figure parameters that can be changed
XSIZE           = 16    #Define the xsize of the figure window
YSIZE           = 12    #Define the ysize of the figure window
Title_font_size = 18    #Define the font size of the titles
Label_font_size = 15    #Define the font size of the labels
TEMP_COLOUR     = 'r'   #Temperature colours see https://matplotlib.org/2.0.2/api/colors_api.html
PSAL_COLOUR     = 'b'   #Salinity colours see https://matplotlib.org/2.0.2/api/colors_api.html
DOXY_COLOUR     = 'c'   #Dissolved Oxy colours see https://matplotlib.org/2.0.2/api/colors_api.html
FLUO_COLOUR     = 'g'   #Chla colours see https://matplotlib.org/2.0.2/api/colors_api.html
Cp_COLOUR       = 'm'   #backscattering colours see https://matplotlib.org/2.0.2/api/colors_api.html
PAR_COLOUR      = 'y' 
DENSITY_COLOUR  = 'k' 
BRUNT_COLOUR    = 'orange'

# Define the figure window including 5 subplots orientated horizontally
fig, ([ax1, ax2, ax3], [ax4, ax5, ax7]) = plt.subplots(2,3, sharey=True, figsize=(XSIZE,YSIZE), \
gridspec_kw={'hspace': 0.2})
    
fig.tight_layout()
    
st = fig.suptitle("CTD-ID: "+str(PROFILE_ID)+" DateTime: "+str(time_date_1[0]),\
          fontsize=20,
          color="k")
st.set_y(0.92)

# Temperature subplot
ax1.plot(temperature_1,depth_1, \
         color = TEMP_COLOUR, marker = 'o', linestyle = 'None')
ax1.plot([np.min(temperature_1)-0.5,np.max(temperature_1)+0.5],[MLD_TEMP,MLD_TEMP],
         color = 'g', marker = 'None', linestyle = '--', label = 'MLD(T) Holt')
ax1.set_ylabel('Depth (m)', fontsize=Label_font_size)
ax1.yaxis.set_tick_params(labelsize=Label_font_size)
ax1.set_ylim([300,0]) 
ax1.set_xlabel('Temperature ($^o$C)', fontsize=Title_font_size, color = TEMP_COLOUR)
ax1.xaxis.set_tick_params(labelsize=Label_font_size)
ax1.set_xlim(xmin=np.min(temperature_1)-0.5, xmax=np.max(temperature_1)+0.5)
ax1.xaxis.set_major_locator(plt.MaxNLocator(2))
ax1.legend(loc="lower right", fontsize=12)

# Salinity subplot
ax2.plot(salinity_1,depth_1, \
         color = PSAL_COLOUR, marker = 'o', linestyle = 'None')
# ax2.set_ylabel('Depth (m)', fontsize=15)
ax2.yaxis.set_tick_params(labelsize=Label_font_size)
ax2.set_ylim([300,0]) 
ax2.set_xlabel('Salinity (PSU)', fontsize=Title_font_size, color = PSAL_COLOUR)
ax2.xaxis.set_tick_params(labelsize=Label_font_size)
ax2.set_xlim(xmin=np.min(salinity_1)-0.05, xmax=np.max(salinity_1)+0.05)
ax2.xaxis.set_major_locator(plt.MaxNLocator(2))

# Density subplot
ax3.plot(density_1,depth_1, \
         color = DENSITY_COLOUR, marker = 'o', linestyle = 'None')
# ax5.set_ylabel('Depth (m)', fontsize=15)
ax3.yaxis.set_tick_params(labelsize=Label_font_size)
ax3.set_ylim([300,0])
ax3.set_xlabel('Density', fontsize=Title_font_size, color = DENSITY_COLOUR)
ax3.xaxis.set_tick_params(labelsize=Label_font_size)
ax3.set_xlim(xmin=np.min(density_1)-0.05, xmax=np.max(density_1)+0.05)
ax3.xaxis.set_major_locator(plt.MaxNLocator(2))

# Fluorescence subplot
ax4.plot(fluorescence_1,depth_1, \
         color = FLUO_COLOUR, marker = 'o', linestyle = 'None')
# ax3.set_ylabel('Depth (m)', fontsize=15)
ax4.yaxis.set_tick_params(labelsize=Label_font_size)
ax4.set_ylim([300,0]) 
ax4.set_xlabel('Fluorescence (RFU)', fontsize=Title_font_size, color = FLUO_COLOUR)
ax4.xaxis.set_tick_params(labelsize=Label_font_size)
ax4.set_xlim(xmin=np.min(fluorescence_1)-0.05, xmax=np.max(fluorescence_1)+0.05)
ax4.xaxis.set_major_locator(plt.MaxNLocator(2))

# Dissolved Oxygen subplot
ax5.plot(doxy_1,depth_1, \
         color = DOXY_COLOUR, marker = 'o', linestyle = 'None') 
ax5.set_xlabel('DOXY (micro mol kg$^{-3}$)', fontsize=Title_font_size, color= DOXY_COLOUR)
ax5.xaxis.set_tick_params(labelsize=Label_font_size)
ax5.set_ylabel('Depth (m)', fontsize=Label_font_size)
ax5.yaxis.set_tick_params(labelsize=Label_font_size)
#ax5.set_xlim(xmin=np.nanmin(doxy_1)-10,xmax=np.nanmax(doxy_1)+10)
ax5.xaxis.set_major_locator(plt.MaxNLocator(2))
#ax5.legend(loc="lower right", fontsize=12)
 
# BVF subplot  
ax7.plot(bvf_1,depth_1, \
         color = BRUNT_COLOUR, marker = 'o', linestyle = 'None')
ax7.plot([np.min(bvf_1)-0.0005,np.max(bvf_1)+0.0005],[MLD_TEMP,MLD_TEMP],color = 'g', marker = 'None', linestyle = '--', label = 'MLD(T) Holt')
# ax6.set_ylabel('Depth (m)', fontsize=15)
ax7.yaxis.set_tick_params(labelsize=Label_font_size)
ax7.set_ylim([300,0])
ax7.set_xlabel('BVF', fontsize=Title_font_size, color = BRUNT_COLOUR)
ax7.xaxis.set_tick_params(labelsize=Label_font_size)
ax7.set_xlim(xmin=np.min(bvf_1)-0.000005, xmax=np.max(bvf_1)+0.000005)
ax7.xaxis.set_major_locator(plt.MaxNLocator(2))
ax7.legend(loc="lower right", fontsize=12)

fig.savefig('plots/BATS-CTD_panel_'+str(PROFILE_ID)+'.png', format='png', dpi=300, bbox_inches="tight")
    
#Complete the plot
plt.show()
#plt.close(fig) # close the figure window