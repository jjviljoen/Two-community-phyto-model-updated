"""
BATS - Inspect & Clean Bottle POC Data

@author: Johan Viljoen - j.j.viljoen@exeter.ac.uk
"""

#%%

### LOAD PACKAGES ###
#General Python Packages
import pandas as pd # data analysis and manipulation tool
import numpy as np # used to work with data arrays
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
from datetime import timedelta, date
from dateutil import relativedelta
from PyAstronomy import pyasl # used to compute decimal year from DateTime & back. Info: https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/decimalYear.html
from math import nan
from bisect import bisect

# Supress
import warnings
warnings.filterwarnings("ignore") # Added to remove the warning "UserWarning: Warning: 'partition' will ignore the 'mask' of the MaskedArray." on 2nd to last cell of code

# Supresses outputs when trying to "divide by zero" or "divide by NaN" or "treatment for floating-point overflow"
np.seterr(all='ignore');

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

def median_zscore_outlier_detection(data, depths, window_size=3, threshold=3.0, replace_with_nans=True):
    # Initialize empty arrays for filtered data and depth
    filtered_data = np.full_like(data, np.nan)
    filtered_depths = np.full_like(depths, np.nan)

    # Loop through the data using the moving window
    for i in range(len(data)):
        start_idx = max(0, i - (window_size - 1) // 2)
        end_idx = min(len(data), i + (window_size + 1) // 2)

        # Extract the data within the moving window
        window_data = data[start_idx:end_idx]

        # Calculate the median and the median absolute deviation (MAD)
        median = np.median(window_data)
        mad = np.median(np.abs(window_data - median))

        # Calculate the Z-score based on the MAD
        z_score = 0.6745 * (data[i] - median) / (1 * mad)
        #z_score = 0.6745 * (data[i] - median) / (1.4826 * mad)

        # Replace or remove outliers based on the specified option
        if np.abs(z_score) <= threshold:
            filtered_data[i] = data[i]
            filtered_depths[i] = depths[i]

    if replace_with_nans:
        return filtered_data, filtered_depths
    else:
        non_nan_mask = ~np.isnan(filtered_data)
        return filtered_data[non_nan_mask], filtered_depths[non_nan_mask]

#%%

### READ & EXTRACT CTD DATA ###

### Read/Import cleaned CTD data from CSV
# CSV filename
filename_1 = 'data/BATS_CTD_Cleaned.csv'
# Load data from csv. "index_col = 0" make first column the index.
ctd        = pd.read_csv(filename_1, index_col = 0)

### Extract required data from CTD dataframe into numpy arrays ###
ctd_time      = ctd.loc[:,'time'].to_numpy()
ctd_date      = ctd.loc[:,'Date'].to_numpy()
time_year     = ctd.loc[:,'yyyy'].to_numpy()
depth         = ctd.loc[:,'depth'].to_numpy()
ID_ctd        = ctd.loc[:,'cruise_ID'].to_numpy()
ctd_Decimal_year = ctd.loc[:,'Dec_Year'].to_numpy()
ctd_DateTime  = pd.to_datetime(ctd['time'].values)

### Cruise ID list for CTD ###
# Extract cruise_ID & Removes Duplicates
ID_list_ctd = pd.unique(ID_ctd) # ID_list_ctd now = x1 ID cruise number per CTD profile

# Compare length of ID_list to all cells containing cruise/profile IDs
print(len(ID_list_ctd))
print(len(ID_ctd))

### Read CTD prof data ###
# CSV filename
filename_2 = 'data/BATS_CTD_profData.csv'
# Load data from csv. "index_col = 0" make first column the index.
ctd_prof = pd.read_csv(filename_2,index_col = 0)
# Inspect ctd_prof df
ctd_prof.info()

# Extract required data from df
try:
    ctd_DateTime_prof = pd.DatetimeIndex(ctd_prof['Time'])
except: 
    ctd_DateTime_prof = pd.DatetimeIndex(ctd_prof.index)

ctd_prof.set_index(ctd_DateTime_prof, inplace= True)
ctd_prof.info()

ctd_date_prof     = ctd_prof.loc[:,'Date'].to_numpy()
ctd_DecYear_prof  = ctd_prof.loc[:,'DecYear'].to_numpy()
MLD               = ctd_prof.loc[:,'MLD'].to_numpy()

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
ctd_date_length = date_span(min(ctd_prof['Date']), max(ctd_prof['Date']))
print("Timespan: "+str(ctd_date_length))
print("Min Date: "+str(min(ctd_prof['Date'])))
print("Max Date: "+str(max(ctd_prof['Date'])))

#%%

### EXTRACT CLEANED PIGMENT BOTTLE DATA & MAKE PIGMENT BOTTLE ID LIST ###

# CSV filename
filename_1 = 'data/BATS_Pigments_Cleaned.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_6   = pd.read_csv(filename_1, index_col = 0)
# Inspect df
bottle_6.info()

### Extract required data from new bottle_6 dataset ###
b_time     = bottle_6.loc[:,'time'].to_numpy()
b_date     = bottle_6.loc[:,'Date'].to_numpy()
b_depth    = bottle_6.loc[:,'depth'].to_numpy()
b_chla     = bottle_6.loc[:,'pigment14'].to_numpy()
b_ID       = bottle_6.loc[:,'Cruise_ID'].to_numpy()
b_Decimal_year = bottle_6.loc[:,'DecYear'].to_numpy()
# Bottle DateTime data
b_DateTime     = pd.to_datetime(bottle_6['time'].values)

### Cruise ID list for Chla ###
# Extract cruise_ID & Removes Duplicates
ID_list_6 = pd.unique(b_ID)

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_6['Date']), max(bottle_6['Date']))
print("Timespan: "+str(b_date_length)) # 32y10m29d

#%%

######
### READ & CLEAN ORIGINAL POC BOTTLE DATA ###
######

# Original txt Bottle data from the BATS BIOS site Dropbox at https://bats.bios.asu.edu/bats-data/
# Data downloaded on 18 July 2023 and converted to CSV

#Filename
file_pigment2 = 'data/bats_bottle_BIOS.csv' #name file location and csv file name

# Read CSV
bot = pd.read_csv(file_pigment2)
# Inspect BATS Bottle data
bot.info()
bot.head()

# Meta Data Info
# POC = Particulate Organic Carbon (ug/kg)

# Rename Columns
bot.rename(columns={"Id": "cruise_ID",
                                "latN": "lat",
                                "decy": "DecYear",
                                "lonW": "lon",
                                "Depth": "depth"},inplace=True)

bot.head()

# Remove rows with depths below 300m
bot = bot[bot["depth"]<410] # 59669 rows

# Replace empty cells of "-999" with python NaNs
bot.replace(-999, np.nan, inplace = True)

# Remove rows with NaN POC values
bottle_poc = bot.dropna(subset=['POC'], inplace=False) # 8739 rows

# Convert Decimal year to DateTime. # np.vectorize replaces a iterative for loop
bottle_poc['time'] = pd.to_datetime(np.vectorize(pyasl.decimalYearGregorianDate)(bottle_poc['DecYear'], "datetime"))
# Convert Datetime to DatetimeIndex format
bottle_poc['time'] = pd.DatetimeIndex(bottle_poc['time'])
bottle_poc['time'].head()

#Extract and reorder required columns
required_columns = ['time',"cruise_ID", "DecYear",'lat','lon','depth','POC']
bottle_poc = bottle_poc[required_columns]
bottle_poc.head()
bottle_poc.info()

# Removes last 2 digits from ID to make all results in profile the same ID
bottle_poc['cruise_ID'] = bottle_poc['cruise_ID'].astype(str).str[:-2].astype(np.int64)

# sort by time
bottle_poc = bottle_poc.sort_values(by=['time','cruise_ID','depth'])
# Reset bottle df index replacing old index column
bottle_poc = bottle_poc.reset_index(drop=True)

#Convert LonW to LonE
bottle_poc['lon'] = bottle_poc['lon']*-1
bottle_poc['lon'].head()

### Convert and separate datetime data into new additional columns ###

# convert to datetime format & extract year into new column called "yyyy"
bottle_poc['yyyy'] = pd.to_datetime(bottle_poc['time']).dt.year 
#print(bottle['yyyy'])

# convert to datetime format & extract mopnth into new column called "mm"
bottle_poc['mm'] = pd.to_datetime(bottle_poc['time']).dt.month
#print(bottle['mm'])

# convert to datetime format & extract Date yyyy-mm-dd into new column called "Date"
bottle_poc['Date'] = pd.to_datetime(bottle_poc['time']).dt.date
#print(bottle['Date'])

#%%

### EXTRACT PROF IN BOX AROUND BATS & AFTER 1990 ###

bottle_poc.info()

#Create copy of df to calculate measurements lost/removed
bot_x = bottle_poc.copy()

# Remove profiles before 1989
bottle_poc = bottle_poc[bottle_poc["Date"]>date(1989,12,31)]

# Filter stations for 0.25 margin around BATS
bats_lat = 31.67
bats_lon = -64.17
margin = 0.25
lat_min, lat_max = bats_lat - margin, bats_lat + margin
lon_min, lon_max = bats_lon - margin, bats_lon + margin
# Slice df for 0.25deg box aorund BATS
bottle_poc = bottle_poc[(bottle_poc["lat"] >= lat_min) & (bottle_poc["lat"] <= lat_max)
                           & (bottle_poc["lon"] >= lon_min) & (bottle_poc["lon"] <= lon_max)]
# Sort df again
bottle_poc = bottle_poc.sort_values(by=['time','depth'])

# Reset bottle df index replacing old index column
bottle_poc = bottle_poc.reset_index(drop=True)

#Calculate measurements lost
prof_lost = len(bot_x) - len(bottle_poc)
print("Bottle Measurements Lost = "+str(prof_lost))

### SAVE CLEANED BOTTLE DF TO CSV ###

# Write intermediate Cleaned bottle df to csv
#bottle_poc.to_csv('data/BATS_Bottle_Cleaned1.csv')

#%%

### EXPLORE POC DATA ###

# Summary stats Table for POC
print(bottle_poc[["POC"]].describe())

# Boxplot
bottle_poc[["POC"]].plot.box()
plt.show()

# Group by Cruise ID profile and count number of samples per profile
bottle_poc_profcount = bottle_poc[["cruise_ID", "POC"]].groupby("cruise_ID").count()
print(bottle_poc_profcount)

# Count number of profiles per number of depths sampled
print(bottle_poc_profcount.groupby("POC")["POC"].count())

# Bar plot of profiles per number per number of depths sampled
bottle_poc_profcount.groupby("POC")["POC"].count().plot.bar()
plt.show()

### COUNT POC PROFILES PER YEAR ###

# Create new df with number of CTD profiles (cruises) per year
bottle_poc_y = bottle_poc[["cruise_ID", "yyyy"]].groupby("yyyy").nunique()

# Bar plot of profiles per year
bottle_poc_y.plot.bar(color = "darkgreen")
plt.show()

# Nice Table for Notebook
print(bottle_poc_y.to_markdown())

#%%

### EXTRACT CLEANED DATA & MAKE NEW POC ID LIST ###

# Sort new df by time and depth again
bottle_poc = bottle_poc.sort_values(by=['time','cruise_ID','depth'])
# Reset bottle df index removing old index with missing numbers after slice
bottle_poc = bottle_poc.reset_index(drop=True)

### Extract required data from new bottle_6 dataset ###
b2_time     = bottle_poc.loc[:,'time'].to_numpy()
b2_date     = bottle_poc.loc[:,'Date'].to_numpy()
b2_depth    = bottle_poc.loc[:,'depth'].to_numpy()
b2_poc      = bottle_poc.loc[:,'POC'].to_numpy()
b2_ID       = bottle_poc.loc[:,'cruise_ID'].to_numpy()
b2_year     = bottle_poc.loc[:,'yyyy'].to_numpy()
b2_month    = bottle_poc.loc[:,'mm'].to_numpy()

#Convert array object to Datetimeindex type
b2_DateTime = pd.to_datetime(bottle_poc['time'].values)

### Cruise_ID list & Removes Duplicates
ID_list_poc = pd.unique(b2_ID)
print(len(ID_list_poc))

#%%

### MATCH BOTTLE POC DATA WITH PIGMENT PROFILES ###

### REPLACE POC DATA ID WITH PIGMENTS IDs MATCHING DATES

#bottle_poc_x = bottle_poc.copy()

ctd_date_d       = pd.to_datetime(ctd_date)
ctd_time_d       = pd.to_datetime(ctd_time)
b_time_t         = pd.to_datetime(b_time)

b_date_d = pd.to_datetime(b_date).date

ID_list_poc_x = np.array(range(len(ID_list_poc))) + nan
day_range = 0 # look for matching dates of same date
count_replace = 0
count = 0
for i in ID_list_poc:
    poc_idx         = np.where(bottle_poc.cruise_ID == i)
    poc_ID_prof_all = b2_ID[poc_idx]
    prof_ID        = np.unique(b2_ID[poc_idx])
    poc_ID_check    = np.isin(prof_ID,ID_list_6)
    prof_date      = np.unique(b2_date[poc_idx])
    prof_date      = pd.unique(prof_date)
    prof_date      = pd.to_datetime(prof_date).date 
    if poc_ID_check == False:
        date_start     = prof_date - timedelta(days=day_range)
        date_end       = prof_date + timedelta(days=day_range)
        date_list      = pd.date_range(date_start[0],date_end[0],freq='d')
        date_list      = date_list.date
        ind = bisect(b_date_d, prof_date, hi=len(b_time)-1)
        date_nearest   = min(b_date_d[ind], b_date_d[ind-1],key=lambda x: abs(x - prof_date[0]))
        date_check_1 = np.isin(date_nearest,date_list)
        if date_check_1 == True:
            date_check_x    = np.where(b_date_d == date_nearest)
            print("### ID MISSING ###")
            print("POC ID: "+str(prof_ID[0]))
            print(prof_date)
            print("Nearest Bottle date: "+str(date_nearest))
            ID_replace      = np.unique(b_ID[date_check_x])
            ID_replace      = ID_replace[0]
            bottle_poc.loc[pd.to_datetime(bottle_poc["Date"].values).date ==  (prof_date[0]), "cruise_ID"] = ID_replace
            #bottle_poc.loc[pd.to_datetime(bottle_poc["Date"].values).date ==  (prof_date[0]), "Date"] = date_nearest
            count_replace = count_replace+1
            print("REPLACED")
            print("Bottle ID - "+str(ID_replace))
    #print(count)
    count=count+1
print("REPLACED = "+str(count_replace))

#%%

### REMOVE PROFILES WITH NO MATCHING PIGMENT profiles ###

# Create new df containing only data for profiles also in pigment bottle list
bottle_poc =  bottle_poc[bottle_poc.cruise_ID.isin(ID_list_6)]

# Sort new df by ID and depth
#bottle_poc = bottle_poc.sort_values(by=['cruise_ID','depth'])
# Sort new df by ID and depth
bottle_poc = bottle_poc.sort_values(by=['Date','depth'])

# Reset bottle df index replacing old index column
bottle_poc = bottle_poc.reset_index(drop=True)

print(len(bottle_poc))

# Write Cleaned bottle df to csv
bottle_poc.to_csv('data/BATS_Bottle_POC_AllDepths.csv')

#%%

### REMOVE SPIKES FROM POC PROFILES ###

# =============================================================================
# # CSV filename
# filename_1 = 'data/BATS_Bottle_POC_AllDepths.csv'
# # Load data from csv. "index_col = 0" make first column the index.
# bottle_poc   = pd.read_csv(filename_1, index_col = 0)
# =============================================================================

# Remove duplicate measurements
bottle_poc.drop_duplicates(subset=['cruise_ID', 'depth'], keep='last', inplace=True)

# Sort new df by ID and depth
bottle_poc = bottle_poc.sort_values(by=['Date','depth'])

# Reset bottle df index replacing old index column
bottle_poc = bottle_poc.reset_index(drop=True)

### EXTRACT CLEANED DATA & MAKE NEW POC ID LIST ###

### Extract required data from new bottle_6 dataset ###
b2_time     = bottle_poc.loc[:,'time'].to_numpy()
b2_date     = bottle_poc.loc[:,'Date'].to_numpy()
b2_depth    = bottle_poc.loc[:,'depth'].to_numpy()
b2_poc      = bottle_poc.loc[:,'POC'].to_numpy()
b2_ID       = bottle_poc.loc[:,'cruise_ID'].to_numpy()
b2_year     = bottle_poc.loc[:,'yyyy'].to_numpy()
b2_month    = bottle_poc.loc[:,'mm'].to_numpy()
b2_Decimal_year = bottle_poc.loc[:,'DecYear'].to_numpy()

### Cruise_ID list & Removes Duplicates
ID_list_poc = pd.unique(b2_ID)
print(len(ID_list_poc))

#New array to store cleaned POC data
POC_f  = np.array(range(len(b2_poc))) + nan
# Loop Each profile
for ID in ID_list_poc:
    # POC data
    prof_poc_idx    = np.where(bottle_poc.cruise_ID == ID)
    prof_poc        = b2_poc[prof_poc_idx]
    prof_poc_depth  = b2_depth[prof_poc_idx]
    
    # Z-SCORE OUTLIER REMOVAL
    filtered_data, filtered_depths = median_zscore_outlier_detection(prof_poc, prof_poc_depth, window_size=3, replace_with_nans=True)

    
    # Convert the filtered data and depths to arrays
    filtered_data = np.array(filtered_data)
    filtered_depths = np.array(filtered_depths)
    
    POC_f[prof_poc_idx] = filtered_data
    
bottle_poc['POC'] = POC_f
#bottle_poc['POC_f'] = POC_f

# Remove rows with NaN POC values
bottle_poc = bottle_poc.dropna(subset=['POC'], inplace=False)

# Reset bottle df index removing old index with missing numbers after slice
bottle_poc = bottle_poc.reset_index(drop=True)

# Write Cleaned bottle df to csv
bottle_poc.to_csv('data/BATS_Bottle_POC_AllDepths.csv')

#%%

### REMOVE PROFILES WITH <6 MEASUREMENTS ###

# Extract required variables
bottle_poc = bottle_poc.reset_index(drop=True)  # Resetting the index for easier access

poc_no_prof = np.zeros(len(bottle_poc))  # Initialize array

for i in range(len(bottle_poc)):
    mask = (bottle_poc['cruise_ID'] == bottle_poc.at[i, 'cruise_ID'])
    poc = bottle_poc.loc[mask, 'POC'].values
    depth = bottle_poc.loc[mask, 'depth'].values

    if depth.min() > 30: # Ensure profiles have measurements shallower than 30m
        poc_no_prof[i] = 0
    else:
        depth_mask = depth <= 350 # Consider only upper 350m to count measurements
        poc = poc[depth_mask]
        depth = depth[depth_mask]
        non_nan_mask = ~np.isnan(poc)
        poc = poc[non_nan_mask]

        poc_no_prof[i] = np.count_nonzero(poc)

bottle_poc['poc_no_prof'] = poc_no_prof

bottle_poc = bottle_poc[bottle_poc['poc_no_prof'] > 5] #4530 #Keep only profile with 6 or more measurements

################

# Remove rows where depth >= 
bottle_poc = bottle_poc[bottle_poc["depth"] < 410]

# Sort new df by time and depth again
bottle_poc = bottle_poc.sort_values(by=['Date','depth'])

# Reset bottle df index replacing old index column
bottle_poc = bottle_poc.reset_index(drop=True)

bottle_poc.info()

# Group by Cruise ID profile and count number of samples per profile
bottle_poc_profcount = bottle_poc[["cruise_ID", "POC"]].groupby("cruise_ID").count()

# Count number of profiles per number of depths sampled
print(bottle_poc_profcount.groupby("POC")["POC"].count())

#%%

# Save Cleaned POC data

# Write Cleaned bottle df to csv
bottle_poc.to_csv('data/BATS_Bottle_POC.csv')

#%%

### EXTRACT CLEANED DATA & MAKE NEW POC ID LIST ###
# CSV filename
filename_1 = 'data/BATS_Bottle_POC.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_poc   = pd.read_csv(filename_1, index_col = 0)

# Remove duplicate measurements
bottle_poc.drop_duplicates(subset=['cruise_ID', 'depth'], keep='last', inplace=True)

# Sort new df by ID and depth
bottle_poc = bottle_poc.sort_values(by=['Date','depth'])

# Reset bottle df index replacing old index column
bottle_poc = bottle_poc.reset_index(drop=True)

### EXTRACT CLEANED DATA & MAKE NEW POC ID LIST ###

### Extract required data from new bottle_6 dataset ###
b2_time     = bottle_poc.loc[:,'time'].to_numpy()
b2_date     = bottle_poc.loc[:,'Date'].to_numpy()
b2_depth    = bottle_poc.loc[:,'depth'].to_numpy()
b2_poc      = bottle_poc.loc[:,'POC'].to_numpy()
b2_ID       = bottle_poc.loc[:,'cruise_ID'].to_numpy()
b2_year     = bottle_poc.loc[:,'yyyy'].to_numpy()
b2_month    = bottle_poc.loc[:,'mm'].to_numpy()
b2_Decimal_year = bottle_poc.loc[:,'DecYear'].to_numpy()

#Convert array object to Datetimeindex type
b2_DateTime = pd.to_datetime(bottle_poc['time'].values)

### Cruise_ID list
ID_list_poc = pd.unique(b2_ID)
print(len(ID_list_poc))
# 419 profiles with 6 or more POC measurements

### CREATE POC PROF DF

# Create POC df with ID list to save single profile values
bottle_poc_prof = bottle_poc.drop_duplicates(subset=['cruise_ID'])
# Reset bottle df index replacing old index column
bottle_poc_prof = bottle_poc_prof.reset_index(drop=True)

print(len(bottle_poc_prof))
print(len(ID_list_poc))

# Slice df to only have ID, time and date columns
bottle_poc_prof = bottle_poc_prof[['cruise_ID','time','Date','lat','lon','DecYear','yyyy','mm' ]]

bottle_poc_prof.info()

# Write Cleaned bottle df to csv
bottle_poc_prof.to_csv('data/BATS_Bottle_POC_profData.csv')


#%%

### POC SINGLE PROFILE PLOT ###

ID_1 = 	10195003#10030002#20335002#30042002#20280003#20093004#10022003#30030003

# Supplemental Example Profile: 10195003

# Find POC profile data

# Get MLD from CTD
prof_MLD_idx = np.where(ID_list_ctd == ID_1)
prof_MLD     = MLD[prof_MLD_idx]

# POC data
prof_poc_idx     = np.where(bottle_poc.cruise_ID == ID_1)
prof_poc_1       = b2_poc[prof_poc_idx]
prof_poc_depth_1 = b2_depth[prof_poc_idx]

print(prof_poc_1)

b2_DateTime_1 = b2_DateTime.date[prof_poc_idx]
print(b2_DateTime_1[0])


print("MLD: "+str(prof_MLD))

#Define the figure window including 5 subplots orientated horizontally
fig, (ax3) = plt.subplots(1, sharey=True, figsize=(6,6), \
gridspec_kw={'hspace': 0.2})

ax3.plot([np.min(prof_poc_1)-0.25,np.max(prof_poc_1)+1.5],[prof_MLD,prof_MLD], \
         color = 'k', marker = 'None', linestyle = '--', label= 'MLD')
ax3.plot(prof_poc_1,prof_poc_depth_1, \
         color = 'g',  marker = 'o', linestyle = 'None',label= 'POC')
ax3.set_ylabel('Depth (m)', fontsize=15)
ax3.yaxis.set_tick_params(labelsize=15)
ax3.set_ylim([270,0]) 
ax3.set_xlabel('POC (ug/kg)', fontsize=15, color = 'k')
ax3.xaxis.set_tick_params(labelsize=15)
ax3.set_xlim(xmin=0.00, xmax=np.nanmax(prof_poc_1)+1.5)
ax3.legend(loc="lower right", fontsize=10,title= ID_1)
ax3.text(np.nanmin(prof_poc_1)+0.005, 268, " Date: "+str(b2_DateTime_1[0]), color='k', fontsize=12)
ax3.xaxis.set_major_locator(plt.MaxNLocator(5))
#ax3.set_xscale('log')
plt.show()
