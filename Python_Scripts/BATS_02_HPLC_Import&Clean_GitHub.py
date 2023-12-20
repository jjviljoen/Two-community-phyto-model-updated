"""
BATS: Bottle HPLC Pigments Import & Clean Original data & match with CTD IDs

@author: Johan Viljoen - j.j.viljoen@exeter.ac.uk
"""
#%%

### LOAD PACKAGES ###
#General Python Packages
import pandas as pd # data analysis and manipulation tool
import numpy as np # used to work with data arrays
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import seaborn as sns
# Import specific modules from packages
from datetime import date
from PyAstronomy import pyasl # used to compute decimal year from DateTime & back. Info: https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/decimalYear.html
from dateutil import relativedelta
from bisect import bisect
from math import nan

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
######
### READ, CLEAN & FILTER ORIGINAL BOTTLE PIGMENT DATA ###
######

# Original csv HPLC data from the BCO-DMO project site on BATS at https://www.bco-dmo.org/dataset/893521

#Filename
file_pigments = 'data/bats_pigments_BCO_DMO.csv' #name file location and csv file name
# Data downloaded on 14 July 2023 and converted to CSV

# Read CSV
bottle   = pd.read_csv(file_pigments) #Read CSV as panda dataframe

# Inspect bottle df
bottle.head()
bottle.info()
# Meta Data Info #
# pigment16 = Turner Chlorophyll a (ug/L)
# pigment14 = Chlorophyll a (HPLC) (ng/L)

# Rename Columns
bottle.rename(columns={"ID": "Cruise_ID",
                                "Latitude": "lat",
                                "decy": "DecYear",
                                "Longitude": "lon",
                                "Depth": "depth",
                                "p14": "pigment14",
                                "p16_Chl": "pigment16"},inplace=True)
bottle.head()

# Remove rows with depths below 300m
bottle = bottle[bottle["depth"]<305] # 5753 rows

# Replace empty cells of "-999" with python NaNs
bottle.replace(-999, np.nan, inplace = True)

# Remove rows with NaN HPLC chla values
bottle.dropna(subset=['pigment14'], inplace = True) # 5575 rows

# Convert Decimal year to DateTime. # np.vectorize replaces a iterative for loop
bottle['time'] = pd.to_datetime(np.vectorize(pyasl.decimalYearGregorianDate)(bottle['DecYear'], "datetime"))
# Convert Datetime to DatetimeIndex format
bottle['time'] = pd.DatetimeIndex(bottle['time'])
bottle['time'].head()

#Extract and reorder required columns
required_columns = ["Cruise_ID", "DecYear",'time','Date', 'lat','lon','depth',
                    'pigment14', 'pigment16']
bottle = bottle[required_columns]
# Inspect reformatted dataframe
bottle.info()

# Remove last 2 digits from ID to make all results in profile the same ID
bottle['Cruise_ID'] = bottle['Cruise_ID'].astype(str).str[:-2].astype(np.int64)

# Convert HPLC Chla to ug/L # Most HPLC concentrations in ng/kg, convert to ug/kg
bottle['pigment14'] = bottle['pigment14']/1000 # replaced original column with converted values
#print(bottle['pigment14'])

# sort by time
bottle = bottle.sort_values(by=['time','depth'])
# Reset bottle df index removing old index with missing numbers after slice
bottle = bottle.reset_index(drop=True)

### Convert and separate datetime data into new additional columns ###

# convert to datetime format & extract year into new column called "yyyy"
bottle['yyyy'] = pd.to_datetime(bottle['Date']).dt.year 
#print(bottle['yyyy'])

# convert to datetime format & extract mopnth into new column called "mm"
bottle['mm'] = pd.to_datetime(bottle['Date']).dt.month
#print(bottle['mm'])

# convert to datetime format & extract Date yyyy-mm-dd into new column called "Date"
bottle['Date'] = pd.to_datetime(bottle['Date']).dt.date
#print(bottle['Date'])

# Write Cleaned bottle df to intermediate csv
bottle.to_csv('data/BATS_Pigments_01.csv')

### Timespan of bottle data ###

# Print start and end dates of bottle data
print("Bottle Dates: "+str(min(bottle['Date']))+" to "+str(max(bottle['Date'])))

# Print period timespan of bottle data using base date subtraction - only days
print("Bottle Date Length: "+str(max(bottle['Date'])-min(bottle['Date'])))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle['Date']), max(bottle['Date']))
print("Timespan: "+str(b_date_length))

#%%

### EXTRACT PROF IN BOX AROUND BATS & AFTER 1990 ###

bottle.info()

#Create copy of df to calculate measurements lost/removed
bottle_x = bottle.copy()

# Remove profiles before 1989
bottle = bottle[bottle["Date"]>date(1989,12,31)]

# Filter stations for 0.25 margin around BATS
bats_lat = 31.67
bats_lon = -64.17
margin = 0.25
lat_min, lat_max = bats_lat - margin, bats_lat + margin
lon_min, lon_max = bats_lon - margin, bats_lon + margin
# Slice df for 0.25deg box aorund BATS
bottle = bottle[(bottle["lat"] >= lat_min) & (bottle["lat"] <= lat_max)
                           & (bottle["lon"] >= lon_min) & (bottle["lon"] <= lon_max)]
# Sort df again
bottle = bottle.sort_values(by=['time','depth'])

# Reset bottle df index replacing old index column
bottle = bottle.reset_index(drop=True)

#Calculate measurements lost
prof_lost = len(bottle_x) - len(bottle)
print("Bottle Measurements Lost = "+str(prof_lost))

#%%
### SUMMARY & DESCRIPTIVE STATS on Bottle Pigment Data ###

# Inspect first few rows of current dataframe
print(bottle.head(5))
#bottle.info()

# Boxplot of Turner and HPLC Chla 
bottle[["pigment16","pigment14"]].plot.box()
plt.show()

# Group by Cruise ID profile and count number of samples per profile
bottle_2 = bottle[["Cruise_ID", "pigment14"]].groupby("Cruise_ID").count()

# Count number of profiles per number of depths sampled
print(bottle_2.groupby("pigment14")["pigment14"].count()) # Few profiles with less than 6 measurements

# Bar plot of profiles per number of depths sampled
bottle_2.groupby("pigment14")["pigment14"].count().plot.bar()
plt.show()

#%%
### Remove bottle profiles with less than 6 measurements

# Remove Chl measurements where Chl is zero in upper 100m
bottle = bottle.drop(bottle[(bottle.pigment14 == 0) & (bottle.depth < 101)].index)

# Remove duplicate measurement in a profile
bottle.drop_duplicates(subset=['Cruise_ID', 'depth'], keep='last', inplace=True)

# Sort new df by time and depth again
bottle = bottle.sort_values(by=['time','depth'])

#Loop
# Extract required variables
bottle = bottle.reset_index(drop=True)  # Resetting the index for easier access

No_CHL_Prof = np.zeros(len(bottle))  # Initialize array

for i in range(len(bottle)):
    mask = (bottle['Cruise_ID'] == bottle.at[i, 'Cruise_ID'])
    chla = bottle.loc[mask, 'pigment14'].values
    depth = bottle.loc[mask, 'depth'].values

    if depth.min() > 30:
        No_CHL_Prof[i] = 0
    else:
        depth_mask = depth <= 300
        chla = chla[depth_mask]
        depth = depth[depth_mask]
        non_nan_mask = ~np.isnan(chla)
        chla = chla[non_nan_mask]

        No_CHL_Prof[i] = np.count_nonzero(chla)

bottle['No_CHL_Prof'] = No_CHL_Prof

#Filter df for profile containing only 6 or more measurements
bottle_6 = bottle[bottle['No_CHL_Prof'].values > 5]

#Sort df
bottle_6 = bottle_6.sort_values(by=['time', 'depth'])

# Reset bottle df index replacing old index column
bottle_6 = bottle_6.reset_index(drop=True)

# Test if new df count only includes 6 or more per Cruise ID
bottle_6[["Cruise_ID","pigment14"]].groupby("Cruise_ID").count()

# Quick pandas boxplots
bottle_6.boxplot(by='mm',column='pigment14')

#%%

### COUNT BOTTLE PROFILES PER YEAR AGAIN ###

# Create new df with number of CTD profiles (cruises) per year
bottle_y = bottle_6[["Cruise_ID", "yyyy"]].groupby("yyyy").nunique()

# Bar plot of profiles per year
bottle_y.plot.bar(color = "darkgreen")
plt.show()

# Nice Table for Notebook
print(bottle_y.to_markdown())

#%%

### EXTRACT CLEANED DATA & MAKE BOTTLE ID LIST ###

### Extract required data from new bottle_6 dataset ###
b_time     = bottle_6.loc[:,'time'].to_numpy()
b_date     = bottle_6.loc[:,'Date'].to_numpy()
b_depth    = bottle_6.loc[:,'depth'].to_numpy()
b_Fchla    = bottle_6.loc[:,'pigment16'].to_numpy()
b_chla     = bottle_6.loc[:,'pigment14'].to_numpy()
b_ID       = bottle_6.loc[:,'Cruise_ID'].to_numpy()
b_year     = bottle_6.loc[:,'yyyy'].to_numpy()
b_month    = bottle_6.loc[:,'mm'].to_numpy()

#Convert bottle float time to Datetimeindex type
b_DateTime = pd.DatetimeIndex(b_time, dtype='datetime64[ns]', name='date_time', freq=None)

bottle_6['DateTime'] = b_DateTime

### Cruise_ID list for new df is ID_list_6
# Removes Duplicates
ID_list_6 = pd.unique(b_ID)

print(len(ID_list_6))

#%%
### SCATTER PLOTS TO INSPECT HPLC VS FLUOROMETER CHLA ##

# Scatter plot HPLC vs Turner Chla
fig, (ax3) = plt.subplots(1, sharey=True, figsize=(8,8))
scatter = ax3.scatter(b_Fchla,b_chla, alpha=0.5, c = 'g')
ax3.set_ylabel('HPLC Chl-a (ug/L)', fontsize=15)
ax3.yaxis.set_tick_params(labelsize=15)
ax3.set_xlabel('Fluorescence Chl-a (ug/L)', fontsize=15, color = 'k')
ax3.xaxis.set_tick_params(labelsize=15)

fig.savefig('plots/HPLCvsTurner_Scatter.png', format='png', dpi=300, bbox_inches="tight")
plt.show()

# Scatter plot HPLC vs Turner Chla
fig, (ax3) = plt.subplots(1, sharey=True, figsize=(8,8), \
gridspec_kw={'hspace': 0.2})   
ax3.scatter(b_Fchla,b_chla, alpha=0.5, c = 'g')
ax3.set_ylabel('HPLC Chl-a (ug/L)', fontsize=15)
ax3.yaxis.set_tick_params(labelsize=15)
ax3.set_xlabel('Fluorescence Chl-a (ug/L)', fontsize=15, color = 'k')
ax3.xaxis.set_tick_params(labelsize=15)
ax3.set_xscale('log')
ax3.set_yscale('log')

fig.savefig('plots/HPLCvsTurner_Scatter.png', format='png', dpi=300, bbox_inches="tight")
plt.show()

#%%

# Box plots per month all HPLC and Turner Chla data (Seaborn package)
fig, (ax1,ax2) = plt.subplots(2, figsize=(8,12),gridspec_kw={'hspace': 0.5})
#fig, ax = plt.subplots()
#fig.set_size_inches((12,4))
sns.boxplot(x='mm',y='pigment14',data=bottle_6,ax=ax1)
ax1.set_title('(a) HPLC Chlorophyll-a (µg/l)', fontsize = 20, color='k')
ax1.set_ylabel('Chl-a (µg/L)', fontsize=15)
ax1.yaxis.set_tick_params(labelsize=15)
ax1.set_xlabel('Month', fontsize=15, color = 'k')
ax1.xaxis.set_tick_params(labelsize=15)

# Box plot of all HPLC chla vs depth
sns.boxplot(x='mm',y='pigment16',data=bottle_6,ax=ax2)
ax2.set_title('(b) Turner Chlorophyll-a (µg/l)', fontsize = 20, color='k')
ax2.set_ylabel('Chl-a (µg/L)', fontsize=15)
ax2.yaxis.set_tick_params(labelsize=15)
ax2.set_xlabel('Month', fontsize=15, color = 'k')
ax2.xaxis.set_tick_params(labelsize=15)

plt.show()
fig.savefig('plots/Monthly_Boxplot_Chla_comparison.png', format='png', dpi=300, bbox_inches="tight")

#%%

### MATCH BOTTLE PIGMENT DATA WITH CTD PROFILES ###

### READ CLEANED CTD DATA FROM CSV ###
# CSV filename
filename_1 = 'data/BATS_CTD_Cleaned.csv'
# Load data from csv. "index_col = 0" make first column the index.
ctd        = pd.read_csv(filename_1, index_col = 0)

### Extract required data from CTD dataframe into numpy arrays ###
ctd_time      = ctd.loc[:,'time'].to_numpy()
ctd_date      = ctd.loc[:,'Date'].to_numpy()
depth         = ctd.loc[:,'depth'].to_numpy()
fluorescence  = ctd.loc[:,'fluorescence'].to_numpy()
ID_ctd        = ctd.loc[:,'cruise_ID'].to_numpy()
ctd_Decimal_year = ctd.loc[:,'Dec_Year'].to_numpy()
ctd_DateTime  = pd.DatetimeIndex(pd.to_datetime(ctd['time']))

### Cruise ID list for CTD ###
# Extract cruise_ID & Remove Duplicates
ID_list_ctd = pd.unique(ctd['cruise_ID'].values) # ID_list_ctd now = x1 ID cruise number per CTD profile

# Compare length of ID_list to all cells containing cruise/profile IDs
print(len(ID_list_ctd))
print(len(ID_ctd))

#%%

### REPLACE BOTTLE PIGMENT IDS WITH MISSING CTD IDs based on same date ###

ID_list_6_x = np.array(range(len(ID_list_6))) + nan
count_replace = 0
count = 0
for i in ID_list_6:
    b_idx          = np.where(bottle_6.Cruise_ID == i)
    b_ID_prof_all  = b_ID[b_idx]
    prof_ID        = np.unique(b_ID[b_idx])
    b_ID_check     = np.isin(prof_ID,ID_list_ctd)
    prof_date      = np.unique(b_date[b_idx])
    prof_date      = pd.unique(prof_date)
    prof_date      = prof_date[0] 
    if b_ID_check == False:
        ctd_date2 = pd.to_datetime(ctd_date).date
        ind = bisect(ctd_date2, prof_date, hi=len(ctd_time)-1)
        date_nearest   = min(ctd_date2[ind], ctd_date2[ind-1],key=lambda x: abs(x - prof_date))
        date_check_x    = np.where(ctd_date2 == prof_date)
        b_date_check    = np.isin(prof_date,ctd_date2[date_check_x])
        print("### ID MISSING ###")
        print("Bottle ID: "+str(prof_ID[0]))
        print(prof_date)
        print("Nearest CTD date: "+str(date_nearest))
        if b_date_check == True:
            ID_replace      = np.unique(ID_ctd[date_check_x])
            ID_replace      = ID_replace[0]
            bottle_6.loc[bottle_6["Date"] == prof_date , "Cruise_ID"] = ID_replace
            count_replace = count_replace+1
            print("REPLACED")
            print("CTD ID - "+str(ID_replace))
        else:
            ID_replace = prof_ID
    else:
        ID_replace = prof_ID
        
    ID_list_6_x[count] = int(ID_replace)

    count=count+1
print("REPLACED = "+str(count_replace))

ID_list_6_x = ID_list_6_x.astype(int)

#%%
### REMOVE BOTTLE PROFILES WITH NO MATCHING CTD profile ###

# Extract new df that only has matching CTD cruise IDs
bottle_6 = bottle_6[bottle_6.Cruise_ID.isin(ID_list_ctd)]

# Sort df again
bottle_6 = bottle_6.sort_values(by=['time','depth'])

# Reset bottle df index replacing old index column
bottle_6 = bottle_6.reset_index(drop=True)

# Test if new df count only includes 6 or more per Cruise ID
bottle_6[["Cruise_ID","pigment14"]].groupby("Cruise_ID").count()

### New Cruise_ID list for new df is ID_list_6
# Removes Duplicates
ID_list_6 = pd.unique(bottle_6['Cruise_ID'].values)

print("Chla Profiles with CTD date: "+str(len(ID_list_6)))

#%%

### Write Cleaned bottle data to CSV ###

# Write Cleaned bottle df to intermediate csv
bottle_6.to_csv('data/BATS_Pigments_Cleaned.csv')

#%%
### Get MLDs for bottle profiles ###

### Read CTD prof data ###
# CSV filename
filename_2 = 'data/BATS_CTD_profData.csv'
# Load data from csv. "index_col = 0" make first column the index.
ctd_prof = pd.read_csv(filename_2, index_col = 0)
# Inspect ctd_prof df
ctd_prof.info()

# Extract required data from df
try:
    ctd_DateTime_prof = pd.DatetimeIndex(ctd_prof['Time'])
#ctd_DateTime_prof = pd.DatetimeIndex(ctd_prof.index)
except: 
    ctd_DateTime_prof = pd.DatetimeIndex(ctd_prof.index)
ctd_date_prof     = ctd_prof.loc[:,'Date'].to_numpy()
MLD               = ctd_prof.loc[:,'MLD'].to_numpy()

# Inspect CTD single profile data
ctd_prof.head()

# Create new single CTD data df containing only data for profiles also in Bottle list
ctdprof_in_bottle =  ctd_prof[ctd_prof.ID_list.isin(ID_list_6)]

# Sort df again
ctdprof_in_bottle = ctdprof_in_bottle.sort_values(by=['Time'])

# Reset bottle df index replacing old index column
ctdprof_in_bottle = ctdprof_in_bottle.reset_index(drop=True)
ctdprof_in_bottle.info()

### Extract required single ctd data from new df ###
try:
    b_time_mld  = ctdprof_in_bottle.loc[:,'Time'].to_numpy()
#ctd_DateTime_prof = pd.DatetimeIndex(ctd_prof.index)
except: 
    b_time_mld  = ctdprof_in_bottle.index
#b_time_mld  = ctdprof_in_bottle.loc[:,'Time'].to_numpy()
b_MLD_prof  = ctdprof_in_bottle.loc[:,'MLD'].to_numpy()
b_MLD_prof_ID  = ctdprof_in_bottle.loc[:,'ID_list'].to_numpy()

#%%

### CREATE BOTTLE PROFILE SINGLE DATA DF ###

# MLD df with time to extract bottle MLD later for each profile ID, some profiles on same time/date so can't use np.unique
bottle_6.info()
bottle_prof = bottle_6[['time','Cruise_ID','DecYear','Date','yyyy','mm',]] # Could also add Lat and Long
bottle_prof = bottle_prof.drop_duplicates(subset=['Cruise_ID'])
print(len(bottle_prof))

# Sort df again first on time & then ID to account to profiles on same date
bottle_prof = bottle_prof.sort_values(by=['time','Cruise_ID']) 

bottle_prof['MLD']      = b_MLD_prof
bottle_prof['MLD_time'] = b_time_mld
bottle_prof['MLD_ID']   = b_MLD_prof_ID

# Reset bottle df index replacing old index column
bottle_prof = bottle_prof.reset_index(drop=True)
bottle_prof.info()
bottle_prof.head()

# test if MLDs added from CTD in correct order via IDs
test_df = bottle_prof[['Cruise_ID','MLD_ID']]
test_df['test'] = np.where(test_df['Cruise_ID'] == test_df['MLD_ID'],True,False)

print(np.count_nonzero(test_df['test']))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
bottle_date_length = date_span(min(bottle_prof['Date']), max(bottle_prof['Date']))
print("Timespan: "+str(bottle_date_length))
print("Min Date: "+str(min(bottle_prof['Date'])))
print("Max Date: "+str(max(bottle_prof['Date'])))

# Save to CSV
bottle_prof.to_csv('data/BATS_Bottle_profData.csv')

#%%

### EXTRACT CLEANED DATA & MAKE BOTTLE ID LIST ###

# CSV filename
filename_1 = 'data/BATS_Pigments_Cleaned.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_6   = pd.read_csv(filename_1, index_col = 0)

bottle_6.info()

### Extract required data from new bottle_6 dataset ###
b_time     = bottle_6.loc[:,'time'].to_numpy()
b_date     = bottle_6.loc[:,'Date'].to_numpy()
b_depth    = bottle_6.loc[:,'depth'].to_numpy()
b_Fchla    = bottle_6.loc[:,'pigment16'].to_numpy()
b_chla     = bottle_6.loc[:,'pigment14'].to_numpy()
b_ID       = bottle_6.loc[:,'Cruise_ID'].to_numpy()
b_year     = bottle_6.loc[:,'yyyy'].to_numpy()
b_month    = bottle_6.loc[:,'mm'].to_numpy()
#b_Decimal_year = ctd.loc[:,'Dec_Year'].to_numpy()
#b_MLD_2    = bottle_6.loc[:,'MLD'].to_numpy()
# Bottle DateTime data
b_DateTime     = pd.DatetimeIndex(pd.to_datetime(bottle_6['time']))

### Cruise_ID list for new df is ID_list_6
# Removes Duplicates
ID_list_6 = pd.unique(b_ID)
#print(ID_list_6)

#%%

# Scatter plot of all HPLC chla vs depth with month as colour
my_cmap = plt.get_cmap('cmo.phase_r',12)

fig, ax = plt.subplots(1, figsize=(6,7), \
gridspec_kw={'hspace': 0.4})   
scatter = ax.scatter(b_chla,b_depth, c = b_month, alpha = 0.6, cmap = my_cmap)
ax.set_ylim([260,0])
ax.set_ylabel('Depth (m)', fontsize=15)
ax.yaxis.set_tick_params(labelsize=14)
ax.xaxis.tick_top()
ax.set_xlabel('HPLC Chla', fontsize=15)   
ax.xaxis.set_label_position('top')
ax.xaxis.set_tick_params(labelsize=14)
# add legend to the plot with names
ax.legend(loc="best", fontsize=11, handles=scatter.legend_elements()[0], 
           title="Month", labels = scatter.legend_elements()[1])
fig.savefig('plots/Bottle-HPLC_chla_depths.png', format='png', dpi=300, bbox_inches="tight")

#%%

### COUNT BOTTLE PROFILES PER month AGAIN ###

# Create new df with number of CTD profiles (cruises) per year
bottle_y = bottle_6[["Cruise_ID", "yyyy"]].groupby("yyyy").nunique()
# Write to csv
#bottle_y.to_csv('data/BATS_Bottle_profiles_per_year.csv')

# Bar plot of profiles per year
bottle_y.plot.bar(color = "darkgreen")
plt.show()

# Nice Table for Notebook
print(bottle_y.to_markdown())

#%%

### EXAMPLE PROFILE PLOT ###
# Example profile of bottle data overlain with matched CTD fluoresence & MLD

# Define profile / Cruise ID number
ID_1 =  10195003#20256004#10186003 #10256008 #10283008 #10009006
# Supplemental Example Profile: 10195003

ID_2 =  ID_1

# Extract all bottle df rows of this profile into new df to print
bot_prof = bottle_6[bottle_6['Cruise_ID'] == ID_1]
print(bot_prof.head(5))

### Define and extract variables for specific profile ###

# CTD data for Profile = ID_1
x = np.where(ctd.cruise_ID == ID_2) # Index for ctd data
depth_1        = depth[x]
fluorescence_1 = fluorescence[x]

# Bottle data for Profile = ID_1
AS = np.where(bottle_6.Cruise_ID == ID_1) # Index for bottle data
b_Fchla_1      = b_Fchla[AS]
b_chla_1       = b_chla[AS]
b_depth_1      = b_depth[AS]
b_DateTime_1   = b_DateTime.date[AS]

# MLD of profile
ASD = np.where(ID_list_ctd == ID_2) # Index for calculated MLD
MLD_1 = MLD[ASD]

b_DateTime_1 = b_DateTime_1[1]
print(b_DateTime_1)

# Print profile MLD
print(float(MLD_1))

#Plot size
XSIZE = 6 #Define the xsize of the figure window
YSIZE = 6 #Define the ysize of the figure window

##Define the figure window with 1 subplot
fig, (ax3) = plt.subplots(1, figsize=(XSIZE,YSIZE), \
gridspec_kw={'hspace': 0})
# MLD Line 
ax3.plot([np.min(b_chla)-0.02,np.max(b_chla)+0.05],[MLD_1,MLD_1], \
         color = 'r', marker = 'None', linestyle = '--', label= 'MLD')
# CTD Fluoresence
ax3.plot(fluorescence_1,depth_1, \
         color = 'g', marker = 'o', linestyle = 'None', label= 'CTD Fluorescence')
# HPLC Chl-a
ax3.plot(b_chla_1,b_depth_1, \
         color = 'b', marker = 'o', linestyle = 'None', label= 'HPLC Chl-a')
# Turner Chl-a
ax3.plot(b_Fchla_1,b_depth_1, \
         color = 'r', marker = 'o', linestyle = 'None', label= 'Turner Chl-a')
# Set axis info and titles
ax3.set_ylabel('Depth (m)', fontsize=15)
ax3.yaxis.set_tick_params(labelsize=13)
ax3.set_ylim([300,0]) 
ax3.set_xlabel('Chl-a (µg/L)', fontsize=15, color = 'k')
ax3.xaxis.set_tick_params(labelsize=13)
if np.max(fluorescence_1) > np.nanmax(b_chla_1):
    ax3.set_xlim(xmin=np.nanmin(b_chla_1)-0.025, xmax=np.max(fluorescence_1)+0.02)
else:
    ax3.set_xlim(xmin=np.nanmin(b_chla_1)-0.025, xmax=np.nanmax(b_chla_1)+0.05)
ax3.xaxis.set_major_locator(plt.MaxNLocator(4))
ax3.legend(loc="lower right", fontsize=13,title= ID_1)
ax3.text(np.min(b_chla_1)+0.01, 298, "Date: "+str(b_DateTime_1), color='k', fontsize=12)
fig.savefig('plots/Bottle_Profile_'+str(ID_1)+'.png', format='png', dpi=300, bbox_inches="tight")

#Complete the plot
plt.show()