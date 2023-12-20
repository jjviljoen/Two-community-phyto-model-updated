"""
BATS - Fit Two-Community Viljoen et al. Model to HPLC Chla & POC data

Model developed and modified based on Brewin et al. 2022 Two community model: https://doi.org/10.1029/2021JC018195

Model used in this script was developed with the paper of Viljoen, Sun and Brewin
Titled: "Climate variability shifts the vertical structure of phytoplankton in the Sargasso Sea"

@author: Johan Viljoen - j.j.viljoen@exeter.ac.uk
"""
### PLOT EVERY PROFILE MODEL FITTED TO?

model_plot_1 = False # If true will plot both Chla & POC model fits for each profile loop

#%%
### LOAD PACKAGES ##
#General Python Packages
import pandas as pd # data analysis and manipulation tool
import numpy as np # used to work with data arrays
import matplotlib as mpl
import matplotlib.ticker as tick
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial" # Set font for all plots
import seaborn as sns
import cmocean
# Import specific modules from packages
from datetime import date
#from PyAstronomy import pyasl # used to compute decimal year from DateTime & back. Info: https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/decimalYear.html
from dateutil import relativedelta
from math import nan
from matplotlib.ticker import FormatStrFormatter
from scipy import interpolate # used to interpolate profiles for contour plots
from lmfit import Minimizer, Parameters, report_fit
from scipy.stats import pearsonr, spearmanr
# Supress
import warnings
warnings.filterwarnings("ignore") # Added to remove the warning "UserWarning: Warning: 'partition' will ignore the 'mask' of the MaskedArray." on 2nd to last cell of code
#
# Supresses outputs when trying to "divide by zero" or "divide by NaN" or "treatment for floating-point overflow"
np.seterr(all='ignore');
#%%

### DEFINE ALL FUNCTIONS ###

## Define function to calculate Morel diffuse attenuation coefficient
def calculate_Kd_Zp(chla_surf):
    """
    Calculate Kd and Zp based on the given prof_chla_surf.
    Equation for Kd using Morel euphotic depth https://doi.org/10.1016/j.rse.2007.03.012

    Parameters:
    - prof_chla_surf (float): Chlorophyll-a concentration at the surface.

    Returns:
    - tuple: (Kd, Zp)
    """
    Kd = 4.6 / 10.**(1.524 - 0.436*np.log10(chla_surf) - \
                    0.0145*np.log10(chla_surf)**2. + 0.0186*np.log10(chla_surf)**3.)
    Zp = 4.6 / Kd
    return Kd, Zp
    """
    Kd_result, Zp_result = calculate_Kd_Zp(prof_chla_surf_value)
    """

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

# Define function for Community 1 normalised Chl-a
# Equation 8 of Brewin et al. 2022 Model
def fcn2min_1pop(params1, X, Y):
    P1 = params1['P1']
    P2 = params1['P2']
    model = 1 - 1./(1+np.exp(-(P1/P2)*(Y-P2))) #Reduce equation to two parameters [p1 and p2] (as depth tends to zero)
    return(model-X)

# Define function for Community 1 and 2 normalised Chl-a
# Equation 7 of Brewin et al. 2022 Model
def fcn2min_2pop(params2, X, Y):
    P1 = params2['P1']
    P2 = params2['P2']
    P3 = params2['P3'] # maximum biomass (normalised)
    P4 = params2['P4'] # psuedo parameter controlling depth of maximum biomas (P3)
    P5 = params2['P5'] # width of peak
    MLD_pop = 1 - 1./(1+np.exp(-(P1/P2)*(Y-P2))) #
    DCM_pop = P3*np.exp(-((Y - ((P4+(P5*3.))))/(P5))**2.) 
    #DCM_pop = P3*np.exp(-((Y - P4)/(P5))**2.) 
    model = MLD_pop + DCM_pop
    return(model-X)

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

# Define function for POC scaling factors for 2 communities (Surface & Subsurface)
def fcn2min_2pop_poc(params3,X1,A1,A2):
    # Input parameters
    P2 = params3['P2']
    P3 = params3['P3']
    model = (1-P3)*A1 + P2*A2 + P3
    return(model-X1)

# Define 3rd step function for getting sigmoid parameters
def fcn2min_2pop_3rd_step2(params2, X2, Y2):
    P2 = params2['P2'] # Tau 1
    P3 = params2['P3'] # maximum biomass (normalised)
    P4 = params2['P4'] # psuedo parameter controlling depth of maximum biomas (P3)
    P5 = params2['P5'] # width of peak
    P6 = params2['P6'] # Com 2 weight for POC
    P7 = params2['P7'] # Background
    P1 = 10**(0.08 * P2 + 0.66) # Re
    # Chl model
    A1 = 1 - 1./(1+np.exp(-(P1/P2)*(Y2-P2))) #
    A2 = P3*np.exp(-((Y2 - ((P4+(P5*3.))))/(P5))**2.) 
    # POC model
    model = (1-P7)*A1 + P6*A2 + P7
    return(model-X2)

# Define function for POC scaling factor for Community 1 (Surface only)
def fcn2min_1pop_poc(params3,X1,A1):
    # Input parameters
    P3 = params3['P3']
    model = (1-P3)*A1 + P3
    return(model-X1)

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
depth         = ctd.loc[:,'depth'].to_numpy()
ID_ctd        = ctd.loc[:,'cruise_ID'].to_numpy()
ctd_Decimal_year = ctd.loc[:,'Dec_Year'].to_numpy()
ctd_DateTime  = pd.to_datetime(ctd['time'].values)

### Cruise ID list for CTD ###
ID_list_ctd = pd.unique(ID_ctd) # ID_list_ctd now = x1 ID cruise number per CTD profile

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

# Extract required data from df
try:
    ctd_DateTime_prof = pd.DatetimeIndex(ctd_prof['Time'])
#ctd_DateTime_prof = pd.DatetimeIndex(ctd_prof.index)
except: 
    ctd_DateTime_prof = pd.DatetimeIndex(ctd_prof.index)

ctd_prof.set_index(ctd_DateTime_prof, inplace= True)

# Extract all MLDs and corresponding dates
ctd_date_prof     = ctd_prof.loc[:,'Date'].to_numpy()
MLD               = ctd_prof.loc[:,'MLD'].to_numpy()

#%%

### EXTRACT CLEANED POC DATA & ID LIST ###

# CSV filename
filename_1 = 'data/BATS_Bottle_POC.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_poc   = pd.read_csv(filename_1, index_col = 0)
bottle_poc.info()

# Remove rows deeper than 400m 
bottle_poc = bottle_poc[bottle_poc["depth"] < 410]

# Sort new df by time, ID and depth
bottle_poc = bottle_poc.sort_values(by=['cruise_ID','depth'])
#bottle_poc = bottle_poc.sort_values(by=['Date','depth'])

# Reset bottle df index replacing old index column
bottle_poc = bottle_poc.reset_index(drop=True)

### EXTRACT CLEANED DATA & MAKE NEW POC ID LIST ###

### Extract required data from new bottle_6 dataset ###
b2_time     = bottle_poc.loc[:,'time'].to_numpy()
b2_date     = bottle_poc.loc[:,'Date'].to_numpy()
b2_depth    = bottle_poc.loc[:,'depth'].to_numpy()
b2_poc      = bottle_poc.loc[:,'POC'].to_numpy()
b2_ID       = bottle_poc.loc[:,'cruise_ID'].to_numpy()
b2_DecYear  = bottle_poc.loc[:,'DecYear'].to_numpy()

#Convert array object to Datetimeindex type
b2_DateTime = pd.to_datetime(bottle_poc['time'].values)

### Cruise_ID list
ID_list_poc = pd.unique(b2_ID)
print(len(ID_list_poc))
# 416 profiles with 6 or more POC measurements that matches bottle pigment list

### Import POC PROF Data
# CSV filename
filename_1 = 'data/BATS_Bottle_POC_profData.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_poc_prof = pd.read_csv(filename_1, index_col = 0)

# Sort df by date and ID
bottle_poc_prof = bottle_poc_prof.sort_values(by=['Date'])

# Reset bottle df index replacing old index column
bottle_poc_prof = bottle_poc_prof.reset_index(drop=True)

print(len(bottle_poc_prof))
print(len(ID_list_poc))

# Print start and end dates of bottle data
print("Bottle Dates: "+str(min(bottle_poc_prof['Date']))+" to "+str(max(bottle_poc_prof['Date'])))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_poc_prof['Date']), max(bottle_poc_prof['Date']))
print("Timespan: "+str(b_date_length))

#%%
#########################################
### FILTER CHLA FOR ONLY POC PROFILES ###
#########################################

### EXTRACT CLEANED PIGMENT BOTTLE DATA & MAKE CHLA ID LIST ###

# CSV filename
filename_1 = 'data/BATS_Pigments_Cleaned.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_6   = pd.read_csv(filename_1, index_col = 0)

bottle_6.info()

# Create new df containing only data for profiles also in pigment bottle list
bottle_6 =  bottle_6[bottle_6.Cruise_ID.isin(ID_list_poc)]

# Sort new df by ID and depth
bottle_6 = bottle_6.sort_values(by=['time','depth'])

# Reset bottle df index replacing old index column
bottle_6 = bottle_6.reset_index(drop=True)

# Write Cleaned bottle df to csv
bottle_6.to_csv('data/BATS_Pigments_Cleaned.csv')

### Extract required data from new bottle_6 dataset ###
b_time     = bottle_6.loc[:,'time'].to_numpy()
b_date     = bottle_6.loc[:,'Date'].to_numpy()
b_depth    = bottle_6.loc[:,'depth'].to_numpy()
b_chla     = bottle_6.loc[:,'pigment14'].to_numpy()
b_ID       = bottle_6.loc[:,'Cruise_ID'].to_numpy()
b_Decimal_year = bottle_6.loc[:,'DecYear'].to_numpy()
b_DateTime     = pd.to_datetime(bottle_6['time'].values)

### Cruise_ID list for new df is ID_list_6
ID_list_6 = pd.unique(b_ID)
print(len(ID_list_6))
print(len(ID_list_poc))

### Bottle Single MLD & DateTimeIndex ###

### Read/Import cleaned Bottle data from CSV
# CSV filename
filename_1 = 'data/BATS_Bottle_profData.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_prof   = pd.read_csv(filename_1, index_col = 0)
bottle_prof.info()

# Create new df containing only data for profiles also in pigment bottle list
bottle_prof =  bottle_prof[bottle_prof.Cruise_ID.isin(ID_list_6)]

# Sort new df by ID and depth
bottle_prof = bottle_prof.sort_values(by=['Date'])

# Reset bottle df index replacing old index column
bottle_prof = bottle_prof.reset_index(drop=True)

# Write Cleaned bottle df to csv
bottle_prof.to_csv('data/BATS_Bottle_profData.csv')

# Extract bottle MLD with corresponding time ###
b_DateTime_prof  = pd.to_datetime(bottle_prof['time'].values)
b_MLD_prof       = bottle_prof.loc[:,'MLD'].to_numpy()

print(len(ID_list_6))
print(len(bottle_prof))

# Print start and end dates of bottle data
print("Bottle Dates: "+str(min(bottle_prof['Date']))+" to "+str(max(bottle_prof['Date'])))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_prof['Date']), max(bottle_prof['Date']))
print("Timespan: "+str(b_date_length))

#%%

### APPLY VILJOEN et al. MODEL TO SINGLE PROFILE ###
# Model modified from Brewin et al. 2022
ID_1 = 10195003#10030002#20335002#10363018#10241009#10273008
# 2 Communities: 20174004, 20280003,20256004,20335002
# Only surface community: 10208005, 10218006, 10265004
# Supplemental Example: 10195003

# Make copy of bottle Chla array
m_chla = b_chla

# CTD MLD for profile
prof_MLD_idx = np.where(ID_list_ctd == ID_1)
prof_MLD     = MLD[prof_MLD_idx]

# Chla Bottle data
prof_bottle_idx = np.where(bottle_6.Cruise_ID == ID_1)
b_DateTime_1 = b_DateTime.date[prof_bottle_idx]
print(b_DateTime_1[0])

prof_depth   = b_depth[prof_bottle_idx]
prof_chla    = m_chla[prof_bottle_idx]

# POC data
prof_poc_idx    = np.where(bottle_poc.cruise_ID == ID_1)
prof_poc        = b2_poc[prof_poc_idx]
prof_poc_depth  = b2_depth[prof_poc_idx]

b2_DateTime_1   = b2_DateTime.date[prof_poc_idx]
print(b2_DateTime_1[0])
# Remove nan from POC data
ads             = np.where(~np.isnan(prof_poc))
prof_poc        = prof_poc[ads]
prof_poc_depth  = prof_poc_depth[ads]

## Find Surface Chla for Morel Kd equation
x1             = np.where(prof_depth == np.min(prof_depth[np.nonzero(prof_chla)]))
prof_chla_surf = prof_chla[x1]

# Compute Morel Kd and Zp with function
Kd, Zp = calculate_Kd_Zp(prof_chla_surf)

##################### 1ST CHLOROPHYLL FIT ##################
### Dimensionalise the profiles
CHL_DIM  = prof_chla/prof_chla_surf  #Dimensionless chl
OPT_DIM  = prof_depth*Kd             #Dimensionless tau
MLD_OD   = prof_MLD*Kd               #Optical depth of mixed layer

X1        = CHL_DIM
Y1        = OPT_DIM

## Profile has to have as a minimum of more measurements +1 than parameters (6 for Eq. 7)        
if len(X1) >= 6:
    ###Fit 1st population
    params1  = Parameters()
    params1.add('P1', value=9., min = 4.6, max = 100)
    params1.add('P2', value=MLD_OD[0])#MLD_OD
    out      = Minimizer(fcn2min_1pop, params1, fcn_args=(X1, Y1))
    result_3   = out.minimize(method = 'powell')
    P1_FIT   = result_3.params['P1'].value
    P2_FIT   = result_3.params['P2'].value
    
    C1_P1   = P1_FIT
    C1_TAU1 = P2_FIT
    
    #REDSEA_TAU1 = np.nan
    #REDSEA_P1   = np.nan
    
    AIC_FIT1 = result_3.aic
    CHI_FIT1 = result_3.chisqr
    MLD_pop  = 1 - 1./(1+np.exp(-(P1_FIT/P2_FIT)*(Y1-P2_FIT)))
    r        = np.corrcoef(X1, MLD_pop)
    #report_fit(result_3) ##uncomment if you want to see results of fit
    ###Fit 2nd population
    params2 = Parameters()
    if r[1,0]**2 >= 0.92:
        #REDSEA_P1 = C1_P1
        #REDSEA_TAU1 = C1_TAU1
        P3_FIT = nan
        P4_FIT = nan
        P5_FIT = nan
    else:
    ###Estimate tau1 and S1 from Optical depth of mixed layer  
        
        ### Max of DCM 1
        DCM1_MAX   = np.max(X1)
        ads = np.where(X1 == np.max(X1))
        DCM1_DEPTH = Y1[ads]  ###divide by three to account for nature of equation
        DCM1_DEPTH = DCM1_DEPTH[0]/3
        
        #Fit1
        Tau1_temp = (MLD_OD[0]*0.62)+2.296 # RedSea
        P1_temp   = 10**(0.08 * Tau1_temp + 0.66) # RedSea
        params2.add('P1', value=P1_temp, vary=False) #Fixed
        params2.add('P2', value=Tau1_temp, vary=False) #Fixed
        params2.add('P3', value=DCM1_MAX, min = 0.0, max = 100.0)
        params2.add('P4', value=DCM1_DEPTH, min = 0.0, max = 10)
        params2.add('P5', value=1.0, min = 0.0)
        res      = Minimizer(fcn2min_2pop,  params2, fcn_args=(X1, Y1))
        result_3 = res.minimize(method = 'powell')
        AIC_FIT2 = result_3.aic
        CHI_FIT2 = result_3.chisqr
        #report_fit(result_3) ##uncomment if you want to see results of fit
        
        REDSEA_TAU1 = Tau1_temp
        REDSEA_P1   = P1_temp
        
        P1_FIT = result_3.params['P1'].value
        P2_FIT = result_3.params['P2'].value
        P3_FIT = result_3.params['P3'].value
        P4_FIT = result_3.params['P4'].value
        P5_FIT = result_3.params['P5'].value
        
        P1_TEMP_2   = P1_FIT                                                                                                         
        TAU1_TEMP_2 = P2_FIT
        BM2_TEMP_2  = P3_FIT
        TAU2_TEMP_2 = P4_FIT + P5_FIT * 3.0
        SIG2_TEMP_2 = P5_FIT   

#################### 3RD STEP: 2nd CHLOROPHYLL FIT USING POC TAU1 ###############
        ### First Fit Initial POC
        if ID_1 in ID_list_poc and np.min(prof_poc_depth) <= 1/Kd:
            #Get surface POC
            ASD      = np.where(prof_poc_depth <= 1/Kd)
            SURF_POC = np.median(prof_poc[ASD])
            prof_poc_NOM   = prof_poc/SURF_POC
            # POC Optical depth
            OPT_DIM_POC    = prof_poc_depth*Kd
            
            # POC model for 2 pop
            MLD_pop_FIT_CHL_POC  = (1 - 1./(1+np.exp(-(P1_TEMP_2/TAU1_TEMP_2)*(OPT_DIM_POC-TAU1_TEMP_2))))#*prof_chla_surf
            DCM_pop_FIT_CHL_POC  = (BM2_TEMP_2*np.exp(-((OPT_DIM_POC - TAU2_TEMP_2)/SIG2_TEMP_2)**2.))#*prof_chla_surf
            TOTAL_CHL_POC        = MLD_pop_FIT_CHL_POC + DCM_pop_FIT_CHL_POC
            ###Fit 1st population
            params3  = Parameters()
            params3.add('P2', value= 0.3, min = 0.01, max = 0.95)
            params3.add('P3', value= 0.2, min = 0.01, max = 0.95)
            X11 = prof_poc_NOM
            A1 = MLD_pop_FIT_CHL_POC
            A2 = DCM_pop_FIT_CHL_POC            
            out = Minimizer(fcn2min_2pop_poc, params3, fcn_args=(X11, A1, A2))
            result_4   = out.minimize(method = 'powell')
            #report_fit(result_4)
            P2_POC_RATE   = result_4.params['P2'].value
            P3_POC_RATE   = result_4.params['P3'].value
            P1_POC_RATE   = 1 - P3_POC_RATE
            
            ### 3rd step Fixing P1 RedSea relationship
            params2  = Parameters()
            params2.add('P2', value=TAU1_TEMP_2) # Tau1 still varies
            params2.add('P3', value=P3_FIT, vary=False)
            params2.add('P4', value=P4_FIT, vary=False)
            params2.add('P5', value=P5_FIT, vary=False)
            params2.add('P6', value=P2_POC_RATE, vary=False)
            params2.add('P7', value=P3_POC_RATE, vary=False)
            Y2 = OPT_DIM_POC
            X2 = prof_poc_NOM
            res      = Minimizer(fcn2min_2pop_3rd_step2,  params2, fcn_args=(X2, Y2))
            result_4 = res.minimize(method = 'powell')
            #report_fit(result_4) ##uncomment if you want to see results 
            P2_FIT_3STEP  = result_4.params['P2'].value # Tau1 still varies
            P1_FIT_3STEP  =  10**(0.08 * P2_FIT_3STEP + 0.66) # Red Sea P1 Tau1 Relationship
            
            ### Fit Chl Again with new Fixed P1 and Tau1 from POC Fit
            Tau1_temp = P2_FIT_3STEP # 3rd step
            P1_temp   = P1_FIT_3STEP # 3rd step
            params2.add('P1', value=P1_temp, vary=False) #Fixed
            params2.add('P2', value=Tau1_temp, vary=False) #Fixed
            params2.add('P3', value=P3_FIT, min = 0.0, max = 100.0)
            params2.add('P4', value=P4_FIT, min = 0.0, max = 10)
            params2.add('P5', value=P5_FIT, min = 0.0)
            res      = Minimizer(fcn2min_2pop,  params2, fcn_args=(X1, Y1))
            result_6 = res.minimize(method = 'powell')
            AIC_FIT6 = result_6.aic
            CHI_FIT6 = result_6.chisqr
            #report_fit(result_6) ##uncomment if you want to see results of fit
        
            if AIC_FIT6 < AIC_FIT1: 
                P1_FIT = result_6.params['P1'].value
                P2_FIT = result_6.params['P2'].value
                P3_FIT = result_6.params['P3'].value
                P4_FIT = result_6.params['P4'].value
                P5_FIT = result_6.params['P5'].value
                print("2nd Fit")
            else:
                P1_FIT = C1_P1
                P2_FIT = C1_TAU1
                P3_FIT = nan
                P4_FIT = nan
                P5_FIT = nan
                print("C2 tried but NAN")
                
        else:
            P1_FIT = nan
            P2_FIT = nan
            P3_FIT = nan
            P4_FIT = nan
            P5_FIT = nan
else:
    P1_FIT = nan
    P2_FIT = nan
    P3_FIT = nan
    P4_FIT = nan
    P5_FIT = nan
###Extract parameters from the chlorophyll fit
P1_TEMP   = P1_FIT                                                                                                         
TAU1_TEMP = P2_FIT
BM2_TEMP  = P3_FIT
TAU2_TEMP = P4_FIT + P5_FIT * 3.0
SIG2_TEMP = P5_FIT    

###CHL MODEL
OPT_DIM  = prof_depth*Kd
MLD_pop_FIT   = (1 - 1./(1+np.exp(-(P1_TEMP/TAU1_TEMP)*(OPT_DIM-TAU1_TEMP))))*prof_chla_surf
if np.isnan(BM2_TEMP):
    DCM_pop_FIT   = MLD_pop_FIT *0
else:
    DCM_pop_FIT   = (BM2_TEMP*np.exp(-((OPT_DIM - TAU2_TEMP)/SIG2_TEMP)**2.)) *prof_chla_surf

CHL_model_fit = MLD_pop_FIT + DCM_pop_FIT

#Higher Res data points for plotting
OPT_DIM2       = np.linspace(0,20,2000)
MLD_pop_FIT2   = (1 - 1./(1+np.exp(-(P1_TEMP/TAU1_TEMP)*(OPT_DIM2-TAU1_TEMP))))*prof_chla_surf
if np.isnan(BM2_TEMP):
    DCM_pop_FIT2   = MLD_pop_FIT2 *0
else:
    DCM_pop_FIT2   = (BM2_TEMP*np.exp(-((OPT_DIM2 - TAU2_TEMP)/SIG2_TEMP)**2.)) *prof_chla_surf

CHL_model_fit2 = MLD_pop_FIT2 + DCM_pop_FIT2

prof_depth2  = OPT_DIM2/Kd

print(MLD_OD)
print("MLD: "+str(prof_MLD))

################# POC fit ##################
if ID_1 in ID_list_poc and np.min(prof_poc_depth) <= 1/Kd:
    #Get surface POC
    ASD      = np.where(prof_poc_depth <= 1/Kd)
    SURF_POC = np.median(prof_poc[ASD])
    prof_poc_NOM   = prof_poc/SURF_POC
    # POC Optical depth
    OPT_DIM_POC    = prof_poc_depth*Kd
    
    # POC model for 1 pop
    if np.all(DCM_pop_FIT)==0:
        MLD_pop_FIT_CHL_POC  = (1 - 1./(1+np.exp(-(P1_TEMP/TAU1_TEMP)*(OPT_DIM_POC-TAU1_TEMP))))#*prof_chla_surf
        MLD_pop_FIT2_POC0    = (1 - 1./(1+np.exp(-(P1_TEMP/TAU1_TEMP)*(OPT_DIM2-TAU1_TEMP))))#*prof_chla_surf
        DCM_pop_FIT_CHL_POC  = np.array(range(len(MLD_pop_FIT_CHL_POC))) + nan
        TOTAL_CHL_POC        = MLD_pop_FIT_CHL_POC
        
        # Parameter estimates        
        ###Fit 1st population
        params3  = Parameters()
        # Suface estimate of rate
        params3.add('P3', value= 0.2, min = 0.01, max = 0.95)
        X1 = prof_poc_NOM
        A1 = MLD_pop_FIT_CHL_POC
        out = Minimizer(fcn2min_1pop_poc, params3, fcn_args=(X1, A1))
        result_4   = out.minimize(method = 'powell')
        #report_fit(result_4)
        P3_POC_RATE   = result_4.params['P3'].value
        P1_POC_RATE   = 1 - P3_POC_RATE
        AIC_FIT1 = result_4.aic
        CHI_FIT1 = result_4.chisqr
        
        MLD_pop_FIT_POC = (MLD_pop_FIT_CHL_POC * P1_POC_RATE)*SURF_POC
        DCM_pop_FIT_POC = ((MLD_pop_FIT_POC * 0))*SURF_POC
        BACKGROUND_POC  = ((MLD_pop_FIT_POC * 0) + P3_POC_RATE)*SURF_POC
        TOTAL_POC_MODEL = MLD_pop_FIT_POC + BACKGROUND_POC
        
        ## High resolution POC
        MLD_pop_FIT2_POC = (MLD_pop_FIT2_POC0 * P1_POC_RATE)*SURF_POC
        DCM_pop_FIT2_POC = ((MLD_pop_FIT2_POC * 0))*SURF_POC
        BACKGROUND2_POC  = ((MLD_pop_FIT2_POC * 0) + P3_POC_RATE)*SURF_POC
        POC_model_fit2   = MLD_pop_FIT2_POC + BACKGROUND2_POC
        
        #Chl-specic POC for communties
        CHL_POC_C1 = P1_POC_RATE/(prof_chla_surf/SURF_POC)
        CHL_POC_C2 = np.nan
        print("C:Chl_1 = "+str(CHL_POC_C1))
        
        print('1 Community Fit')
            
    # POC model for 2 pops
    else:
        MLD_pop_FIT_CHL_POC  = (1 - 1./(1+np.exp(-(P1_TEMP/TAU1_TEMP)*(OPT_DIM_POC-TAU1_TEMP))))#*prof_chla_surf
        DCM_pop_FIT_CHL_POC  = (BM2_TEMP*np.exp(-((OPT_DIM_POC - TAU2_TEMP)/SIG2_TEMP)**2.))#*prof_chla_surf
        MLD_pop_FIT2_POC0   = (1 - 1./(1+np.exp(-(P1_TEMP/TAU1_TEMP)*(OPT_DIM2-TAU1_TEMP))))#*prof_chla_surf
        DCM_pop_FIT2_POC0   = (BM2_TEMP*np.exp(-((OPT_DIM2 - TAU2_TEMP)/SIG2_TEMP)**2.))# *prof_chla_surf
        TOTAL_CHL_POC        = MLD_pop_FIT_CHL_POC + DCM_pop_FIT_CHL_POC
        
        ###Fit 1st population
        params3  = Parameters()
        params3.add('P2', value= 0.3, min = 0.01, max = 0.95)
        params3.add('P3', value= 0.2, min = 0.01, max = 0.95)
        X1 = prof_poc_NOM
        A1 = MLD_pop_FIT_CHL_POC
        A2 = DCM_pop_FIT_CHL_POC            
        out = Minimizer(fcn2min_2pop_poc, params3, fcn_args=(X1, A1, A2))
        result_4   = out.minimize(method = 'powell')
        #report_fit(result_4)
        P2_POC_RATE   = result_4.params['P2'].value
        P3_POC_RATE   = result_4.params['P3'].value
        P1_POC_RATE   = 1 - P3_POC_RATE
        #AIC_FIT1 = result_4.aic
        #CHI_FIT1 = result_4.chisqr
        
        MLD_pop_FIT_POC = (MLD_pop_FIT_CHL_POC * P1_POC_RATE)*SURF_POC
        DCM_pop_FIT_POC = (DCM_pop_FIT_CHL_POC * P2_POC_RATE)*SURF_POC
        BACKGROUND_POC  = ((DCM_pop_FIT_CHL_POC * 0) + P3_POC_RATE)*SURF_POC
        TOTAL_POC_MODEL = MLD_pop_FIT_POC + DCM_pop_FIT_POC + BACKGROUND_POC
        
        ## High resolution PP
        MLD_pop_FIT2_POC   = (MLD_pop_FIT2_POC0 * P1_POC_RATE)*SURF_POC
        DCM_pop_FIT2_POC   = (DCM_pop_FIT2_POC0 * P2_POC_RATE)*SURF_POC
        BACKGROUND2_POC  = ((DCM_pop_FIT2_POC * 0) + P3_POC_RATE)*SURF_POC
        POC_model_fit2 = MLD_pop_FIT2_POC + DCM_pop_FIT2_POC + BACKGROUND2_POC
        
        #Chl-specic POC for communties
        CHL_POC_C1 = P1_POC_RATE/(prof_chla_surf/SURF_POC)
        CHL_POC_C2 = P2_POC_RATE/(prof_chla_surf/SURF_POC)
    
    print("C:Chl_1 = "+str(CHL_POC_C1))
    print("C:Chl_2 = "+str(CHL_POC_C2))

# PLOT MODEL FIT TO BOTH CHLA & POC
if ID_1 in ID_list_poc:
    #Plot optical depth
    XSIZE = 10 #Define the xsize of the figure window
    YSIZE = 6 #Define the ysize of the figure window
    
    #Set plot up
    fig, ([ax1, ax3]) = plt.subplots(1,2, figsize=(XSIZE,YSIZE))
    fig.subplots_adjust(wspace=0.08,hspace=0.1)
    fig.patch.set_facecolor('White')
    ### PLOT CHLA MODEL TO SINGLE PROFILE
    ax1.plot([np.min(prof_chla)-0.025,np.max(prof_chla)+(prof_chla[0]*0.3)],[prof_MLD,prof_MLD],
             color = 'k', marker = 'None', linestyle = '--', label= 'MLD')
    ax1.plot(prof_chla,prof_depth, \
             color = 'g', marker = 'X',markersize =8, linestyle = '-', label= 'Data')
    ax1.plot(CHL_model_fit,prof_depth, \
             color = 'k', marker = 'o', linestyle = 'None',label= 'Total')
    ax1.plot(MLD_pop_FIT,prof_depth, \
             color = 'r', marker = 'o', linestyle = 'None', label= 'Surface')
    ax1.plot(DCM_pop_FIT,prof_depth, \
             color = 'b', marker = 'o', linestyle = 'None',label= 'Subsurface')
    ax1.plot(CHL_model_fit2,prof_depth2, \
             color = 'k',  linestyle = '-',label= None)
    ax1.plot(MLD_pop_FIT2,prof_depth2, \
             color = 'r', linestyle = '-', label= None)
    ax1.plot(DCM_pop_FIT2,prof_depth2, \
             color = 'b',  linestyle = '-',label= None)
    ax1.set_title('(a)', fontsize = 19, color='k')
    ax1.set_ylabel('Depth (m)', fontsize=19)
    ax1.yaxis.set_tick_params(labelsize=16)
    ax1.set_ylim([210,0]) 
    ax1.set_xlabel('Chl-a (mg m$^{-3}$)', fontsize=19, color = 'k')
    ax1.xaxis.set_tick_params(labelsize=16)
    #ax1.set_xlim(xmin=-0.015, xmax=np.max(prof_chla)+0.5)
    if np.nanmax(prof_chla) > np.nanmax(CHL_model_fit):
        ax1.set_xlim(xmin= -0.015, xmax=np.max(prof_chla)+(prof_chla[0]*0.2))
    else:
        ax1.set_xlim(xmin= -0.015, xmax=np.max(CHL_model_fit)+(CHL_model_fit[0]*0.2))
    ax1.legend(loc="lower right", fontsize=12)
    #ax1.text(np.min(prof_chla)+0.02, 205, "ID: "+str(ID_1)+"\nDate: "+str(b_DateTime_1[0]), color='k', fontsize=12)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    ### PLOT POC Model Fit ###
    ax3.plot([np.min(MLD_pop_FIT_POC)-1,np.max(prof_poc)+(prof_poc[0]*0.3)],[prof_MLD,prof_MLD], \
             color = 'k', marker = 'None', linestyle = '--', label= 'MLD')
    ax3.plot(prof_poc,prof_poc_depth, \
             color = 'orange',  marker = 'X', markersize = 8, linestyle = '-',label= 'Data')
    ax3.plot(BACKGROUND_POC,prof_poc_depth, \
             color = 'm', marker = '.', markersize = 8, linestyle = '--',label= 'Non-Algal Bk')
    ax3.plot(MLD_pop_FIT_POC,prof_poc_depth, \
             color = 'r', marker = 'o', linestyle = 'None', label= 'Surface')
    ax3.plot(DCM_pop_FIT_POC,prof_poc_depth, \
             color = 'b', marker = 'o', linestyle = 'None',label= 'Subsurface')
    ax3.plot(MLD_pop_FIT2_POC,prof_depth2, \
             color = 'r', linestyle = '-', label= None)
    ax3.plot(DCM_pop_FIT2_POC,prof_depth2, \
             color = 'b',  linestyle = '-',label= None)
    ax3.plot(TOTAL_POC_MODEL,prof_poc_depth, \
             color = 'k', marker = 'o', linestyle = 'None',label= 'Total')
    ax3.plot(POC_model_fit2,prof_depth2, \
             color = 'k',  linestyle = '-',label= None)
    ax3.set_title('(b)', fontsize = 19, color='k')
    #ax3.set_ylabel('Depth (m)', fontsize=17)
    ax3.yaxis.set_tick_params(labelsize=16)
    ax3.set(yticklabels=[])
    ax3.set_ylim([210,0]) 
    ax3.set_xlabel('POC (mg m$^{-3}$)', fontsize=19, color = 'k')
    ax3.xaxis.set_tick_params(labelsize=16)
    if np.nanmax(prof_poc) > np.nanmax(TOTAL_POC_MODEL):
        ax3.set_xlim(xmin= -1, xmax=np.max(prof_poc)+(prof_poc[0]*0.2))
    else:
        ax3.set_xlim(xmin= -1, xmax=np.max(TOTAL_POC_MODEL)+(TOTAL_POC_MODEL[0]*0.2))
    ax3.legend(loc="lower right", fontsize=12)
    #ax3.text(1, 260, " Date: "+str(b_DateTime_1[0]), color='k', fontsize=12)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax3.yaxis.set_major_locator(plt.MaxNLocator(5))
    #ax3.set_xscale('log')
    fig.savefig('plots/BATS_ModelFit_Chl_POC_'+str(ID_1)+'.png', format='png', dpi=300, bbox_inches="tight")
# =============================================================================
#     fig.savefig('plots/BATS_ModelFit_Chl_POC_Tranparent_'+str(ID_1)+'_poster.png', format='png',
#                 dpi=300, bbox_inches="tight", transparent=True)
# =============================================================================
    plt.show()

#%%

### LOOP MODEL FIT ###

#Include plotting of every profile model fitted?
model_plot = False #If true will plot both Chla & POC model fits for each profile loop
solve_method = 'powell'#leastsq, powell

# Fit Model to which Chla parameter?
m_chla = b_chla

print(np.max(b_chla))

# ID list to loop over
ID_list_bottle = ID_list_6

# Arrays to store Chl Model results
MLD_pop_FIT     = np.array(range(len(m_chla))) + nan
DCM_pop_FIT     = np.array(range(len(m_chla))) + nan
CHL_model_fit   = np.array(range(len(m_chla))) + nan
CHL_model_input = np.array(range(len(m_chla))) + nan # record chla data used
CHL_diff        = np.array(range(len(m_chla))) + nan # array to store difference

# Arrays to store POC model results
MLD_pop_FIT_POC  = np.array(range(len(b2_poc))) + nan
DCM_pop_FIT_POC  = np.array(range(len(b2_poc))) + nan
TOTAL_POC_MODEL  = np.array(range(len(b2_poc))) + nan
POC_diff         = np.array(range(len(b2_poc))) + nan
BACKGROUND_POC   = np.array(range(len(b2_poc))) + nan
POC_model_input  = np.array(range(len(b2_poc))) + nan

#Chl-specic POC for communties - C:Chl ratio
CHL_POC_C1 = np.array(range(len(ID_list_poc))) + nan
CHL_POC_C2 = np.array(range(len(ID_list_poc))) + nan

#Chl-specic POC scaling factors for communties & background C* to plot model parameters
CHL_POC_C1_d = np.array(range(len(ID_list_poc))) + nan
CHL_POC_C2_d = np.array(range(len(ID_list_poc))) + nan
POC_bk_d = np.array(range(len(ID_list_poc))) + nan

# Arrays to store MLD and IDs for POC dataframe (might not be needed )
MLD_model_fit_POC = np.array(range(len(ID_list_poc))) + nan
df_poc_ID = np.array(range(len(ID_list_poc))) + nan
df_poc_date = np.array(range(len(ID_list_poc)), dtype='datetime64[s]')
df_poc_DecYear = np.array(range(len(ID_list_poc))) + nan

# Arrays to store model parameters for each profile
P1_model_fit  = np.array(range(len(ID_list_bottle))) + nan
P2_model_fit  = np.array(range(len(ID_list_bottle))) + nan
P3_model_fit  = np.array(range(len(ID_list_bottle))) + nan
P4_model_fit  = np.array(range(len(ID_list_bottle))) + nan
P5_model_fit  = np.array(range(len(ID_list_bottle))) + nan
MLD_model_fit = np.array(range(len(ID_list_bottle))) + nan
KD_mod        = np.array(range(len(ID_list_bottle))) + nan
CHL_surf_mod  = np.array(range(len(ID_list_bottle))) + nan
zp  = np.array(range(len(ID_list_bottle))) + nan
#KD_mod2 = np.empty(len(b_ID))+nan

count = 0
pp_count = 0
poc_count = 0 # count to save for POC single prof values
for i in ID_list_bottle:
    #i = 10256008 
    where_bottle_idx  = np.where(bottle_6.Cruise_ID == i)
    prof_date         = np.unique(b_date[where_bottle_idx])
    prof_MLD_idx      = np.where(ID_list_ctd == i)
    prof_MLD          = MLD[prof_MLD_idx]
    #prof_PP_idx       = np.where(PPpd.Cruise_ID == i)
    
    if prof_MLD >= 0:
        MLD_model_fit[count] = prof_MLD
    
        prof_depth   = b_depth[where_bottle_idx]
        prof_chla    = m_chla[where_bottle_idx]
        prof_date    = np.unique(b_date[where_bottle_idx])
        
        b_DateTime_1 = b_DateTime.date[where_bottle_idx]
        prof_date    = b_DateTime_1[0]
        
        # POC data
        prof_poc_idx    = np.where(bottle_poc.cruise_ID == i)
        prof_poc        = b2_poc[prof_poc_idx]
        prof_poc_depth  = b2_depth[prof_poc_idx]
        prof_poc_DecYear = b2_DecYear[prof_poc_idx]
        
        # CHL used in model
        CHL_model_input[where_bottle_idx] = prof_chla
        
        #Get Surface Chla for Kd computation
        x1        = np.where(prof_depth == np.min(prof_depth[np.nonzero(prof_chla)])) # to account for some profiles where top chla measurement is zero, take next one as as surface.
        prof_chla_surf = prof_chla[x1]
    
        # Compute Morel Kd and Zp with function
        Kd, Zp = calculate_Kd_Zp(prof_chla_surf)
        # Store Kd & Euphotic depth (Zp)
        KD_mod[count]=Kd
        zp[count] = Zp
        
        # Save surface Chla used to dimensionalise profile and compute Kd
        CHL_surf_mod[count] = prof_chla_surf
        
        #####################Chlorophyll fit##################
        ###Dimensionalise the profiles
        CHL_DIM  = prof_chla/prof_chla_surf  #Dimensionless chl
        OPT_DIM  = prof_depth*Kd   #Dimensionless tau
        MLD_OD   = prof_MLD*Kd     #Optical depth of mixed layer
    
        ###Process data to be used for Chl fit
        X1        = CHL_DIM
        Y1        = OPT_DIM
        
        ## Profile has to have as a minimum of more measurements +1 than parameters (6 for Eq. 7)        
        if len(X1) > 5:
            ###Fit 1st population
            params1  = Parameters()
            params1.add('P1', value=9., min = 4.6, max = 100)
            params1.add('P2', value=MLD_OD[0])#MLD_OD
            out      = Minimizer(fcn2min_1pop, params1, fcn_args=(X1, Y1))
            result_3   = out.minimize(method = 'powell')
            P1_FIT   = result_3.params['P1'].value
            P2_FIT   = result_3.params['P2'].value
            
            C1_P1   = P1_FIT
            C1_TAU1 = P2_FIT
            
            AIC_FIT1 = result_3.aic
            CHI_FIT1 = result_3.chisqr
            MLD_pop  = 1 - 1./(1+np.exp(-(P1_FIT/P2_FIT)*(Y1-P2_FIT)))
            r        = np.corrcoef(X1, MLD_pop)
            #report_fit(result_3) ##uncomment if you want to see results of fit
            ###Fit 2nd population
            params2 = Parameters()
            if r[1,0]**2 >= 0.92:
                P3_FIT = nan
                P4_FIT = nan
                P5_FIT = nan
            else:
            ###Estimate tau1 and S1 from Optical depth of mixed layer  
                
                ### Max of DCM 1
                DCM1_MAX   = np.max(X1)
                ads = np.where(X1 == np.max(X1))
                DCM1_DEPTH = Y1[ads]  ###divide by three to account for nature of equation
                DCM1_DEPTH = DCM1_DEPTH[0]/3
                
                #Fit1
                #Tau1_temp = (MLD_OD[0]*0.62)+3.843 #BATS
                #P1_temp   = 10**(0.05 * Tau1_temp + 0.663) #BATS
                Tau1_temp = (MLD_OD[0]*0.62)+2.296 # RedSea
                P1_temp   = 10**(0.08 * Tau1_temp + 0.66) # RedSea
                params2.add('P1', value=P1_temp, vary=False) #Fixed
                params2.add('P2', value=Tau1_temp, vary=False) #Fixed
                params2.add('P3', value=DCM1_MAX, min = 0.0, max = 100.0)
                params2.add('P4', value=DCM1_DEPTH, min = 0.0, max = 10)
                params2.add('P5', value=1.0, min = 0.0)
                res      = Minimizer(fcn2min_2pop,  params2, fcn_args=(X1, Y1))
                result_3 = res.minimize(method = 'powell')
                AIC_FIT2 = result_3.aic
                CHI_FIT2 = result_3.chisqr
                #report_fit(result_3) ##uncomment if you want to see results of fit
                
                P1_FIT = result_3.params['P1'].value
                P2_FIT = result_3.params['P2'].value
                P3_FIT = result_3.params['P3'].value
                P4_FIT = result_3.params['P4'].value
                P5_FIT = result_3.params['P5'].value
                
                P1_TEMP_2   = P1_FIT                                                                                                         
                TAU1_TEMP_2 = P2_FIT
                BM2_TEMP_2  = P3_FIT
                TAU2_TEMP_2 = P4_FIT + P5_FIT * 3.0
                SIG2_TEMP_2 = P5_FIT   

        #################### ATTEMPT AT 3RD STEP IN CHL FIT ###############
                ### POC
                if i in ID_list_poc and np.min(prof_poc_depth) <= 1/Kd:
                    ################# POC fit ##################
                    #Get surface POC
                    ASD      = np.where(prof_poc_depth <= 1/Kd)
                    SURF_POC = np.median(prof_poc[ASD])
                    prof_poc_NOM   = prof_poc/SURF_POC
                    # Cp Optical depth
                    OPT_DIM_POC    = prof_poc_depth*Kd       
                    # POC model for 1 pop 
                
                    MLD_pop_FIT_CHL_POC  = (1 - 1./(1+np.exp(-(P1_TEMP_2/TAU1_TEMP_2)*(OPT_DIM_POC-TAU1_TEMP_2))))#*prof_chla_surf
                    DCM_pop_FIT_CHL_POC  = (BM2_TEMP_2*np.exp(-((OPT_DIM_POC - TAU2_TEMP_2)/SIG2_TEMP_2)**2.))#*prof_chla_surf
                    TOTAL_CHL_POC        = MLD_pop_FIT_CHL_POC + DCM_pop_FIT_CHL_POC
                    ###Fit 1st population
                    params3  = Parameters()
                    params3.add('P2', value= 0.3, min = 0.01, max = 0.95)
                    params3.add('P3', value= 0.2, min = 0.01, max = 0.95)
                    X11 = prof_poc_NOM
                    A1 = MLD_pop_FIT_CHL_POC
                    A2 = DCM_pop_FIT_CHL_POC            
                    out = Minimizer(fcn2min_2pop_poc, params3, fcn_args=(X11, A1, A2))
                    result_4   = out.minimize(method = 'powell')
                    #report_fit(result_4)
                    P2_POC_RATE   = result_4.params['P2'].value # Subsurface scaling factor
                    P3_POC_RATE   = result_4.params['P3'].value # Background on algal
                    P1_POC_RATE   = 1 - P3_POC_RATE # Surface POC scaling factor
                                   
                    ### 3rd step Fixing P1 RedSea relationship
                    params2  = Parameters()
                    params2.add('P2', value=TAU1_TEMP_2) # Tau1 still varies
                    params2.add('P3', value=P3_FIT, vary=False)
                    params2.add('P4', value=P4_FIT, vary=False)
                    params2.add('P5', value=P5_FIT, vary=False)
                    params2.add('P6', value=P2_POC_RATE, vary=False)
                    params2.add('P7', value=P3_POC_RATE, vary=False)
                    Y2 = OPT_DIM_POC
                    X2 = prof_poc_NOM
                    res      = Minimizer(fcn2min_2pop_3rd_step2,  params2, fcn_args=(X2, Y2))
                    result_4 = res.minimize(method = 'powell')
                    #report_fit(result_4) ##uncomment if you want to see results 
                    P2_FIT_3STEP  = result_4.params['P2'].value # Tau1 still varies
                    P1_FIT_3STEP  =  10**(0.08 * P2_FIT_3STEP + 0.66) # Red Sea Relationship
                    
                    ### Fit Chl Again with new Fixed P1 and Tau1 from POC Fit
                    Tau1_temp = P2_FIT_3STEP # 3rd step
                    P1_temp   = P1_FIT_3STEP # 3rd step
                    params2.add('P1', value=P1_temp, vary=False) #Fixed
                    params2.add('P2', value=Tau1_temp, vary=False) #Fixed
                    params2.add('P3', value=P3_FIT, min = 0.0, max = 100.0)
                    params2.add('P4', value=P4_FIT, min = 0.0, max = 10)
                    params2.add('P5', value=P5_FIT, min = 0.0)
                    res      = Minimizer(fcn2min_2pop,  params2, fcn_args=(X1, Y1))
                    result_6 = res.minimize(method = 'powell')
                    AIC_FIT6 = result_6.aic
                    CHI_FIT6 = result_6.chisqr
                    #report_fit(result_6) ##uncomment if you want to see results of fit
                
                    if AIC_FIT6 < AIC_FIT1: 
                        P1_FIT = result_6.params['P1'].value
                        P2_FIT = result_6.params['P2'].value
                        P3_FIT = result_6.params['P3'].value
                        P4_FIT = result_6.params['P4'].value
                        P5_FIT = result_6.params['P5'].value
                        print("2nd Fit")
                    
                    else:
                        P1_FIT = C1_P1
                        P2_FIT = C1_TAU1
                        #REDSEA_P1 = C1_P1
                        #REDSEA_TAU1 = C1_TAU1
                        P3_FIT = nan
                        P4_FIT = nan
                        P5_FIT = nan
                        print("C2 tried but NAN")
                else:
                    P1_FIT = nan
                    P2_FIT = nan
                    P3_FIT = nan
                    P4_FIT = nan
                    P5_FIT = nan
        else:
            P1_FIT = nan
            P2_FIT = nan
            P3_FIT = nan
            P4_FIT = nan
            P5_FIT = nan
        ###Extract parameters from the chlorophyll fit
        P1_TEMP   = P1_FIT                                                                                                         
        TAU1_TEMP = P2_FIT
        BM2_TEMP  = P3_FIT
        TAU2_TEMP = P4_FIT + P5_FIT * 3.0
        SIG2_TEMP = P5_FIT    
        
        P1_model_fit[count] = P1_TEMP
        P2_model_fit[count] = TAU1_TEMP
        P3_model_fit[count] = BM2_TEMP
        P4_model_fit[count] = TAU2_TEMP
        P5_model_fit[count] = SIG2_TEMP
    
        ###CHL MODEL
        OPT_DIM  = prof_depth*Kd
        if np.isnan(BM2_TEMP):
            MLD_pop_FIT[where_bottle_idx]   = (1 - 1./(1+np.exp(-(P1_TEMP/TAU1_TEMP)*(OPT_DIM-TAU1_TEMP))))*prof_chla_surf
            DCM_pop_FIT[where_bottle_idx]   = MLD_pop_FIT[where_bottle_idx] *0
        else:
            MLD_pop_FIT[where_bottle_idx]   = (1 - 1./(1+np.exp(-(P1_TEMP/TAU1_TEMP)*(OPT_DIM-TAU1_TEMP))))*prof_chla_surf
            DCM_pop_FIT[where_bottle_idx]   = (BM2_TEMP*np.exp(-((OPT_DIM - TAU2_TEMP)/SIG2_TEMP)**2.)) *prof_chla_surf
        
        CHL_model_fit[where_bottle_idx] = MLD_pop_FIT[where_bottle_idx] + DCM_pop_FIT[where_bottle_idx]
        CHL_diff[where_bottle_idx]      = CHL_model_fit[where_bottle_idx] - prof_chla
        
        ##################### POC fit ##################
        if i in ID_list_poc and np.min(prof_poc_depth) <= 1/Kd: #at least 1 POC measurement in first optical depth
            pp_count = pp_count + 1
            
            POC_model_input[prof_poc_idx] = prof_poc
            #Get surface POC
            ASD      = np.where(prof_poc_depth <= 1/Kd)
            #SURF_POC = prof_poc[0]
            SURF_POC = np.median(prof_poc[ASD])
            prof_poc_NOM   = prof_poc/SURF_POC
            # POC Optical depth
            OPT_DIM_POC    = prof_poc_depth*Kd
            
            # POC model for 1 pop
            if np.all(DCM_pop_FIT[where_bottle_idx])==0:
                print('POC Fit - 1 Community')
                MLD_pop_FIT_CHL_POC  = (1 - 1./(1+np.exp(-(P1_TEMP/TAU1_TEMP)*(OPT_DIM_POC-TAU1_TEMP))))#*prof_chla_surf
                DCM_pop_FIT_CHL_POC  = np.array(range(len(MLD_pop_FIT_CHL_POC))) + nan
                TOTAL_CHL_POC        = MLD_pop_FIT_CHL_POC
        
                ###Fit 1st population
                params3  = Parameters()
                # Suface estimate of rate
                params3.add('P3', value= 0.2, min = 0.01, max = 0.95)
                X1 = prof_poc_NOM
                A1 = MLD_pop_FIT_CHL_POC
                out = Minimizer(fcn2min_1pop_poc, params3, fcn_args=(X1, A1))
                result_4   = out.minimize(method= solve_method)
                #report_fit(result_4)
                P3_POC_RATE   = result_4.params['P3'].value
                P1_POC_RATE   = 1 - P3_POC_RATE
                AIC_FIT1 = result_4.aic
                CHI_FIT1 = result_4.chisqr
                
                MLD_pop_FIT_POC[prof_poc_idx] = (MLD_pop_FIT_CHL_POC * P1_POC_RATE)*SURF_POC
                DCM_pop_FIT_POC[prof_poc_idx] = (MLD_pop_FIT_POC[prof_poc_idx]*0)*SURF_POC
                BACKGROUND_POC[prof_poc_idx]  = ((MLD_pop_FIT_POC[prof_poc_idx] * 0) + P3_POC_RATE)*SURF_POC
                TOTAL_POC_MODEL[prof_poc_idx] = MLD_pop_FIT_POC[prof_poc_idx] + BACKGROUND_POC[prof_poc_idx]
                POC_diff[prof_poc_idx]        = TOTAL_POC_MODEL[prof_poc_idx] - prof_poc
                
                #Chl-specic POC for communties
                CHL_POC_C1_d[poc_count] = P1_POC_RATE
                CHL_POC_C2_d[poc_count] = np.nan
                POC_bk_d[poc_count]     = P3_POC_RATE
                CHL_POC_C1[poc_count] = P1_POC_RATE/(prof_chla_surf/SURF_POC)
                CHL_POC_C2[poc_count] = np.nan
                
                df_poc_ID[poc_count]   = i
                df_poc_date[poc_count] = prof_date
                df_poc_DecYear[poc_count] = prof_poc_DecYear[0]

            # POC model for 2 pops
            else:
                print('POC Fit - 2 Communities')
                
                MLD_pop_FIT_CHL_POC  = (1 - 1./(1+np.exp(-(P1_TEMP/TAU1_TEMP)*(OPT_DIM_POC-TAU1_TEMP))))#*prof_chla_surf
                DCM_pop_FIT_CHL_POC  = (BM2_TEMP*np.exp(-((OPT_DIM_POC - TAU2_TEMP)/SIG2_TEMP)**2.))#*prof_chla_surf
                TOTAL_CHL_POC        = MLD_pop_FIT_CHL_POC + DCM_pop_FIT_CHL_POC
        
                ###Fit 1st population
                params3  = Parameters()
                params3.add('P2', value= 0.3, min = 0.01, max = 0.95)
                params3.add('P3', value= 0.2, min = 0.01, max = 0.95)
                X1 = prof_poc_NOM
                A1 = MLD_pop_FIT_CHL_POC
                A2 = DCM_pop_FIT_CHL_POC            
                out = Minimizer(fcn2min_2pop_poc, params3, fcn_args=(X1, A1, A2))
                result_4   = out.minimize(method= solve_method)
                #report_fit(result_4)
                P2_POC_RATE   = result_4.params['P2'].value
                P3_POC_RATE   = result_4.params['P3'].value
                P1_POC_RATE   = 1 - P3_POC_RATE
                AIC_FIT1 = result_4.aic
                CHI_FIT1 = result_4.chisqr
                
                MLD_pop_FIT_POC[prof_poc_idx] = (MLD_pop_FIT_CHL_POC * P1_POC_RATE)*SURF_POC
                DCM_pop_FIT_POC[prof_poc_idx] = (DCM_pop_FIT_CHL_POC * P2_POC_RATE)*SURF_POC
                BACKGROUND_POC[prof_poc_idx]  = ((DCM_pop_FIT_POC[prof_poc_idx] * 0) + P3_POC_RATE)*SURF_POC
                TOTAL_POC_MODEL[prof_poc_idx] = MLD_pop_FIT_POC[prof_poc_idx] + DCM_pop_FIT_POC[prof_poc_idx] + BACKGROUND_POC[prof_poc_idx]
                POC_diff[prof_poc_idx]        = TOTAL_POC_MODEL[prof_poc_idx] - prof_poc

                #Chl-specic POC for communties
                CHL_POC_C1_d[poc_count] = P1_POC_RATE #Dimensionless POC scaling factor
                CHL_POC_C2_d[poc_count] = P2_POC_RATE
                POC_bk_d[poc_count]     = P3_POC_RATE
                CHL_POC_C1[poc_count] = P1_POC_RATE/(prof_chla_surf/SURF_POC)
                CHL_POC_C2[poc_count] = P2_POC_RATE/(prof_chla_surf/SURF_POC)
                
                # Save ID and Date of POC profiles fitted
                df_poc_ID[poc_count]   = i
                df_poc_date[poc_count] = prof_date
                df_poc_DecYear[poc_count] = prof_poc_DecYear[0]
                
        if model_plot == True:
            #PLOT SUBPLOT PANEL
            if i in ID_list_poc and np.min(prof_poc_depth) <= 1/Kd:
                ### PLOT CHLA MODEL TO SINGLE PROFILE
                
                # Model variables for plot
                chla_surf  = MLD_pop_FIT[where_bottle_idx]
                chla_sub   = DCM_pop_FIT[where_bottle_idx]
                chla_total = CHL_model_fit[where_bottle_idx]
                
                XSIZE = 12 #Define the xsize of the figure window
                YSIZE = 6 #Define the ysize of the figure window
                
                #Set plot up
                fig, ([ax1, ax3]) = plt.subplots(1,2, figsize=(XSIZE,YSIZE), constrained_layout=True, \
                gridspec_kw={'hspace': 0.2})
                fig.patch.set_facecolor('White')
    
                ax1.plot([np.min(chla_sub)-0.025,np.max(prof_chla)+0.1],[prof_MLD,prof_MLD],color = 'k', marker = 'None', linestyle = '--', label= 'MLD')
                ax1.plot(prof_chla,prof_depth, \
                         color = 'g', marker = 'o',markersize =8, label= 'HPLC Chl-a')
                ax1.plot(chla_total,prof_depth, \
                         color = 'k', marker = 'o',markersize =6,label= 'CHLA_MOD')
                ax1.plot(chla_surf,prof_depth, \
                         color = 'r', marker = 'o',markersize =5, label= 'SURF_CHLA')
                ax1.plot(chla_sub,prof_depth, \
                         color = 'b', marker = 'o',markersize =4,label= 'SUB_CHLA')
                ax1.set_ylabel('Depth (m)', fontsize=16)
                ax1.yaxis.set_tick_params(labelsize=15)
                ax1.set_ylim([300,0]) 
                ax1.set_xlabel('Chl-a concentration (µg/l)', fontsize=16, color = 'k')
                ax1.xaxis.set_tick_params(labelsize=15)
                ax1.set_xlim(xmin=-0.015, xmax=np.max(prof_chla)+0.1)
                ax1.legend(loc="lower right", fontsize=10,title= 'ID:'+str(i))
                ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
                ax1.text(np.min(chla_sub)+0.01, 290, "Date: "+str(prof_date), color='k', fontsize=12)
                
                ### PLOT POC MODEL TO SINGLE PROFILE
                
                # Model variables for plot
                poc_surf  = MLD_pop_FIT_POC[prof_poc_idx]
                poc_sub   = DCM_pop_FIT_POC[prof_poc_idx]
                poc_total = TOTAL_POC_MODEL[prof_poc_idx]
                poc_bk    = BACKGROUND_POC[prof_poc_idx]
    
                ax3.plot([np.min(poc_surf)-1,np.max(prof_poc)+(prof_poc[0]*0.3)],[prof_MLD,prof_MLD], \
                         color = 'k', marker = 'None', linestyle = '--', label= 'MLD')
                ax3.plot(poc_surf,prof_poc_depth, \
                         color = 'r', marker = 'o', label= 'SURF_POC')
                ax3.plot(poc_sub,prof_poc_depth, \
                         color = 'b', marker = 'o', label= 'SUB_POC')
                ax3.plot(poc_bk,prof_poc_depth, \
                         color = 'm', marker = '.', markersize = 8, linestyle = '--',label= 'Background')
                ax3.plot(prof_poc,prof_poc_depth, \
                         color = 'orange',  marker = 'X', markersize = 8, linestyle = '-',label= 'POC')
                ax3.plot(poc_total,prof_poc_depth, \
                         color = 'k', marker = 'o', label= 'POC_MOD')
                ax3.set_ylabel('Depth (m)', fontsize=16)
                ax3.yaxis.set_tick_params(labelsize=15)
                ax3.set_ylim([270,0]) 
                ax3.set_xlabel('POC (ug/kg)', fontsize=16, color = 'k')
                ax3.xaxis.set_tick_params(labelsize=15)
                if np.nanmax(prof_poc) > np.nanmax(poc_total):
                    ax3.set_xlim(xmin= -1, xmax=np.max(prof_poc)+(prof_poc[0]*0.2))
                else:
                    ax3.set_xlim(xmin= -1, xmax=np.max(poc_total)+(poc_total[0]*0.2))
                ax3.text(1, 260, " Date: "+str(b_DateTime_1[0]), color='k', fontsize=12)
                ax3.legend(loc="lower right", fontsize=10,title= 'ID:'+str(i))
                ax3.xaxis.set_major_locator(plt.MaxNLocator(5))
                #ax3.set_xscale('log')
                plt.show()
                
                fig.savefig('plots/LoopFit/Panels/Chl_POC_ModelFit_'+str(i)+'.png', format='png', dpi=300, bbox_inches="tight")
                plt.show()
        
        print(i) 
        if i in ID_list_poc:
            MLD_model_fit_POC[poc_count] = prof_MLD
            poc_count = poc_count + 1 # count to save for POC single prof values
  
    print(count)
    count=count+1

print("CHLA Fit Possible: "+ str(np.count_nonzero(~np.isnan(MLD_model_fit))))
print("CHLA Fitted: "+ str(np.count_nonzero(~np.isnan(P1_model_fit))))
print("POC Fit: "+ str(pp_count))
print("POC Count: "+ str(poc_count))

#%%

### PLOT SCATTER OF RAW Data VS MODEL Results & PEARSON CORRELATION ###

# Remove NaNs from data
#Chla
ads          = np.where(~np.isnan(CHL_model_fit))
CHL_model_1  = CHL_model_fit[ads]
CHL_raw_1    = CHL_model_input[ads]
b_depth_1    = b_depth[ads]

#POC
ads          = np.where(~np.isnan(TOTAL_POC_MODEL))
POC_model_1  = TOTAL_POC_MODEL[ads]
POC_raw_1    = POC_model_input[ads]
b2_depth_1   = b2_depth[ads]

# Correlation of Raw Chla vs Model Chla
STATS_REG  = spearmanr(CHL_raw_1, CHL_model_1)
#R value
R_chla     = ("{0:.2f}".format(STATS_REG[0]))
#P value
P_chla     = ("{0:.3f}".format(STATS_REG[1]))
print([R_chla,P_chla])

# Correlation of Raw POC vs Model POC
STATS_REG  = spearmanr(POC_raw_1, POC_model_1)
#R value
R_poc      = ("{0:.2f}".format(STATS_REG[0]))
#P value
P_poc      = ("{0:.3f}".format(STATS_REG[1]))
print([R_poc,P_poc])

#Set plot up
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,6))
fig.subplots_adjust(wspace=0.26,hspace=0.2)
fig.patch.set_facecolor('White')

#Chla Subplot
im1 = ax1.scatter(CHL_raw_1,CHL_model_1, c = b_depth_1,alpha = 0.5,cmap = 'viridis_r', 
            label = 'R = '+str(R_chla)+'; $p$ = '+str(P_chla))
ax1.set_title('(a) Chl-a', fontsize = 18, color='k')
ax1.set_ylabel('Model Chl-a (mg m$^{-3}$)', fontsize=16)
ax1.yaxis.set_tick_params(labelsize=15)
ax1.set_xlabel('Raw Chl-a  (mg m$^{-3}$)', fontsize=16)
ax1.xaxis.set_tick_params(labelsize=15)
ax1.legend(loc="upper left", fontsize=12,title= 'Spearman Correlation', title_fontsize=14)
ax1.set_xlim([-0.05,1.21]) 
ax1.set_ylim([-0.05,1.21]) 
ax1.locator_params(nbins=7)
cbar1 = fig.colorbar(im1,ax=ax1)
cbar1.ax.locator_params(nbins=6)
cbar1.set_label("Depth (m)", size  = 16)
cbar1.ax.tick_params(labelsize = 15)

#POC Subplot
im2 = ax2.scatter(POC_raw_1,POC_model_1, c = b2_depth_1,alpha = 0.5,cmap = 'viridis_r', 
            label = 'R = '+str(R_poc)+'; $p$ = '+str(P_poc))
ax2.set_title('(b) POC', fontsize = 18, color='k')
ax2.set_ylabel('Model POC (mg m$^{-3}$)', fontsize=16)
ax2.yaxis.set_tick_params(labelsize=15)
ax2.set_xlabel('Raw POC (mg m$^{-3}$)', fontsize=16)
ax2.xaxis.set_tick_params(labelsize=15)
ax2.legend(loc="upper left", fontsize=12,title= 'Spearman Correlation', title_fontsize=14)
ax2.set_xlim([-5,141]) 
ax2.set_ylim([-5,141]) 
ax2.locator_params(nbins=8)
cbar2 = fig.colorbar(im2,ax=ax2)
cbar2.ax.locator_params(nbins=6)
cbar2.set_label("Depth (m)", size  = 16)
cbar2.ax.tick_params(labelsize = 15)

fig.savefig('plots/BATS_Scatter_Raw_vs_ModelResults_CHL_POC_Spearman.png', format='png', dpi=300, bbox_inches="tight")
plt.show()
sns.reset_orig()

#%%

### SAVE MODEL RESULTS TO DATAFRAMES & EXPORT TO CSV ###

# Add Chla model results to bottle pigment df
bottle_6['surf_model']  = MLD_pop_FIT
bottle_6['sub_model']   = DCM_pop_FIT
bottle_6['total_model'] = CHL_model_fit
bottle_6['CHL_used']    = CHL_model_input
bottle_6['model_diff']  = CHL_diff

# Summary stats for surface & subsurface Chla
bottle_6[['surf_model','sub_model']].describe()

# Save dataframe with Chla Model output to CSV
bottle_6.to_csv('data/BATS_Bottle_Pigments_ModelResults.csv')

# Add single bottle profile data to separate bottle prof df
print(len(bottle_prof))
print(len(P3_model_fit))
bottle_prof['Kd']           = KD_mod
bottle_prof['Zp']           = zp
bottle_prof['MLD_model']    = MLD_model_fit
bottle_prof['CHL_surf_mod'] = CHL_surf_mod
bottle_prof['P1_TEMP']      = P1_model_fit #P1_TEMP
bottle_prof['TAU1_TEMP']    = P2_model_fit #TAU1_TEMP
bottle_prof['BM2_TEMP']     = P3_model_fit #BM2_TEMP = DCM max
bottle_prof['TAU2_TEMP']    = P4_model_fit #TAU2_TEMP = Depth of DCM max
bottle_prof['SIG2_TEMP']    = P5_model_fit #SIG2_TEMP
bottle_prof['DCM_peak']     = bottle_prof['BM2_TEMP']*bottle_prof['CHL_surf_mod']
bottle_prof['DCM_depth']    = bottle_prof['TAU2_TEMP']/bottle_prof['Kd']

# Check DCM depth summary stats
bottle_prof['DCM_depth'].describe()

# Save bottle single data to CSV
bottle_prof.to_csv('data/BATS_Bottle_Pigments_profData.csv')

# Add POC model results to bottle POC df
bottle_poc['surf_model']  = MLD_pop_FIT_POC
bottle_poc['sub_model']   = DCM_pop_FIT_POC
bottle_poc['background']  = BACKGROUND_POC
bottle_poc['total_model'] = TOTAL_POC_MODEL
bottle_poc['POC_used']    = POC_model_input
bottle_poc['model_diff']  = POC_diff

# Calculate Phyto Carbon and Ratio directly in DF
bottle_poc['Phyto_POC'] = bottle_poc['surf_model'] + bottle_poc['sub_model']
bottle_poc['Phyto_POC_Ratio'] = bottle_poc['Phyto_POC'] / bottle_poc['total_model']

# Save new POC dataframe with Model results to CSV
bottle_poc.to_csv('data/BATS_Bottle_POC_ModelResults.csv')

# Setup new DF for Model POC prof data (single values per profile)
bottle_poc_prof_x = pd.DataFrame()
bottle_poc_prof_x['cruise_ID']  = df_poc_ID.astype('int')
bottle_poc_prof_x['Date']       = df_poc_date
bottle_poc_prof_x['DecYear']    = df_poc_DecYear
bottle_poc_prof_x['yyyy'] = pd.to_datetime(bottle_poc_prof_x['Date']).dt.year 
bottle_poc_prof_x['mm']   = pd.to_datetime(bottle_poc_prof_x['Date']).dt.month

bottle_poc_prof_x['CHL_POC_C1'] = CHL_POC_C1
bottle_poc_prof_x['CHL_POC_C2'] = CHL_POC_C2
bottle_poc_prof_x['MLD_model']  = MLD_model_fit_POC # MLD used to fit model
bottle_poc_prof_x['CHL_POC_C1_d'] = CHL_POC_C1_d # dimensionless
bottle_poc_prof_x['CHL_POC_C2_d'] = CHL_POC_C2_d # dimensionless
bottle_poc_prof_x['POC_bk_d'] = POC_bk_d # dimensionless

print(len(bottle_poc_prof_x))

bottle_poc_prof_x.info()

# Remove prof where model could not be fitted and replace old prof df
bottle_poc_prof = bottle_poc_prof_x.dropna(subset=['CHL_POC_C1'])

# Reset df index replacing old index column
bottle_poc_prof = bottle_poc_prof.reset_index(drop=True)

bottle_poc_prof.info()

# Save new POC dataframe with Model output to CSV
bottle_poc_prof.to_csv('data/BATS_Bottle_POC_ProfData.csv')


#%%

# Test Model difference here

#%%

### FILTER CHLA MODEL RESULTS FOR ONLY FITTED and prep for contour plots

# CSV filename
filename_1 = 'data/BATS_Bottle_Pigments_ModelResults.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_6   = pd.read_csv(filename_1, index_col = 0)

bottle_6.info()

# Remove NaN fit profiles
# Drop POC profiles not fitted - NaN
bottle_6 = bottle_6.dropna(subset=['surf_model'])

# Sort new df by ID and depth
#bottle_6 = bottle_6.sort_values(by=['time','depth'])
bottle_6 = bottle_6.sort_values(by=['Date','depth'])

# Reset bottle df index replacing old index column
bottle_6 = bottle_6.reset_index(drop=True)

bottle_6.info()

# Save new dataframe with Model output to CSV
bottle_6.to_csv('data/BATS_Bottle_Pigments_ModelResults.csv')

### EXTRACT CLEANED DATA & MAKE NEW ID LIST for fitted profiles ###

### Extract required data from new bottle_6 dataset ###
b_time     = bottle_6.loc[:,'time'].to_numpy()
b_date     = bottle_6.loc[:,'Date'].to_numpy()
b_depth    = bottle_6.loc[:,'depth'].to_numpy()
b_Fchla    = bottle_6.loc[:,'pigment16'].to_numpy()
b_chla     = bottle_6.loc[:,'pigment14'].to_numpy()
m_chla = b_chla
b_ID       = bottle_6.loc[:,'Cruise_ID'].to_numpy()
b_year     = bottle_6.loc[:,'yyyy'].to_numpy()
b_month    = bottle_6.loc[:,'mm'].to_numpy()
b_Decimal_year = bottle_6.loc[:,'DecYear'].to_numpy()

# Bottle DateTime data
b_DateTime     = pd.to_datetime(bottle_6['time'].values)

### Cruise_ID list for new df is ID_list_6

# Converts to pandas timeseries array
ID_list_6 = pd.Series(b_ID)
# Removes Duplicates
ID_list_6 = pd.unique(ID_list_6)
print(len(ID_list_6))
# 413 profiles with 6 or more POC measurements

# Extract POC Model Results
MLD_pop_FIT      = bottle_6.loc[:,'surf_model'].to_numpy()
DCM_pop_FIT      = bottle_6.loc[:,'sub_model'].to_numpy()
CHL_model_fit    = bottle_6.loc[:,'total_model'].to_numpy()
CHL_model_input  = bottle_6.loc[:,'CHL_used'].to_numpy()
CHL_diff         = bottle_6.loc[:,'model_diff'].to_numpy()

# Import POC prof Data
# CSV filename
filename_1 = 'data/BATS_Bottle_Pigments_profData.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_prof   = pd.read_csv(filename_1, index_col = 0)

bottle_prof.info()

# Remove NaN fit profiles
# Drop POC profiles not fitted - NaN
bottle_prof = bottle_prof.dropna(subset=['P1_TEMP'])
# Reset bottle df index replacing old index column
bottle_prof = bottle_prof.reset_index(drop=True)

print(len(bottle_prof))

# Save new dataframe with Model output to CSV
bottle_prof.to_csv('data/BATS_Bottle_Pigments_profData.csv')

#%%
######
### CONTOUR PLOTS OF MODEL RESULTS ###
#####

### Interpolate data ###

ID_list_bottle = ID_list_6

#m_chla = b_chla

# Set xy (area) of contour plot
y = 300
x2 = len(ID_list_bottle)

print(len(bottle_6))

#Depth
New_depth = np.array(range(0, y))

CHLA_m_surf   = np.empty([x2,y])#+nan
CHLA_m_subb   = np.empty([x2,y])#+nan
CHLA_m_totl   = np.empty([x2,y])#+nan
CHLA_inter    = np.empty([x2,y])#+nan

Depth_bot     = np.empty([x2,y])#+nan
Time_bot      = (np.empty([x2,y], dtype='datetime64[s]'))

# Chlorophyll-a data interpolation
count = 0    
for i in ID_list_bottle:
    a = m_chla[bottle_6.Cruise_ID == i]
    b = b_depth[bottle_6.Cruise_ID == i]
    valid1 = ~np.isnan(b)
    valid2 = ~np.isnan(a)
    a      = a[valid2]
    c      = b[valid2]
    if len(b) > 1:
        interpfunc       = interpolate.interp1d(c,a, kind='linear',fill_value="extrapolate")
        xxx              = interpfunc(New_depth)
        CHLA_inter[count,:]  = xxx
    count=count+1

count = 0    
for i in ID_list_bottle:
    if len(b_depth[bottle_6.Cruise_ID == i]) > 1:
        interpfunc       = interpolate.interp1d(b_depth[bottle_6.Cruise_ID == i],MLD_pop_FIT[bottle_6.Cruise_ID == i], kind='linear',fill_value="extrapolate")
        xxx              = interpfunc(New_depth)
        CHLA_m_surf[count,:]  = xxx
    count=count+1

count = 0    
for i in ID_list_bottle:
    if len(b_depth[bottle_6.Cruise_ID == i]) > 1:
        interpfunc       = interpolate.interp1d(b_depth[bottle_6.Cruise_ID == i],DCM_pop_FIT[bottle_6.Cruise_ID == i], kind='linear',fill_value="extrapolate")
        xxx              = interpfunc(New_depth)
        CHLA_m_subb[count,:]  = xxx
    count=count+1

count = 0    
for i in ID_list_bottle:
    a = CHL_model_fit[bottle_6.Cruise_ID == i]
    b = b_depth[bottle_6.Cruise_ID == i]
# =============================================================================
#     valid1 = ~np.isnan(b)
#     valid2 = ~np.isnan(a)
#     a      = a[valid2]
#     c      = b[valid2]
# =============================================================================
    if len(b) > 1:
        interpfunc       = interpolate.interp1d(b,a, kind='linear',fill_value="extrapolate")
        xxx              = interpfunc(New_depth)
        CHLA_m_totl[count,:]  = xxx
        AD               = b_time[bottle_6.Cruise_ID == i]
        Time_bot[count,:]  = AD[0]    
        Depth_bot[count,:] = New_depth
    count=count+1

#%%
### Plot
#Figure parameters that can be changed 
TEMP_COL          = mpl.cm.magma  #Temp colour scale (see https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)
PSAL_COL          = mpl.cm.winter #Salinity colour scale (see https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)
DOXY_COL          = mpl.cm.copper #Diss OXY colour scale (see https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)
CHLA_COL = cmocean.cm.algae
POC_COL  = cmocean.cm.tempo
BBP_COL           = mpl.cm.cividis#bbp colour scale (see https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)
DIFF_COL          = cmocean.cm.balance
XSIZE             = 14            #Define the xsize of the figure window
YSIZE             = 16           #Define the ysize of the figure window
Title_font_size   = 19            #Define the font size of the titles
Label_font_size_x = 17            #Define the font size of the x-labels
Label_font_size_y = 17            #Define the font size of the y-labels
tick_length       = 6 
Cbar_title_size   = 17            #Define the font size of the Colourbar title
Cbar_label_size   = 17           #Define the font size of the Colourbar labels
pad_width         = 0.02
Percentiles_upper = 99            #Upper percentiles used to constrain the colour scale
Percentiles_lower = 1  

#Define the figure window including 5 subplots orientated vertically
fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5, sharex=True, figsize=(XSIZE,YSIZE))
fig.subplots_adjust(wspace=0.2,hspace=0.27)
   
#Set date range for x-axis 
#2022-12-16
#x axistest#
xaxi = [date(1989,12,31),date(2022,12,16)]

#SUBPLOT 1: Raw CHL TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = CHLA_inter

PCT_1          = np.nanpercentile(b_chla, Percentiles_lower)
PCT_2          = np.nanpercentile(b_chla, Percentiles_upper)
#PCT_1 = 0.0
#PCT_2 = 0.5
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im1            = ax1.contourf(Time_bot, Depth_bot, IN_DATA, levels,cmap = CHLA_COL, extend = 'max')
##Set axis info and titles
ax1.set_ylim([200,0]) 
#ax1.set_xlim([xaxi[0],xaxi[-1]]) 
ax1.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax1.set_title('(a) Chl-a Data', fontsize = Title_font_size, color='k')
ax1.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax1.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar1 = fig.colorbar(im1, ax=ax1, pad = pad_width)
cbar1.ax.locator_params(nbins=5)
cbar1.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar1.ax.tick_params(labelsize = Cbar_label_size)
cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

#SUBPLOT 2: Model of total chlA
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = CHLA_m_totl
PCT_1          = np.nanpercentile(b_chla, Percentiles_lower)
PCT_2          = np.nanpercentile(b_chla, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im2            = ax2.contourf(Time_bot, Depth_bot, IN_DATA, levels,cmap = CHLA_COL, extend = 'max')
##Set axis info and titles
ax2.set_ylim([200,0]) 
ax2.set_xlim([xaxi[0],xaxi[-1]]) 
ax2.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax2.set_title('(b) Model Total Chl-a', fontsize = Title_font_size, color='k')
ax2.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax2.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar2 = fig.colorbar(im2, ax=ax2, pad = pad_width)
cbar2.ax.locator_params(nbins=5)
cbar2.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar2.ax.tick_params(labelsize = Cbar_label_size)
cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

#SUBPLOT 3: Model of total chlA
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = CHLA_m_totl - CHLA_inter
# PCT_1          = np.nanpercentile(CHLA_2, Percentiles_lower)
# PCT_2          = np.nanpercentile(CHLA_2, Percentiles_upper)
PCT_1 = -0.35
PCT_2 = 0.35
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im3            = ax3.contourf(Time_bot, Depth_bot, IN_DATA, levels,cmap = mpl.cm.bwr, extend = 'both')
##Set axis info and titles
ax3.set_ylim([200,0]) 
#ax3.set_xlim([xaxi[0],xaxi[-1]]) 
ax3.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax3.set_title('(c) Model - Data', fontsize = Title_font_size, color='k')
ax3.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax3.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar3 = fig.colorbar(im3, ax=ax3, pad = pad_width)
cbar3.ax.locator_params(nbins=5)
cbar3.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar3.ax.tick_params(labelsize = Cbar_label_size)
cbar3.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

#SUBPLOT 4: CHL A surface TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = CHLA_m_surf
PCT_1          = np.nanpercentile(b_chla, Percentiles_lower)
PCT_2          = np.nanpercentile(b_chla, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im4            = ax4.contourf(Time_bot, Depth_bot, IN_DATA, levels,cmap = CHLA_COL,alpha =1, extend = 'max')
##Set axis info and titles
ax4.set_ylim([200,0]) 
#ax4.set_xlim([xaxi[0],xaxi[-1]]) 
ax4.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax4.set_title('(d) Surface Chl-a', fontsize = Title_font_size, color='r')
ax4.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax4.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar4 = fig.colorbar(im4, ax=ax4, pad = pad_width)
cbar4.ax.locator_params(nbins=5)
cbar4.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar4.ax.tick_params(labelsize = Cbar_label_size)
cbar4.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

#SUBPLOT 5: CHL A subsurface TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = CHLA_m_subb
PCT_1          = np.nanpercentile(b_chla, Percentiles_lower)
PCT_2          = np.nanpercentile(b_chla, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im5          = ax5.contourf(Time_bot, Depth_bot, IN_DATA, levels,cmap = CHLA_COL,alpha =1, extend = 'max')
##Set axis info and titles
ax5.set_ylim([200,0]) 
#ax5.set_xlim([xaxi[0],xaxi[-1]]) 
ax5.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax5.set_title('(e) Subsurface Chl-a', fontsize = Title_font_size, color='b')
ax5.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax5.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar5 = fig.colorbar(im5, ax=ax5, pad = pad_width)
cbar5.ax.locator_params(nbins=5)
cbar5.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar5.ax.tick_params(labelsize = Cbar_label_size)
cbar5.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax5.set_xlabel('Year', fontsize=Cbar_title_size, color='k')
# Save Plot
fig.savefig('plots/BATS_Contour_Model_Chla_sns.png', format='png', dpi=300, bbox_inches="tight")
plt.show()

#%%

### FILTER POC MODEL RESULTS FOR ONLY FITTED profiles and prep for contour plots

# CSV filename
filename_1 = 'data/BATS_Bottle_POC_ModelResults.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_poc   = pd.read_csv(filename_1, index_col = 0)

bottle_poc.info()

# Drop POC profiles not fitted - NaN
bottle_poc = bottle_poc.dropna(subset=['surf_model'])

# Sort new df by ID and depth
bottle_poc = bottle_poc.sort_values(by=['Date','depth'])

# Reset bottle df index replacing old index column
bottle_poc = bottle_poc.reset_index(drop=True)

# Save new dataframe with Model output to CSV
bottle_poc.to_csv('data/BATS_Bottle_POC_ModelResults.csv')

### EXTRACT CLEANED DATA & MAKE NEW POC ID LIST ###

### Extract required data from new bottle_6 dataset ###
b2_time     = bottle_poc.loc[:,'time'].to_numpy()
b2_time_2   = pd.to_datetime(bottle_poc['time'])
b2_date     = bottle_poc.loc[:,'Date'].to_numpy()
b2_depth    = bottle_poc.loc[:,'depth'].to_numpy()
b2_poc      = bottle_poc.loc[:,'POC'].to_numpy()
b2_ID       = bottle_poc.loc[:,'cruise_ID'].to_numpy()
b2_year     = bottle_poc.loc[:,'yyyy'].to_numpy()
b2_month    = bottle_poc.loc[:,'mm'].to_numpy()

#Convert array object to Datetimeindex type
b2_DateTime = pd.DatetimeIndex(b2_time_2, dtype='datetime64[ns]', name='date_time', freq=None)

### Cruise_ID list

# Converts to pandas timeseries array
ID_list_poc = pd.Series(b2_ID)
# Removes Duplicates
ID_list_poc = pd.unique(ID_list_poc)
print(len(ID_list_poc))
# 416 profiles with 6 or more POC measurements

# Extract POC Model Results
MLD_pop_FIT_POC = bottle_poc.loc[:,'surf_model'].to_numpy()
DCM_pop_FIT_POC = bottle_poc.loc[:,'sub_model'].to_numpy()
TOTAL_POC_MODEL = bottle_poc.loc[:,'total_model'].to_numpy()
BACKGROUND_POC  = bottle_poc.loc[:,'background'].to_numpy()
POC_diff        = bottle_poc.loc[:,'model_diff'].to_numpy()

# Import POC prof Data
# CSV filename
filename_1 = 'data/BATS_Bottle_POC_ProfData.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_poc_prof   = pd.read_csv(filename_1, index_col = 0)

bottle_poc_prof.info()

# Remove NaN fit profiles
# Drop POC profiles not fitted - NaN
bottle_poc_prof = bottle_poc_prof.dropna(subset=['CHL_POC_C1'])
# Reset bottle df index replacing old index column
bottle_poc_prof = bottle_poc_prof.reset_index(drop=True)

print(len(bottle_poc_prof))

#%%
######
### CONTOUR PLOTS OF POC MODEL RESULTS ###
#####

print(len(bottle_poc))
print(len(b2_depth))

### Interpolate data ###

ID_list_bottle = ID_list_poc

#m_chla = b_chla

# Set xy (area) of contour plot
y = 300
x2 = len(ID_list_bottle)

#Depth
New_depth = np.array(range(0, y))

POC_m_surf   = np.empty([x2,y])#+nan
POC_m_subb   = np.empty([x2,y])#+nan
POC_m_totl   = np.empty([x2,y])#+nan
POC          = np.empty([x2,y])#+nan
POC_m_bk     = np.empty([x2,y])#+nan

Depth_bot     = np.empty([x2,y])#+nan
Time_bot      = (np.empty([x2,y], dtype='datetime64[s]'))

# Raw POC interpolation
count = 0    
for i in ID_list_bottle:
    a = b2_poc[bottle_poc.cruise_ID == i]
    b = b2_depth[bottle_poc.cruise_ID == i]
    valid1 = ~np.isnan(b)
    valid2 = ~np.isnan(a)
    a      = a[valid2]
    c      = b[valid2]
    if len(b) > 1:
        interpfunc    = interpolate.interp1d(c,a, kind='linear',fill_value="extrapolate")
        xxx           = interpfunc(New_depth)
        POC[count,:]  = xxx
        AD                 = b2_time[bottle_poc.cruise_ID == i]
        Time_bot[count,:]  = AD[0]    
        Depth_bot[count,:] = New_depth
    count=count+1

# POC surface interpolation
count = 0    
for i in ID_list_bottle:
    a = MLD_pop_FIT_POC[bottle_poc.cruise_ID == i]
    b = b2_depth[bottle_poc.cruise_ID == i]
    if len(b) > 1:
        interpfunc       = interpolate.interp1d(b,a, kind='linear',fill_value="extrapolate")
        xxx              = interpfunc(New_depth)
        POC_m_surf[count,:]  = xxx
    count=count+1

# POC subsurface interpolation
count = 0    
for i in ID_list_bottle:
    a = DCM_pop_FIT_POC[bottle_poc.cruise_ID == i]
    b = b2_depth[bottle_poc.cruise_ID == i]
    if len(b) > 1:
        interpfunc       = interpolate.interp1d(b,a, kind='linear',fill_value="extrapolate")
        xxx              = interpfunc(New_depth)
        POC_m_subb[count,:]  = xxx
    count=count+1

# POC total model interpolation
count = 0    
for i in ID_list_bottle:
    a = TOTAL_POC_MODEL[bottle_poc.cruise_ID == i]
    b = b2_depth[bottle_poc.cruise_ID == i]
    if len(b) > 1:
        interpfunc       = interpolate.interp1d(b,a, kind='linear',fill_value="extrapolate")
        xxx              = interpfunc(New_depth)
        POC_m_totl[count,:]  = xxx
    count=count+1
    
# POC background interpolation
count = 0    
for i in ID_list_bottle:
    a = BACKGROUND_POC[bottle_poc.cruise_ID == i]
    b = b2_depth[bottle_poc.cruise_ID == i]
    if len(b) > 1:
        interpfunc       = interpolate.interp1d(b,a, kind='linear',fill_value="extrapolate")
        xxx              = interpfunc(New_depth)
        POC_m_bk[count,:]  = xxx
    count=count+1

#%%
### Plot
#Figure parameters that can be changed 
TEMP_COL          = mpl.cm.magma  #Temp colour scale (see https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)
PSAL_COL          = mpl.cm.winter #Salinity colour scale (see https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)
DOXY_COL          = mpl.cm.copper #Diss OXY colour scale (see https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)
CHLA_COL = cmocean.cm.algae
POC_COL = cmocean.cm.tempo
BBP_COL           = mpl.cm.cividis#bbp colour scale (see https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)
DIFF_COL          = mpl.cm.bwr
XSIZE             = 14            #Define the xsize of the figure window
YSIZE             = 18           #Define the ysize of the figure window
Title_font_size   = 19            #Define the font size of the titles
Label_font_size_x = 17            #Define the font size of the x-labels
Label_font_size_y = 17            #Define the font size of the y-labels
tick_length       = 6 
Cbar_title_size   = 17            #Define the font size of the Colourbar title
Cbar_label_size   = 17           #Define the font size of the Colourbar labels
pad_width         = 0.02
Percentiles_upper = 99            #Upper percentiles used to constrain the colour scale
Percentiles_lower = 1
  
#Define the figure window including 5 subplots orientated vertically
fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6, sharex=True, figsize=(XSIZE,YSIZE))
fig.subplots_adjust(wspace=0.2,hspace=0.27)

#Set date range for x-axis     
#2022-12-16
#x axistest#
xaxi = [date(1989,12,31),date(2022,12,16)]

#SUBPLOT 1: POC time series TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POC
PCT_1          = np.nanpercentile(b2_poc, Percentiles_lower)
PCT_2          = np.nanpercentile(b2_poc, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im1            = ax1.contourf(Time_bot, Depth_bot, IN_DATA, levels,cmap = POC_COL, alpha =1,extend = 'max')
##Set axis info and titles
ax1.set_ylim([200,0]) 
#ax1.set_xlim([xaxi[0],xaxi[-1]]) 
ax1.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax1.set_title('(a) POC Data', fontsize = Title_font_size, color='k')
ax1.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax1.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar1 = fig.colorbar(im1, ax=ax1, pad = pad_width)
cbar1.ax.locator_params(nbins=5)
cbar1.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar1.ax.tick_params(labelsize = Cbar_label_size)
cbar1.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 2: POC total model time series
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POC_m_totl
PCT_1          = np.nanpercentile(b2_poc, Percentiles_lower)
PCT_2          = np.nanpercentile(b2_poc, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im2            = ax2.contourf(Time_bot, Depth_bot, IN_DATA, levels,cmap = POC_COL,alpha =1,extend = 'max')
##Set axis info and titles
ax2.set_ylim([200,0]) 
ax2.set_xlim([xaxi[0],xaxi[-1]]) 
ax2.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax2.set_title('(b) Model Total POC', fontsize = Title_font_size, color='k')
ax2.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax2.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar2 = fig.colorbar(im2, ax=ax2, pad = pad_width)
cbar2.ax.locator_params(nbins=5)
cbar2.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar2.ax.tick_params(labelsize = Cbar_label_size)
cbar2.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 3: POC total model Difference time series
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POC_m_totl - POC
#PCT_1          = np.nanpercentile(POC_diff, Percentiles_lower)
#PCT_2          = np.nanpercentile(POC_diff, Percentiles_upper)
PCT_1 = -40
PCT_2 = 40
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im3            = ax3.contourf(Time_bot, Depth_bot, IN_DATA, levels,cmap = DIFF_COL, extend = 'both')
##Set axis info and titles
ax3.set_ylim([200,0]) 
#ax3.set_xlim([xaxi[0],xaxi[-1]]) 
ax3.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax3.set_title('(c) Model - Data', fontsize = Title_font_size, color='k')
ax3.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax3.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar3 = fig.colorbar(im3, ax=ax3, pad = pad_width)
cbar3.ax.locator_params(nbins=5)
cbar3.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar3.ax.tick_params(labelsize = Cbar_label_size)
cbar3.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 4: POC surface TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POC_m_surf
PCT_1          = np.nanpercentile(b2_poc, Percentiles_lower)
PCT_2          = np.nanpercentile(b2_poc, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im4            = ax4.contourf(Time_bot, Depth_bot, IN_DATA, levels,cmap = POC_COL,alpha =1,extend = 'max')
##Set axis info and titles
ax4.set_ylim([200,0]) 
#ax4.set_xlim([xaxi[0],xaxi[-1]]) 
ax4.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax4.set_title('(d) Surface Phytoplankton Carbon', fontsize = Title_font_size, color='r')
ax4.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax4.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar4 = fig.colorbar(im4, ax=ax4, pad = pad_width)
cbar4.ax.locator_params(nbins=5)
cbar4.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar4.ax.tick_params(labelsize = Cbar_label_size)
cbar4.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 5: POC subsurface TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POC_m_subb
PCT_1          = np.nanpercentile(b2_poc, Percentiles_lower)
PCT_2          = np.nanpercentile(b2_poc, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im5          = ax5.contourf(Time_bot, Depth_bot, IN_DATA, levels,cmap = POC_COL,alpha =1,extend = 'max')
##Set axis info and titles
ax5.set_ylim([200,0]) 
#ax5.set_xlim([xaxi[0],xaxi[-1]]) 
ax5.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax5.set_title('(e) Subsurface Phytoplankton Carbon', fontsize = Title_font_size, color='b')
ax5.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax5.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar5 = fig.colorbar(im5, ax=ax5, pad = pad_width)
cbar5.ax.locator_params(nbins=5)
cbar5.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar5.ax.tick_params(labelsize = Cbar_label_size)
cbar5.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 6: POC background TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POC_m_bk
PCT_1          = np.nanpercentile(b2_poc, Percentiles_lower)
PCT_2          = np.nanpercentile(b2_poc, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im6          = ax6.contourf(Time_bot, Depth_bot, IN_DATA, levels,cmap = POC_COL)
##Set axis info and titles
ax6.set_ylim([200,0]) 
#ax5.set_xlim([xaxi[0],xaxi[-1]]) 
ax6.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax6.set_title('(f) Model POC Background', fontsize = Title_font_size, color='m')
ax6.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax6.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar6 = fig.colorbar(im6, ax=ax6, pad = pad_width)
cbar6.ax.locator_params(nbins=5)
cbar6.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar6.ax.tick_params(labelsize = Cbar_label_size)
cbar6.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
ax6.set_xlabel('Year', fontsize=Title_font_size, color='k')

# Save Plot
fig.savefig('plots/BATS_Contour_Model_POC_sns.png', format='png', dpi=350, bbox_inches="tight")
plt.show()