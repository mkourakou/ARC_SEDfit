#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 17:27:59 2023

@author: mado  

AGN models used here are produced by Markos Polkas
ARC SEDs are streamlined by Markos Polkas     

"""




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.font_manager import FontProperties

import pickle
from astropy import units as  u
from astropy.cosmology import FlatLambdaCDM

import itertools

import scipy.constants as cst
from scipy.integrate import trapz, simps

import ultranest
import ultranest.stepsampler
from pathlib import Path
font_path = './fonts/eb_garamond_font/EbGaramond12Regular_LjX5.otf'
font = FontProperties(fname=font_path, size=20)
fontBigger = FontProperties(fname=font_path, size=22)
fontBiggest = FontProperties(fname=font_path, size=24)
font_leg = FontProperties(fname=font_path, size=9)
font_leg2 = FontProperties(fname=font_path, size=12)
font_leg3 = FontProperties(fname=font_path, size=14)


import os
if not os.path.exists('./Constrained_Log_fit_L3'): 
    os.makedirs('./Constrained_Log_fit_L3')

outpath = './Constrained_Log_fit_L3/'


#################################
# Load templates
#################################

route_AGNtemps = './SED_FIT/models/'
route_MARKtemps = './MARKOS/templates/'
route_Daletemps = './SED_FIT/'

# ------------------ Stellar population -----------------#
BC03dict = pickle.load(open(route_AGNtemps + 'GALAXY/BC03_SEDs.pickle', 'rb'), encoding='latin1')  
mask_nebabs = np.where(BC03dict['wavelength'].value>=0.12e4)  #this is angstrom

# ------------------ ISM Dust -----------------#

pd_agn0 = pd.read_csv(route_Daletemps+'DUST EMISSION MODELS/Joint AGN, star-forming infrared, submillimeter, radio SED models presented in Dale + (2014)/The (rest-frame) spectra/spectra.0.00AGNt.dat', 
                      sep=" ",header=None,names=["lambda","flux_0.0625","flux_0.1250","flux_0.1875","flux_0.25",
                                                 "flux_0.3125","flux_0.375","flux_0.4375","flux_0.5","flux_0.5625",
                                                "flux_0.625","flux_0.6875","flux_0.75","flux_0.8125","flux_0.875",
                                                "flux_0.9375","flux_1.0","flux_1.0625","flux_1.1250","flux_1.1875",
                                                "flux_1.25","flux_1.3125","flux_1.375","flux_1.4375","flux_1.5",
                                                "flux_1.5625","flux_1.625","flux_1.6875","flux_1.75","flux_1.8125",
                                                "flux_1.875","flux_1.9375","flux_2.0","flux_2.0625","flux_2.1250",
                                                "flux_2.1875","flux_2.25","flux_2.3125","flux_2.375","flux_2.4375","flux_2.5",
                                                "flux_2.5625","flux_2.625","flux_2.6875","flux_2.75","flux_2.8125",
                                                "flux_2.875","flux_2.9375","flux_3.0","flux_3.0625","flux_3.1250",
                                                "flux_3.1875","flux_3.25","flux_3.3125","flux_3.375","flux_3.4375",
                                                "flux_3.5","flux_3.5625","flux_3.625","flux_3.6875","flux_3.75","flux_3.8125",
                                                "flux_3.875","flux_3.9375","flux_4.0"])


dust_grid = np.loadtxt(route_Daletemps+'DUST EMISSION MODELS/Joint AGN, star-forming infrared, submillimeter, radio SED models presented in Dale + (2014)/The (rest-frame) spectra/spectra.0.00AGNt.dat')[:,1:]

# ------------------ TORUS hot dust -----------------#
#CAT3D
TOR_CAT3D = pickle.load(open(route_AGNtemps + 'TORUS/CAT3D_markos.pickle', 'rb'), encoding='latin1') 

# ---------------- BBB ----------------------#
BBBtemps = np.load(route_MARKtemps+'BBB_templates.npy',allow_pickle=True).item()


#################################
# Load data
#################################

route_data = "./data/"

df = pd.read_pickle(route_data+"Sample120_SED_markos.pcl")

#-------    Add Short Names  -----------------------#
Short_names = ['J0006-0623 / PKS 0003-066', 'J0009-3216 / IC 1531', 'J0048+3157 / NGC 262', 'J0057+3021 / NGC 315', 'J0106-2718 / PKS 0104-275', 
               'J0106-4034 / PKS 0104-408', 'J0112-6634 / PKS 0110-668', 'J0118-2141 / PKS 0116-219','J0119+3210 / 4C +31.04 / QSO B0116+319', 'J0125-0005 / PKS 0122-003',
               'J0154-2040 / LEDA 2822840', 'J0158-2459 / PKS 0156-252', 'J0217-0820 / PKS 0214-085', 'J0219+0120 / PKS 0216+011', 'J0238+1636 / PKS 0235+164', 
               'J0239-0234 / PKS 0237-027', 'PKS 0324-228','J0329-2357 / PKS 0327-241', 'J0343-2530 / PKS 0341-256', ' J0504-1014 / PKS 0502-10', 
               'J0510+1800 / PKS 0507+179', 'J0529-7245 / PKS 0530-727', 'J0623-6436 / IERS B0622-645', 'J0720-3407 / PKS 0718-340', 'J0748+2400 / PKS 0745+241', 
               'J0758+3747 / 3C 189 / NGC 2484', 'J0830+2410 / IERS B0827+2421 ', 'J0837+2454 / QSO B0834+2504','J0840+2949 / 4C +29.30 / CGCG 150-014', 'J0909+0121 / 4C 01.24B / PKS 0906+015', 
               'J0914+0245 / PKS 0912+029', 'J0940+2603 / IERS B0937+262', 'J0943-0819 / PKS 0941-08', 'J0946+1017 / IERS B0943+105', 'J0954+1743 / PKS 0952+179', 
               'J1000-3139 / NGC 3100', 'J1008+0029 / PKS 1005+007', 'J1010-0200 / 4C -01.21 / PKS 1008-017', 'J1019-2219 / MRC 1017-220 / LEDA 2826255', 'J1022+3041 / QSO B1019+309', 
               'J1038+0512 / PKS 1036+054', 'J1058-8003 / PKS 1057-797', 'J1107-4449 / PKS 1104-445', 'J1109-3732 / NGC 3557', 'J1136-0330 / 4C -03.44 / MRC 1133-032', 
               'J1140-2629 / PKS 1138-262', 'J1146-2447 / PKS 1143-245', 'J1147-0724 / PKS 1145-071', 'J1217+3007 / PKS 1215+303', 'J1220+0203 / UM 492 / PKS 1217+023', 
               'J1221+2813 / QSO B1219+285', 'J1248-4118 / NGC 4696 / PKS 1245-410', 'J1301-3226 / PKS 1258-321', 'J1304-0346 / PKS 1302-034', 'J1332+0200 / 3C 287.1 / PKS 1330+022',
               'J1336-3357 / IC 4296 / PKS 1333-336', 'J1348+2635 / 4C 26.42 / PKS 1346+268', 'J1351-2912 / PKS 1348-289', 'J1356-3421 / PKS 1353-341', 'J1359+0159 / PKS 1356+022', 
               'J1407-2701 / IC 4374 / PKS 1404-267', 'J1419+0628 / 3C 298 / PKS 1416+067', 'J1427+2348 / PKS 1424+240', 'J1505+0326 / PKS 1502+036', 'J1520+2016 / 3C 318 / PKS 1517+204', 
               'J1521+0420 / CGCG 049-138 / PKS 1518+045', 'J1547+2052 / PKS 1545+210', 'J1602+0157 / 3C 327 / PKS 1559+021', 'J1610-3958 / QSO B1606-398', 'J1723-6500 / NGC 6328 / PKS 1718-649',
               'J1743-0350 / PKS 1741-038', 'J1805+1101 / 3C 368 / PKS 1802+110', 'J1945-5520 / NGC 6812 / PKS 1941-554', 'J2000-1748 / PKS 1958-179', 'J2009-4849 / PKS 2005-489', 
               'J2051-2702 / LEDA 2830631 / MRC 2048-272', 'J2056-4714 / PKS 2052-474', 'J2131-3837 / NGC 7075 / PKS 2128-388', 'J2131-1207 / PKS 2128-123', 'J2134-0153 / 4C -02.81 / PKS 2131-021',  
               'J2141-3729 / PKS 2138-377', 'J2239-5701 / PKS 2236-572', 'J2257-3627 / IC 1459 / PKS 2254-367', 'J2320+0812 / NGC 7626 / PKS 2318+079', 'J2320+0513 / PKS 2318+049',
               'J2325-1207 / PKS 2322-123', 'J2341+0018 / PKS 2338+000', 'J2349+0534 / IERS B2346+052', 'J1140+1743 / NGC 3801 / PKS 1137+180', 'J0039+0319 / NGC 193 / PKS 0036+030',
               'J1410+1733 / NGC 5490 / PKS 1407+177', 'J1323+3133 / NGC 5127 / LEDA 46809', 'J1324+3622 / NGC 5141 / LEDA 46906', 'J0125-0122 / NGC 541 / LEDA 5305', 'J0156+0537 / NGC 741 / PKS 0153+053',
               'J0709+4836 / NGC 2329', 'J2214+1350 / NGC 7236 / PKS 2212+135', 'J1104+3812 / Mrk 421 / PKS 1101+384', 'J2335+2701 / NGC 7720 / PKS 2335+267', 'J0505-2835 / PKS 0503-286',
               'J1449+6316 / IC 1065 / 3C 305', 'J0058+2651 / NGC 326 / PKS 0055+265', 'J1842+7946 / 3C 390.3 / IERS B1845+797', 'J1348+2635 / 4C 26.42 / PKS 1346+268', 'J1407+2827 / Mrk 0668 / QSO B1404+286' ,
               'J1321+4235 / 3C 285 / LEDA 46625', 'J1709+3426 / 4C 34.45 / LEDA 2820370', 'J1552+2005 / 3C 326 / PKS 1550+202', 'J1531+2404 / 3C 321 / PKS 1529+242', 'J0747-1917 / PKS 0745-191',
               'J1158+2621 / 4C 26.35 / PKS 1155+266', 'J0009+1244 / 4C 12.03 / PKS 0007+124', 'J011651-2052 / PKS 0114-21', 'J0234+3134 / 3C 68.2 / LEDA 2820169', 'J0408-2418 / LEDA 2823818 / MRC 0406-244',
               'J2106-2405 / LEDA 2830749 / MRC 2104-242', 'J0242-2132 / PKS 0240-217' , 'J1305-1033 / PKS 1302-103', 'J0403+2600 / PKS 0400+258' , 'J1347+1217 / 4C 12.50 / PKS 1345+125']

df['names_mado'] = np.array(Short_names).copy()

#====================================================================================#
#====================================================================================#

#========================   SELECT GALAXY by INDEX    ===============================#

#====================================================================================#
#====================================================================================#


jjj = 6   
iglx = df.index[jjj] 
redshift = df['z_new'][iglx]
cosmo=FlatLambdaCDM(H0=70, Om0=0.3)
D = cosmo.luminosity_distance(redshift)/u.Mpc*u.Mpc.to(u.cm) #in cm
age_universe = cosmo.age(redshift)
age_UpLim = age_universe.value*1000       # from Gyrs ----->  Myrs
print("Age of the universe : ",age_UpLim, 'Is blazar :', df['blazar_flag'][iglx], 
      'ARC index : ', iglx, 'number :', jjj) 


reduced_data = np.array([df.loc[iglx]['reduced_data'][i]['data'] for i in (df.loc[iglx]['reduced_data'].keys())])
#reduced_data  (18 by 3 matrix) triplets of datavalues for 18 total keys/ frequency bands

data = reduced_data
data = data[data[:, 0].argsort()]


#################################
# Functions
#################################



def make_bins(midpoints):
    
    """ A general function for turning an array of bin midpoints into an
    array of bin left hand side positions and bin widths. Splits the
    distance between bin midpoints equally in linear space.
    Parameters
    ----------
    midpoints : numpy.ndarray
        Array of bin midpoint positions
    make_rhs : bool
        Whether to add the position of the right hand side of the final
        bin to bin_lhs, defaults to false.
    """
    
    bin_widths = np.zeros_like(midpoints)
    bin_lhs = np.zeros(midpoints.shape[0]+1)
    bin_lhs[0] = midpoints[0] - (midpoints[1]-midpoints[0])/2
    bin_widths[-1] = (midpoints[-1] - midpoints[-2])
    bin_lhs[-1] = midpoints[-1] + (midpoints[-1]-midpoints[-2])/2
    bin_lhs[1:-1] = (midpoints[1:] + midpoints[:-1])/2
    bin_widths[:-1] = bin_lhs[1:-1]-bin_lhs[:-2]

    return bin_lhs, bin_widths 
    




def GALAXYred_Calzetti(wavel_mum, Lumi_ergs, GAebv):

    """
    This function computes the effect of reddening in the galaxy template (Calzetti law)
    ## input:
    -frequencies in Hz
    - Fluxes in Fnu
    - the reddening value E(B-V)_gal
    ## output:
    """
    
    RV = 4.05
    k = np.zeros(len(wavel_mum))

    w0 = tuple([wavel_mum <= 0.12])
    w1 = tuple([wavel_mum < 0.63])
    w2 = tuple([wavel_mum >= 0.63])
    
    x1 = np.argmin(np.abs(wavel_mum - 0.12))
    x2 = np.argmin(np.abs(wavel_mum - 0.125))

    k[w2] = 2.659 * (-1.857 + 1.040 /wavel_mum[w2])+RV
    k[w1] = 2.659 * (-2.156 + (1.509/wavel_mum[w1]) - (0.198/wavel_mum[w1]**2) + (0.011/wavel_mum[w1]**3))+RV
    if (wavel_mum[x1] - wavel_mum[x2]) != 0:  # avoid division by zero
        k[w0] = k[x1] + ((wavel_mum[w0] - 0.12) * (k[x1] - k[x2]) / (wavel_mum[x1] - wavel_mum[x2])) +RV
    else:
        k[w0] = 0
    
    gal_k= k
    Lumi_red_ergs = Lumi_ergs *  (10**(-0.4 * gal_k * GAebv))
    return wavel_mum, Lumi_red_ergs

#################################

x_data_mum = 1e6*cst.c/np.flip(data[:,0])  # wavelength in microns
y_data_ergs = np.flip(data[:,1])       # Luminosity in erg/s
y_data_errs = np.flip(data[:,2])

fit_window_mum = [0.12, 500.0]
fit_mask = np.where(np.logical_and(x_data_mum>fit_window_mum[0], x_data_mum<fit_window_mum[1]))

#-------------------   Mask data------------------# 
x_data_mum = x_data_mum[fit_mask]
y_data_ergs = y_data_ergs[fit_mask]
y_data_errs = y_data_errs[fit_mask]


#################################
# BINS, LIMITS, and template values
#################################

#============================   VALUES  ===================================#
#----------- BBB (accretion disk) -----------------#
BBBkeys = list(BBBtemps.keys())
BBBred_values = np.array(np.arange(0.,100.,10)/100)
#------------  STELLAR ------------#
Metalli_values = np.copy(BC03dict['metallicity-values'])  # in Z_solar
Age_values = np.copy(BC03dict['age-values'].value/1e6)  # in Myrs
Tau_values = np.copy(BC03dict['tau-values'].value)      # in Gyrs
#------------  DUST  ------------#
Alpha_sf_values = np.array([0.0625, 0.1250, 0.1875, 0.25,0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625,
                   0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.0625, 1.1250, 1.1875, 1.25, 
                   1.3125, 1.375, 1.4375, 1.5, 1.5625, 1.625, 1.6875, 1.75, 1.8125, 1.875, 
                   1.9375, 2.0, 2.0625, 2.1250, 2.1875, 2.25, 2.3125, 2.375, 2.4375, 2.5, 
                   2.5625, 2.625, 2.6875, 2.75, 2.8125, 2.875, 2.9375, 3.0, 3.0625, 3.1250, 
                   3.1875, 3.25, 3.3125, 3.375, 3.4375, 3.5, 3.5625, 3.625, 3.6875, 3.75, 
                   3.8125, 3.875, 3.9375, 4.0])

#------------  TORUS  ------------#
TORkeys = list(TOR_CAT3D.keys())[1:].copy()
TOR_values = np.array(np.arange(0.,12.,1))                       
    
#============================  BINS  ===================================#
#----------- BBB (accretion disk) -----------------#

BBBred_bins =  make_bins(BBBred_values)[0]
BBBred_bins[-1] =1.1
#------------  STELLAR ------------#
Metalli_bins = make_bins(Metalli_values)[0]   #in Z_solar
Metalli_bins[0] = 0.0
Metalli_bins[-1] = 5.0
Age_bins = make_bins(Age_values)[0]       #in Myrs
Age_bins[0] = 1.0
Age_bins[-1] = 1.3e4
Tau_bins = make_bins(Tau_values)[0]      #in Gyrs
Tau_bins[0] = 1e-2
Tau_bins[-1] = 14
#------------  DUST  ------------#
Alpha_sf_bins = make_bins(Alpha_sf_values)[0]   
Alpha_sf_bins[0] = 0.0
Alpha_sf_bins[-1] = 4.5
#------------  TORUS  ------------#
TORi_bins = make_bins(TOR_values)[0] 

#============================   LIMITS  ===================================#
#----------- BBB (accretion disk) -----------------#
BBBred_lims = [0.0, 0.9]
LogBBBnorm_lims = [-10, 5]
#------------  STELLAR ------------#
StellarMass_limits = [1e7,1e15]     # [1e4, 1e14]
LogStellarMass_lims = [7,15]
Metal_lims =[0.0, 5.0]
LogAge_lims = [0,np.log10(age_UpLim)]
LogAge_o_lims = [3.14,np.log10(age_UpLim)]
LogAge_y_lims = [1,3.13]
LogTau_lims = [-2, 1.13]
EBV_lims = [0,1]  #unit cube
#------------  DUST  -------------#  
DustMass_lims = [1e0,1e20]  
LogDustMass_lims = [0, 20]
Alpha_sf_lims = [Alpha_sf_bins[0] , Alpha_sf_bins[-1]]
#------------  TORUS  ------------#
TorMass_lims = [1e-10,1e5] 
TOR_lims = [0,11]
LogTorMass_lims = [-10, 8]


#################################


def Galaxy_Model(x_data_mum, metalo, ageo, tauo, logstmasso, metaly, agey, tauy, logstmassy, ebvo, ebvy,
                bbbred, logbbbnorm, tor, logTormass, alpha_sf, logDmass):
    #Step 1 choose stellar
    metalo_i,  tauo_i, metaly_i,  tauy_i  = 0,0,0,0;
    for i in range(len(Metalli_values)):                             # Metal values Old
        if Metalli_bins[i+1]> metalo>=Metalli_bins[i]:  metalo_i = i;
    for i in range(len(Tau_values)):                                 # Tau values old
        if Tau_bins[i+1]> tauo>=Tau_bins[i]:  tauo_i = i;
    for i in range(len(Metalli_values)):                            # Metal values young
        if Metalli_bins[i+1]> metaly>=Metalli_bins[i]:  metaly_i = i;
    for i in range(len(Tau_values)):                                 # Tau values young
        if Tau_bins[i+1]> tauy>=Tau_bins[i]:  tauy_i = i;
            
    if (ageo>=Age_bins[14] ) and (ageo<Age_bins[15] ): ageo_i = 14;   #Age_values Old
    else : 
        if (ageo>=Age_bins[15] ) and (ageo<Age_bins[16] ): ageo_i = 15;   
        else : 
            if (ageo>=Age_bins[16] ) and (ageo<Age_bins[17] ): ageo_i = 16;   
            else : 
                if (ageo>=Age_bins[17] ) and (ageo<Age_bins[18] ): ageo_i = 17;   
                else :  
                    if (ageo>=Age_bins[18] ) and (ageo<Age_bins[19] ): ageo_i = 18;   
                    else :  ageo_i = 19; 
                        
    if (agey>=Age_bins[0] ) and (agey<Age_bins[1] ): agey_i = 0;     #Age_values Young
    else : 
        if (agey>=Age_bins[1] ) and (agey<Age_bins[2] ): agey_i = 1;   
        else : 
            if (agey>=Age_bins[2] ) and (agey<Age_bins[3] ): agey_i = 2;   
            else : 
                if (agey>=Age_bins[3] ) and (agey<Age_bins[4] ): agey_i = 3;   
                else : 
                    if (agey>=Age_bins[4] ) and (agey<Age_bins[5] ): agey_i = 4;   
                    else : 
                        if (agey>=Age_bins[5] ) and (agey<Age_bins[6] ): agey_i = 5;   
                        else : 
                            if (agey>=Age_bins[6] ) and (agey<Age_bins[7] ): agey_i = 6;   
                            else : 
                                if (agey>=Age_bins[7] ) and (agey<Age_bins[8] ): agey_i = 7;   
                                else : 
                                    if (agey>=Age_bins[8] ) and (agey<Age_bins[9] ): agey_i = 8;   
                                    else :  
                                        if (agey>=Age_bins[9] ) and (agey<Age_bins[10] ): agey_i = 9;     
                                        else : 
                                            if (agey>=Age_bins[10] ) and (agey<Age_bins[11] ): agey_i = 10;   
                                            else : 
                                                if (agey>=Age_bins[11] ) and (agey<Age_bins[12] ): agey_i = 11;   
                                                else : 
                                                    if (agey>=Age_bins[12] ) and (agey<Age_bins[13] ): agey_i = 12;   
                                                    else : agey_i = 13;
    
    
    
    
    
    
    x_sttemplate_mum = BC03dict['wavelength'].value[mask_nebabs]/1e4  # Angstrom ---> microns
    x_sttemplate_mum[np.where(x_sttemplate_mum==0.0)]=1e-10
    sttemp_o_lumi = BC03dict['SED'][metalo_i,ageo_i,tauo_i,0, 0, :].value[mask_nebabs] * BC03dict['wavelength'].value[mask_nebabs] * 3.826e33  #lambda L_lambda (erg/s)
    sttemp_y_lumi = BC03dict['SED'][metaly_i,agey_i,tauy_i,0, 0, :].value[mask_nebabs] * BC03dict['wavelength'].value[mask_nebabs] * 3.826e33  #lambda L_lambda (erg/s)
    
    RV = 4.05
    k = np.zeros(len(x_sttemplate_mum))
    w0 = tuple([x_sttemplate_mum <= 0.12])
    w1 = tuple([x_sttemplate_mum < 0.63])
    w2 = tuple([x_sttemplate_mum >= 0.63])
    x1 = np.argmin(np.abs(x_sttemplate_mum - 0.12))
    x2 = np.argmin(np.abs(x_sttemplate_mum - 0.125))
    k[w2] = 2.659 * (-1.857 + 1.040 /x_sttemplate_mum[w2])+RV
    k[w1] = 2.659 * (-2.156 + (1.509/x_sttemplate_mum[w1]) - (0.198/x_sttemplate_mum[w1]**2) + (0.011/x_sttemplate_mum[w1]**3))+RV
    if (x_sttemplate_mum[x1] - x_sttemplate_mum[x2]) != 0:  # avoid division by zero
        k[w0] = k[x1] + ((x_sttemplate_mum[w0] - 0.12) * (k[x1] - k[x2]) / (x_sttemplate_mum[x1] - x_sttemplate_mum[x2])) +RV
    else:
        k[w0] = 0
    gal_k= k
    
    sttemp_o_lumi_red = sttemp_o_lumi *  (10**(-0.4 * gal_k * ebvo))
    sttemp_y_lumi_red = sttemp_y_lumi *  (10**(-0.4 * gal_k * ebvy))
    
    logsttemp_o_lumi = logstmasso + np.log10(sttemp_o_lumi)
    logsttemp_y_lumi = logstmassy + np.log10(sttemp_y_lumi)
    logsttemp_o_lumi_red = logstmasso + np.log10(sttemp_o_lumi_red)
    logsttemp_y_lumi_red = logstmassy + np.log10(sttemp_y_lumi_red)
    
    Int_sto = simps((10**logsttemp_o_lumi)/x_sttemplate_mum, x_sttemplate_mum)
    Int_sto_red = simps((10**logsttemp_o_lumi_red)/x_sttemplate_mum, x_sttemplate_mum)
    Abs_sto = Int_sto - Int_sto_red
    #LogAbs_sto = np.log10(Abs_sto)
    Int_sty = simps((10**logsttemp_y_lumi)/x_sttemplate_mum, x_sttemplate_mum)
    Int_sty_red = simps((10**logsttemp_y_lumi_red)/x_sttemplate_mum, x_sttemplate_mum)
    Abs_sty = Int_sty - Int_sty_red
    #LogAbs_sty = np.log10(Abs_sty)
    
    y_stomodel_ergs = np.interp(x_data_mum, x_sttemplate_mum, logsttemp_o_lumi_red ,left=0, right=0)
    y_stymodel_ergs = np.interp(x_data_mum, x_sttemplate_mum, logsttemp_y_lumi_red ,left=0, right=0)
    
    #===========================   Step 2 choose ISM DUST template   ===============================================#
    alpha_sf_i  = 0;
    for i in range(len(Alpha_sf_values)):
        if Alpha_sf_bins[i+1]>alpha_sf>=Alpha_sf_bins[i]: alpha_sf_i = i;                                                                                                                                                                                                                                                     
    x_dusttemplate_mum = pd_agn0['lambda'].to_numpy()  #in microns                                                                                                                                                                                                                                                          
    dusttemp_lumi_ergs = (dust_grid[:, alpha_sf_i]+7+57.07854852935109+logDmass) #in erg/s PER SOLAR MASS (watt-->erg/s, H atom mass ---> solar mass)
    y_dustmodel_ergs = np.interp(x_data_mum, x_dusttemplate_mum, dusttemp_lumi_ergs ,left=0, right=0)
    
    Int_dust = simps((10**dusttemp_lumi_ergs)/x_dusttemplate_mum, x_dusttemplate_mum)
    
    #===========================   Step 3 choose TORUS template   ===============================================#
    tor_i  = 0;
    for i in range(len(TOR_values)):
        if TORi_bins[i+1]>tor>=TORi_bins[i]: tor_i = i;
    x_tortemplate_mum = 1e6 * cst.c/(TOR_CAT3D[TORkeys[0]]['data'][:,0].copy())  # Hz ---> microns/ no flip needed
    TOR_CAT3D[TORkeys[tor_i]]['data'][:,1][np.where(TOR_CAT3D[TORkeys[tor_i]]['data'][:,1] ==0.0)] = 1.0
    tortemp_lumi = logTormass + np.log10(TOR_CAT3D[TORkeys[int(tor_i)]]['data'][:,1])
    y_tormodel_ergs = np.interp(x_data_mum, x_tortemplate_mum, tortemp_lumi ,left=0, right=0)
    
    Int_tor = simps((10**tortemp_lumi)/x_tortemplate_mum, x_tortemplate_mum)
    #LogLumi_tor = np.log10(Int_tor)
    
    
    #===========================   Step 4 choose BBB template   ===============================================#
    bbb_i = 0;
    for i in range(len(BBBred_values)):
        if BBBred_bins[i+1]>bbbred>=BBBred_bins[i]: bbb_i = i;
    
    x_bbbtemplate_mum = 1e6*cst.c/np.flip(BBBtemps[BBBkeys[0]][:,0])        # reddening  = 0 
    x_bbbtemplate_mum[np.where(x_bbbtemplate_mum==0.0)]=1e-10
    y_bbbtemplate_ergs = logbbbnorm + np.log10(np.flip(BBBtemps[BBBkeys[bbb_i]][:,1]))
    y_bbbtemplate0_ergs = logbbbnorm + np.log10(np.flip(BBBtemps[BBBkeys[0]][:,1]))
    mask_bbb = np.where(x_bbbtemplate_mum>=0.12)
    x_bbbtemplate_mum = x_bbbtemplate_mum[mask_bbb]
    y_bbbtemplate_ergs = y_bbbtemplate_ergs[mask_bbb]
    y_bbbtemplate0_ergs = y_bbbtemplate0_ergs[mask_bbb]
    y_bbbmodel_ergs =  np.interp(x_data_mum, x_bbbtemplate_mum, y_bbbtemplate_ergs ,left=0, right=0)
    
    Int_BBB0 = simps((10**y_bbbtemplate0_ergs)/x_bbbtemplate_mum, x_bbbtemplate_mum)
    Int_BBB = simps((10**y_bbbtemplate_ergs)/x_bbbtemplate_mum, x_bbbtemplate_mum)
    Abs_bbb = Int_BBB0 - Int_BBB
    #LogAbs_bbb = np.log10(Abs_bbb)
    
    
    #================== CONSTRAINTS  =================================================#
    if Int_dust<(0.8*(Abs_sto+Abs_sty)): return -1e20*np.ones(len(x_data_mum));

    if Int_tor<(0.8*Abs_bbb): return -1e20*np.ones(len(x_data_mum));
    
    return np.log10(10**y_stomodel_ergs + 10**y_stymodel_ergs + 10**y_dustmodel_ergs + 10**y_tormodel_ergs + 10**y_bbbmodel_ergs)



parameters = ['Zo', 'AgeO', 'TauO', 'logMass_StO', 'Zy', 'AgeY', 'TauY', 'logMass_StY', 'EBVo', 'EBVy', 
              'BBBred', 'logBBBnorm', 'Tor', 'logTornorm', 'alpha_d', 'LogMass_D']


def prior_transform(cube):
    params = cube.copy()
    
    metalo_lo, metalo_hi = Metalli_bins[0], Metalli_bins[-1]
    metaly_lo, metaly_hi = Metalli_bins[0], Metalli_bins[-1]
    ageo_lo, ageo_hi = LogAge_o_lims[0], LogAge_o_lims[1]
    agey_lo, agey_hi = LogAge_y_lims[0], LogAge_y_lims[1]
    tauo_lo, tauo_hi = LogTau_lims[0], LogTau_lims[1]
    tauy_lo, tauy_hi = LogTau_lims[0], LogTau_lims[1]    
    mass_stelo_lo, mass_stelo_hi = LogStellarMass_lims[0], LogStellarMass_lims[1]
    mass_stely_lo, mass_stely_hi = LogStellarMass_lims[0], LogStellarMass_lims[1]
    
    bbbred_lo, bbbred_hi = BBBred_lims[0], BBBred_lims[1]
    bbbnorm_lo, bbbnorm_hi = LogBBBnorm_lims[0], LogBBBnorm_lims[1]
    tor_lo, tor_hi = TOR_lims[0], TOR_lims[1]
    mass_tor_lo, mass_tor_hi = LogTorMass_lims[0], LogTorMass_lims[1]
    alpha_lo, alpha_hi = Alpha_sf_lims[0], Alpha_sf_lims[1]
    mass_d_lo, mass_d_hi = LogDustMass_lims[0], LogDustMass_lims[1]
    
    params[0] = cube[0] * (metalo_hi - metalo_lo) + metalo_lo     # Metallicity old
    params[1] = 10**(cube[1] * (ageo_hi - ageo_lo) + ageo_lo)   # Age
    params[2] = 10**(cube[2] * (tauo_hi - tauo_lo) + tauo_lo)   # Tau
    params[3] = cube[3] * (mass_stelo_hi - mass_stelo_lo) + mass_stelo_lo    # Log Old Stellar Mass 
    params[4] = cube[4] * (metaly_hi - metaly_lo) + metaly_lo     # Metallicity old
    params[5] = 10**(cube[5] * (agey_hi - agey_lo) + agey_lo)   # Age
    params[6] = 10**(cube[6] * (tauy_hi - tauy_lo) + tauy_lo)   # Tau
    params[7] = cube[7] * (mass_stely_hi - mass_stely_lo) + mass_stely_lo    # Log Old Stellar Mass
    params[8] = cube[8]    #EBVo from 0 to 1
    params[9] = cube[9]    #EBVy from 0 to 1
    
    params[10] = cube[10] * (bbbred_hi - bbbred_lo) + bbbred_lo     # BBB reddening
    params[11] = (cube[11] * (bbbnorm_hi - bbbnorm_lo) + bbbnorm_lo)   # BBB normalisation
    
    params[12] = cube[12] * (tor_hi - tor_lo) + tor_lo     # Torus
    params[13] = (cube[13] * (mass_tor_hi - mass_tor_lo) + mass_tor_lo)   # Torus normalisation
    params[14] = cube[14] * (alpha_hi - alpha_lo) + alpha_lo     # Metallicity 
    params[15] = (cube[15] * (mass_d_hi - mass_d_lo) + mass_d_lo)   # Log ISM Dust
    
    
    
    return params



def log_likelihood(params):
    # unpack the current parameters:
    MetallO, AgeO, TauO, LogMassStO, MetallY, AgeY, TauY, LogMassStY, EBVO, EBVY, Bred, Bnorm, Tor, Tornorm, a, LogMassD  = params

    # compute for each x point, where it should lie in y
    y_model = Galaxy_Model(x_data_mum, metalo=MetallO, ageo=AgeO, tauo=TauO, logstmasso=LogMassStO,
                           metaly=MetallY, agey=AgeY, tauy=TauY, logstmassy = LogMassStY, ebvo=EBVO, 
                           ebvy=EBVY,bbbred=Bred, logbbbnorm=Bnorm, tor=Tor, logTormass=Tornorm, 
                           alpha_sf=a, logDmass=LogMassD)
    # compute likelihood
    loglike = -0.5 * (((y_model - y_data_ergs) / y_data_errs)**2).sum()
    if LogMassStY > (LogMassStO -np.log10(4)): return -1e20;
    return loglike


#################################


sampler000 = ultranest.ReactiveNestedSampler(parameters,log_likelihood, prior_transform)

nsteps = 2 * len(parameters)
# create step sampler:
sampler000.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=nsteps,
                                                            generate_direction=ultranest.stepsampler.generate_mixture_random_direction )


result000 = sampler000.run(min_num_live_points=800)



print(np.shape(sampler000.results['weighted_samples']['upoints']))
print(np.shape(sampler000.results['weighted_samples']['points']))
print(np.shape(sampler000.results['weighted_samples']['logl']))
print(np.shape(sampler000.results['samples']))
print('\n Z_o/Z_solar:', np.median(sampler000.results['samples'][:,0]),
      '\n Age_o/Myrs:', np.median(sampler000.results['samples'][:,1]), 
      '\n Tau_o/Gyrs:', np.median(sampler000.results['samples'][:,2]),
      '\n logMass_sto/M_solar:', np.median(sampler000.results['samples'][:,3]),
      '\n Z_y/Z_solar:', np.median(sampler000.results['samples'][:,4]),
      '\n Age_y/Myrs:', np.median(sampler000.results['samples'][:,5]), 
      '\n Tau_y/Gyrs:', np.median(sampler000.results['samples'][:,6]),
      '\n logMass_sty/M_sol:', np.median(sampler000.results['samples'][:,7]),
      '\n EBVo:', np.median(sampler000.results['samples'][:,8]),
      '\n EBVy:', np.median(sampler000.results['samples'][:,9]), 
      '\n BBBreddening:', np.median(sampler000.results['samples'][:,10]),
      '\n BBBnormalisation:', np.median(sampler000.results['samples'][:,11]),
      '\n Torus:', np.median(sampler000.results['samples'][:,12]),
      '\n Torusnormalisation:', np.median(sampler000.results['samples'][:,13]), 
      '\n alpha_dust:', np.median(sampler000.results['samples'][:,14]),
      '\n logMdust/M_solar:', np.median(sampler000.results['samples'][:,15]),) 
      
      


#################################
# Saving Samples
#################################


stdt = {'Parametre': ['Z_o/Z_solar', 'Age_o/Myrs', 'Tau_o/Gyrs', 'logMass_sto/M_solar', 'Z_y/Z_solar', 
                      'Age_y/Myrs', 'Tau_y/Gyrs', 'logMass_sty/M_solar', 'EBV_o', 'EBV_y', 'BBBreddening', 
                     'BBB_normalisation', 'Torus', 'Torus_normalisation', 'alpha_dust', 'logMass_dust/M_solar'],
        'Median': [np.median(sampler000.results['samples'][:,0]),np.median(sampler000.results['samples'][:,1]),
                   np.median(sampler000.results['samples'][:,2]),np.median(sampler000.results['samples'][:,3]),
                   np.median(sampler000.results['samples'][:,4]),np.median(sampler000.results['samples'][:,5]),
                   np.median(sampler000.results['samples'][:,6]),np.median(sampler000.results['samples'][:,7]),
                   np.median(sampler000.results['samples'][:,8]),np.median(sampler000.results['samples'][:,9]),
                   np.median(sampler000.results['samples'][:,10]),np.median(sampler000.results['samples'][:,11]),
                   np.median(sampler000.results['samples'][:,12]),np.median(sampler000.results['samples'][:,13]),
                   np.median(sampler000.results['samples'][:,14]),np.median(sampler000.results['samples'][:,15])
                  ], 
       'Q25': [np.quantile(sampler000.results['samples'][:,0], 0.25),np.quantile(sampler000.results['samples'][:,1], 0.25),
                   np.quantile(sampler000.results['samples'][:,2], 0.25),np.quantile(sampler000.results['samples'][:,3], 0.25),
                   np.quantile(sampler000.results['samples'][:,4], 0.25),np.quantile(sampler000.results['samples'][:,5], 0.25),
                   np.quantile(sampler000.results['samples'][:,6], 0.25),np.quantile(sampler000.results['samples'][:,7], 0.25),
                   np.quantile(sampler000.results['samples'][:,8], 0.25),np.quantile(sampler000.results['samples'][:,9], 0.25),
                   np.quantile(sampler000.results['samples'][:,10], 0.25),np.quantile(sampler000.results['samples'][:,11], 0.25),
                   np.quantile(sampler000.results['samples'][:,12], 0.25),np.quantile(sampler000.results['samples'][:,13], 0.25),
                   np.quantile(sampler000.results['samples'][:,14], 0.25),np.quantile(sampler000.results['samples'][:,15], 0.25)
                  ],
       'Q50': [np.quantile(sampler000.results['samples'][:,0], 0.50),np.quantile(sampler000.results['samples'][:,1], 0.50),
                   np.quantile(sampler000.results['samples'][:,2], 0.50),np.quantile(sampler000.results['samples'][:,3], 0.50),
                   np.quantile(sampler000.results['samples'][:,4], 0.50),np.quantile(sampler000.results['samples'][:,5], 0.50),
                   np.quantile(sampler000.results['samples'][:,6], 0.50),np.quantile(sampler000.results['samples'][:,7], 0.50),
                   np.quantile(sampler000.results['samples'][:,8], 0.50),np.quantile(sampler000.results['samples'][:,9], 0.50),
                   np.quantile(sampler000.results['samples'][:,10], 0.50),np.quantile(sampler000.results['samples'][:,11], 0.50),
                   np.quantile(sampler000.results['samples'][:,12], 0.50),np.quantile(sampler000.results['samples'][:,13], 0.50),
                   np.quantile(sampler000.results['samples'][:,14], 0.50),np.quantile(sampler000.results['samples'][:,15], 0.50)
                  ],
       'Q68': [np.quantile(sampler000.results['samples'][:,0], 0.68),np.quantile(sampler000.results['samples'][:,1], 0.68),
                   np.quantile(sampler000.results['samples'][:,2], 0.68),np.quantile(sampler000.results['samples'][:,3], 0.68),
                   np.quantile(sampler000.results['samples'][:,4], 0.68),np.quantile(sampler000.results['samples'][:,5], 0.68),
                   np.quantile(sampler000.results['samples'][:,6], 0.68),np.quantile(sampler000.results['samples'][:,7], 0.68),
                   np.quantile(sampler000.results['samples'][:,8], 0.68),np.quantile(sampler000.results['samples'][:,9], 0.68),
                   np.quantile(sampler000.results['samples'][:,10], 0.68),np.quantile(sampler000.results['samples'][:,11], 0.68),
                   np.quantile(sampler000.results['samples'][:,12], 0.68),np.quantile(sampler000.results['samples'][:,13], 0.68),
                   np.quantile(sampler000.results['samples'][:,14], 0.68),np.quantile(sampler000.results['samples'][:,15], 0.68)
                  ],
       'Q75': [np.quantile(sampler000.results['samples'][:,0], 0.75),np.quantile(sampler000.results['samples'][:,1], 0.75),
                   np.quantile(sampler000.results['samples'][:,2], 0.75),np.quantile(sampler000.results['samples'][:,3], 0.75),
                   np.quantile(sampler000.results['samples'][:,4], 0.75),np.quantile(sampler000.results['samples'][:,5], 0.75),
                   np.quantile(sampler000.results['samples'][:,6], 0.75),np.quantile(sampler000.results['samples'][:,7], 0.75),
                   np.quantile(sampler000.results['samples'][:,8], 0.75),np.quantile(sampler000.results['samples'][:,9], 0.75),
                   np.quantile(sampler000.results['samples'][:,10], 0.75),np.quantile(sampler000.results['samples'][:,11], 0.75),
                   np.quantile(sampler000.results['samples'][:,12], 0.75),np.quantile(sampler000.results['samples'][:,13], 0.75),
                   np.quantile(sampler000.results['samples'][:,14], 0.75),np.quantile(sampler000.results['samples'][:,15], 0.75)
                  ],
       'Q95': [np.quantile(sampler000.results['samples'][:,0], 0.95),np.quantile(sampler000.results['samples'][:,1], 0.95),
                   np.quantile(sampler000.results['samples'][:,2], 0.95),np.quantile(sampler000.results['samples'][:,3], 0.95),
                   np.quantile(sampler000.results['samples'][:,4], 0.95),np.quantile(sampler000.results['samples'][:,5], 0.95),
                   np.quantile(sampler000.results['samples'][:,6], 0.95),np.quantile(sampler000.results['samples'][:,7], 0.95),
                   np.quantile(sampler000.results['samples'][:,8], 0.95),np.quantile(sampler000.results['samples'][:,9], 0.95),
                   np.quantile(sampler000.results['samples'][:,10], 0.95),np.quantile(sampler000.results['samples'][:,11], 0.95),
                   np.quantile(sampler000.results['samples'][:,12], 0.25),np.quantile(sampler000.results['samples'][:,13], 0.95),
                   np.quantile(sampler000.results['samples'][:,14], 0.95),np.quantile(sampler000.results['samples'][:,15], 0.95)
                  ],
       'Q99': [np.quantile(sampler000.results['samples'][:,0], 0.99),np.quantile(sampler000.results['samples'][:,1], 0.99),
                   np.quantile(sampler000.results['samples'][:,2], 0.99),np.quantile(sampler000.results['samples'][:,3], 0.99),
                   np.quantile(sampler000.results['samples'][:,4], 0.99),np.quantile(sampler000.results['samples'][:,5], 0.99),
                   np.quantile(sampler000.results['samples'][:,6], 0.99),np.quantile(sampler000.results['samples'][:,7], 0.99),
                   np.quantile(sampler000.results['samples'][:,8], 0.99),np.quantile(sampler000.results['samples'][:,9], 0.99),
                   np.quantile(sampler000.results['samples'][:,10], 0.99),np.quantile(sampler000.results['samples'][:,11], 0.99),
                   np.quantile(sampler000.results['samples'][:,12], 0.99),np.quantile(sampler000.results['samples'][:,13], 0.99),
                   np.quantile(sampler000.results['samples'][:,14], 0.99),np.quantile(sampler000.results['samples'][:,15], 0.99)
                  ],
       'One_sigma': [np.quantile(sampler000.results['samples'][:,0], 0.6827),np.quantile(sampler000.results['samples'][:,1], 0.6827),
                   np.quantile(sampler000.results['samples'][:,2], 0.6827),np.quantile(sampler000.results['samples'][:,3], 0.6827),
                   np.quantile(sampler000.results['samples'][:,4], 0.6827),np.quantile(sampler000.results['samples'][:,5], 0.6827),
                   np.quantile(sampler000.results['samples'][:,6], 0.6827),np.quantile(sampler000.results['samples'][:,7], 0.6827),
                   np.quantile(sampler000.results['samples'][:,8], 0.6827),np.quantile(sampler000.results['samples'][:,9], 0.6827),
                   np.quantile(sampler000.results['samples'][:,10], 0.6827),np.quantile(sampler000.results['samples'][:,11], 0.6827),
                   np.quantile(sampler000.results['samples'][:,12], 0.6827),np.quantile(sampler000.results['samples'][:,13], 0.6827),
                   np.quantile(sampler000.results['samples'][:,14], 0.6827),np.quantile(sampler000.results['samples'][:,15], 0.6827)
                  ],
       'Two_sigma': [np.quantile(sampler000.results['samples'][:,0], 0.9545),np.quantile(sampler000.results['samples'][:,1], 0.9545),
                   np.quantile(sampler000.results['samples'][:,2], 0.9545),np.quantile(sampler000.results['samples'][:,3], 0.9545),
                   np.quantile(sampler000.results['samples'][:,4], 0.9545),np.quantile(sampler000.results['samples'][:,5], 0.9545),
                   np.quantile(sampler000.results['samples'][:,6], 0.9545),np.quantile(sampler000.results['samples'][:,7], 0.9545),
                   np.quantile(sampler000.results['samples'][:,8], 0.9545),np.quantile(sampler000.results['samples'][:,9], 0.9545),
                   np.quantile(sampler000.results['samples'][:,10], 0.9545),np.quantile(sampler000.results['samples'][:,11], 0.9545),
                   np.quantile(sampler000.results['samples'][:,12], 0.9545),np.quantile(sampler000.results['samples'][:,13], 0.9545),
                   np.quantile(sampler000.results['samples'][:,14], 0.9545),np.quantile(sampler000.results['samples'][:,15], 0.9545)
                  ],
       'Three_sigma': [np.quantile(sampler000.results['samples'][:,0], 0.9973),np.quantile(sampler000.results['samples'][:,1], 0.9973),
                   np.quantile(sampler000.results['samples'][:,2], 0.9973),np.quantile(sampler000.results['samples'][:,3], 0.9973),
                   np.quantile(sampler000.results['samples'][:,4], 0.9973),np.quantile(sampler000.results['samples'][:,5], 0.9973),
                   np.quantile(sampler000.results['samples'][:,6], 0.9973),np.quantile(sampler000.results['samples'][:,7], 0.9973),
                   np.quantile(sampler000.results['samples'][:,8], 0.9973),np.quantile(sampler000.results['samples'][:,9], 0.9973),
                   np.quantile(sampler000.results['samples'][:,10], 0.9973),np.quantile(sampler000.results['samples'][:,11], 0.9973),
                   np.quantile(sampler000.results['samples'][:,12], 0.9973),np.quantile(sampler000.results['samples'][:,13], 0.9973),
                   np.quantile(sampler000.results['samples'][:,14], 0.9973),np.quantile(sampler000.results['samples'][:,15], 0.9973)
                  ],
        'MaximumLikelihoodPoint': sampler000.results['maximum_likelihood']['point'],
        'PosteriorMean': sampler000.results['posterior']['mean'],
        'PosteriorStdev': sampler000.results['posterior']['stdev'], 
        'PosteriorMedian': sampler000.results['posterior']['median'], 
        'PosteriorErrlo': sampler000.results['posterior']['errlo'], 
        'PosteriorErrup': sampler000.results['posterior']['errup'], 
        'PosteriorInformation_gain_bits': sampler000.results['posterior']['information_gain_bits']}


df_stat = pd.DataFrame(stdt)
df_stat.to_csv(outpath+str(jjj)+'_Prelim_Stat_ARC'+str(iglx)+'.csv', encoding='utf-8', index=False)


['Z_o/Z_solar', 'Age_o/Myrs', 'Tau_o/Gyrs', 'logMass_sto/M_solar', 'Z_y/Z_solar', 
                      'Age_y/Myrs', 'Tau_y/Gyrs', 'logMass_sty/M_solar', 'EBV_o', 'EBV_y', 'BBBreddening', 
                     'BBB_normalisation', 'Torus', 'Torus_normalisation', 'alpha_dust', 'logMass_dust/M_solar']


Samp = {'Z_o/Z_solar': list(sampler000.results['samples'][:,0]),
        'Age_o/Myrs':  list(sampler000.results['samples'][:,1]),
        'Tau_o/Gyrs': list(sampler000.results['samples'][:,2]),
        'logMass_sto/M_solar': list(sampler000.results['samples'][:,3]),
        'Z_y/Z_solar': list(sampler000.results['samples'][:,4]),
        'Age_y/Myrs':  list(sampler000.results['samples'][:,5]),
        'Tau_y/Gyrs': list(sampler000.results['samples'][:,6]),
        'logMass_sty/M_solar': list(sampler000.results['samples'][:,7]),
        'EBV_o': list(sampler000.results['samples'][:,8]),
        'EBV_y':  list(sampler000.results['samples'][:,9]),
        'BBBreddening': list(sampler000.results['samples'][:,10]),
        'BBB_normalisation': list(sampler000.results['samples'][:,11]),
        'Torus': list(sampler000.results['samples'][:,12]),
        'Torus_normalisation': list(sampler000.results['samples'][:,13]),
        'alpha_dust': list(sampler000.results['samples'][:,14]),
        'logMass_dust/M_solar': list(sampler000.results['samples'][:,15]),
        'WeightedSamplesPoints_Z_o/Z_solar': list(sampler000.results['weighted_samples']['points'][:,0]),
        'WeightedSamplesPoints_Age_o/Myrs':  list(sampler000.results['weighted_samples']['points'][:,1]),
        'WeightedSamplesPoints_Tau_o/Gyrs': list(sampler000.results['weighted_samples']['points'][:,2]),
        'WeightedSamplesPoints_logMass_sto/M_solar': list(sampler000.results['weighted_samples']['points'][:,3]),
        'WeightedSamplesPoints_Z_y/Z_solar': list(sampler000.results['weighted_samples']['points'][:,4]),
        'WeightedSamplesPoints_Age_y/Myrs':  list(sampler000.results['weighted_samples']['points'][:,5]),
        'WeightedSamplesPoints_Tau_y/Gyrs': list(sampler000.results['weighted_samples']['points'][:,6]),
        'WeightedSamplesPoints_logMass_sty/M_solar': list(sampler000.results['weighted_samples']['points'][:,7]),
        'WeightedSamplesPoints_EBV_o': list(sampler000.results['weighted_samples']['points'][:,8]),
        'WeightedSamplesPoints_EBV_y':  list(sampler000.results['weighted_samples']['points'][:,9]),
        'WeightedSamplesPoints_BBBreddening': list(sampler000.results['weighted_samples']['points'][:,10]),
        'WeightedSamplesPoints_BBB_normalisation': list(sampler000.results['weighted_samples']['points'][:,11]),
        'WeightedSamplesPoints_Torus': list(sampler000.results['weighted_samples']['points'][:,12]),
        'WeightedSamplesPoints_Torus_normalisation': list(sampler000.results['weighted_samples']['points'][:,13]),
        'WeightedSamplesPoints_alpha_dust': list(sampler000.results['weighted_samples']['points'][:,14]),
        'WeightedSamplesPoints_logMass_dust/M_solar': list(sampler000.results['weighted_samples']['points'][:,15]),
        'WeightedSamplesLogL':list(sampler000.results['weighted_samples']['logl'])
          }



df_samples = pd.DataFrame(Samp)
df_samples.to_csv(outpath+str(jjj)+'_Prelim_Samples_ARC'+str(iglx)+'.csv', encoding='utf-8', index=False)




















