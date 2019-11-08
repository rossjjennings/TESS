import numpy as np
from find_tp import find_tp
from tess_lightcurves import tess_lightcurves



# Open list of targets to run
#f = np.loadtxt('/Users/Admin/Documents/Research_Lisa/TESS/TIC_HZ_list_DR1_4.txt')
f = np.loadtxt('/Users/Admin/Documents/Research_Lisa/TESS/TICs_408_UV.txt')

for i in range(len(f)):
    
    print('----------------------\n'+'Analyzing TIC'+str(int(f[i]))+'...')
    
    # Check all available sectors
    slist = ["0000","0001","0002","0003","0004","0005","0006"]
    for k in range(len(slist)):
        print('Searching sector '+str(slist[k])+'...')
        a = find_tp(int(f[i]),slist[k])


        if a[0] == 1:
            print('Extracting light curve...')
    
            tess_lightcurves(str(a[1]+str(a[2])),int(f[i]))