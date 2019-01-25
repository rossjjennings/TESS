# Code to determine tp file location from TIC ID
# Full documentation at: https://archive.stsci.edu/tess/all_products.html

import requests

# File location: tid/s{sctr}/{tid1}/{tid2}/{tid3}/{tid4}/
# File name: tess{date-time}-s{sctr}-{tid}-{scid}-{cr}_tp.fits

# Notes on file location:
# {sctr} = A zero-padded, four-digit integer indicating the sector in which 
#   the data were collected, starting with Sector 1
#{tid1} = A zero-padded, four-digit integer consisting of digits 1-4 of the 
#   full, zero-padded TIC ID.
#{tid2} = A zero-padded, four-digit integer consisting of digits 5-8 of the 
#   full, zero-padded TIC ID.
#{tid3} = A zero-padded, four-digit integer consisting of digits 9-12 of the 
#   full, zero-padded TIC ID.
#{tid4} = A zero-padded, four-digit integer consisting of digits 13-16 of the
#    full, zero-padded TIC ID.

# Notes on file name:
# {date-time} = The timestamp associated with this file, in the yyyydddhhmmss 
#   format.
# {sctr} = A zero-padded, four-digit integer indicating the sector in which 
#   the data were collected, starting with Sector 1
# {tid} = A zero-padded, 16-digit target identifier that refers to an object in 
#   the TESS Input Catalog.
# {scid} = A zero-padded, four-digit identifier of the spacecraft configuration 
#   map used to process this data.
#{cr} = A string character that denotes the cosmic ray mitigation procedure. 
#   Possible values are:
#       'x': No mitigation performed at the SPOC.
#       's': Mitigation performed on the spacecraft.
#       'a': A SPOC mitigation algorithm was used.
#       'b': Both a SPOC and onboard spacecraft algorithm was used.

# Returns:
# m: 0, if no match, 1 if there is a match
# fl: filepath (str)
# fn: filename (str)

# Inputs (put in as strings):
# TIC: TESS ID number
# sec: data sector (doesn't need to be zero-padded)
def find_tp(TIC,sec):
    
    fl = 0
    fn = 0
    
    # Make sure that the TIC is zero-padded to be 16 digits
    l = len(str(TIC))
    if l < 16:
        # Figure out how many zeroes need to be added to the front of the TIC
        n = 16 - l
        for i in range(n):
            TIC = '0'+str(TIC)
            
    # Make sure that the sector number is zero-padded to be 4 digits
    ls = len(str(sec))
    if ls < 4:
        # Figure out how many zeros need to be added to the front of the TIC
        ns = 4 - ls
        for i in range(ns):
            sec = '0'+str(sec)
        
            
    # Construct the string of the file location
    fl = 'tid/s'+str(sec)+'/'+str(TIC[0:4])+'/'+str(TIC[4:8])+'/'+\
    str(TIC[8:12])+'/'+str(TIC[12:16])+'/'
    
    # Add to the base link
    fl = 'https://archive.stsci.edu/missions/tess/'+str(fl)
    
    
    # Search for correct directory
    r = requests.get(fl)
    rc = r.content
    rf = rc.split()
    
    # Search directory for a target pixel file (_tp.fits)
    for i in range(len(rf)):
        l = str(rf[i])
        
        # Look for the target pixel file
        if l.find('tp.fits') != -1:
            
            # Find first quotation mark and fits (image name is between)
            st = l.find('"')
            end = l.find('fits')
            
            # Extract file name
            fn = l[st+1:end+4]
    
    # Determine whether an image was found      
    if fn != 0:
        m = 1
        print("Image found at: "+str(fl)+str(fn))
        
    if fn == 0:
        m = 0
        print("No image found.")
        
            
            
    return m,fl,fn
            
            
            
        
            
            


                
                
              
