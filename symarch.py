# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 21:23:05 2016

@author: ricevind
"""
import os
import datetime

def arch_create(name, geo):
    geometries = {'ane':'Aneurysm', 'bif':'Bifurcation', 'aor':'Aorta'}
    
    if geo not in geometries:
      return 'Wrong geometry folder'
    release = '/home/ricevind/Documents/Praca MGR/Release'
    
    folder = os.path.join(release,geometries[geo], name)
    try:
      os.mkdir(folder)
      os.mkdir(os.path.join(folder, 'Zdjecia'))
    except:
      print('prawdopodobnie już istnieje')
      
    return folder
    
  
def arch_param(folder, **kwargs):
    with open(os.path.join(folder, 'params'), 'a') as file:
      newline = "######################## {} ########################## \n".format(datetime.datetime.now())
      file.write(newline)
      for name in kwargs:
        file.write(name+' = '+str(kwargs[name])+'\n')
