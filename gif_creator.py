# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:38:25 2015

@author: ricevind
"""
import os
filename = 'rura_tau_5525ny90_Re497_poussile_n1'

os.system("mencoder 'mf://*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o " + filename + ".mpg")
    
#for i, j, k in os.walk(r'/home/ricevind/Documents/Praca MGR/Zdjecia'):
#     for k in k:
#         if len(k) == 12:
#             os.rename(k, '0'+k[4:6]+'.png')
#         if len(k) == 13:
#             os.rename(k, k[4:7]+'.png')
#         else:
#             continue