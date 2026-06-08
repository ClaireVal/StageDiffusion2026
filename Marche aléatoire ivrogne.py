#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:51:20 2026

@author: valenchonc
"""

###  MARCHE ALEATOIRE DE L'IVROGNE  ###

import numpy.random as rd

#Initialisation de p, qui sera tiré aléatoirement à chaque pas
p = 0
#Initialisation des pas
left = 0
right = 0
N=10000

#N pas de l'ivrogne
for i in range(N):
    p = rd.random()
    if u<p:
        left+=1
    else:
        right+=1

print("p=",p,"  (left,right)=",(left, right), "  proportion de gauche=", left/(left+right))