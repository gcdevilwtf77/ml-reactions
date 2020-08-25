#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:30:17 2020

@author: ty
"""

from __future__ import print_function

import numpy as np
from scipy.integrate import ode
import cno_rhs_x as cno
import matplotlib.pyplot as plt
import pandas as pd
from time import time

def burn(X0, rho, T, tmax, nsave):

    r = ode(cno.rhs).set_integrator("vode", method="bdf",
                                    with_jacobian=False,
                                    atol=1.e-8, rtol=1.e-8,
                                    nsteps = 1500000, order=5) #, min_step=dt)

    t = 0.0

    r.set_initial_value(X0, t)

    r.set_f_params(rho, T)

    dt = tmax/nsave
    AllDataPoints=[]
    AllDataPoints.append(np.append(r.y,[r.t,rho,T]))

    istep = 1
    while r.successful() and istep <= nsave:
        r.integrate(t+dt*istep)

        if r.successful():
            AllDataPoints.append(np.append(r.y,[r.t,rho,T]))
            istep = istep + 1
        else:
            print("An integration error occurred at time {}".format(r.t))

    return AllDataPoints


if __name__ == "__main__":

    # initialize as mass fractions first
    X0 = np.zeros((cno.nnuc), dtype=np.float64)

    X0[cno.ip] = 0.7
    X0[cno.ihe4] = 0.28
    X0[cno.ic12] = 0.02

    rho = 10000.0
    T = np.linspace(1.e8,5.e8,1000)

    # estimate the H destruction time
    
    StartTime=time()
    
    k=0
    
    for Temperatures in T:
    
        StartLoopTime=time()
        
        Xdot = cno.rhs(0.0, X0, rho, Temperatures)
        
        TimeScaling=10
        tmax = TimeScaling*np.abs(X0[cno.ip]/Xdot[cno.ip])
        print("tmax: {}".format(tmax))
    
        nsteps = 100
    
        AllDataSpatialPoints = burn(X0, rho, Temperatures, tmax, nsteps)
        
        if k==0:
            
            AllDataSpatialPointsDataFrame=pd.DataFrame(AllDataSpatialPoints,columns=
                                                       ['ip','ihe4','ic12','ic13',
                                                        'in13','in14','in15','io14',
                                                        'io15','t','rho','T'])
        else:
            
            AllDataSpatialPointsDataFrame2=pd.DataFrame(AllDataSpatialPoints,columns=
                                                       ['ip','ihe4','ic12','ic13',
                                                        'in13','in14','in15','io14',
                                                        'io15','t','rho','T'])
            AllDataSpatialPointsDataFrame=pd.concat([AllDataSpatialPointsDataFrame,AllDataSpatialPointsDataFrame2],axis=0)
    
#        AllDataSpatialPointsDataFrame.to_csv('IntitialTemperature:'+str(Temperatures)+'.csv',index=False)
        
        if k%10==0:
            print(k,time()-StartLoopTime)
        
        k+=1
        
    print((time()-StartTime)/60**2)
    
    AllDataSpatialPointsDataFrame.to_csv('IntitialTemperaturestMax'+str(TimeScaling)+'.csv',index=False)

#    plt.loglog(t, np.array(X_H), label="H1")
#    plt.loglog(t, np.array(X_He), label="He4")
#    plt.loglog(t, np.array(X_O14), label="O14")
#    plt.loglog(t, np.array(X_O15), label="O15")
#
#    plt.xlabel("time [s]")
#    plt.ylabel("mass fraction, X")
#
#    plt.legend(frameon=False)
#
#    plt.savefig("burn.png")
