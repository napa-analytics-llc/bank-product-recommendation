#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

#%% 
import pandas as pd
import numpy as np
import operator


from sklearn.model_selection import train_test_split
from datetime import datetime

def loadData (filename):
    
    df = pd.read_csv(filename,delimiter='|')
    df = df.sort_values(by=['ID_Customer', 'Cod_Fecha'], ascending=[1, 1]).reset_index(drop=True)
    return df
  
def tratamientoFecha(df):
    
    dfFecha=df.copy()
    
    #Conversion of Cod_Fecha column to year-month format
    dfFecha["Cod_Fecha"] =  pd.to_datetime(dfFecha["Cod_Fecha"],format='%Y-%m')
    primerMes = pd.datetime(1950, 1, 1, 0, 0, 0)
    
    #Realizamos la resta del numero de dias desde 1950 y creamos la columna
    # Creation of new column "Num_dias": Subtract from the number of days since 1950.
    dfFecha["Num_Dias"] =  dfFecha["Cod_Fecha"]-primerMes 
    dfFecha["Num_Dias"] = (dfFecha["Num_Dias"] / np.timedelta64(1, 'D')).astype(int)
    
    return dfFecha
  
def mapProduct (df, prioris=True):
    dfMap=df.copy()
    mapProduct.mapProd = {}
    mapProduct.mapPrioris = {}
    
    
    if(prioris):
        dfPrioris= (dfMap.groupby("Cod_Prod").count().ID_Customer)/float(len(dfMap))
        
        dfPrioris = dfPrioris.sort_values(ascending = False)
        dfPrioris = dfPrioris.to_frame()
        
        dfPrioris.columns = ['Priori']
        dfPrioris["Labels"] = list(range(len(dfPrioris)))
        
        
        mapProduct.mapPrioris = pd.Series(dfPrioris.Labels.values,index=dfPrioris.index).to_dict()
        
        dfMap = dfMap.replace({"Cod_Prod": mapProduct.mapPrioris})
        
           
    else:
        #we convert the columns to categorical data.
        dfMap['Cod_Prod'] = dfMap['Cod_Prod'].astype('category')
        #Extraemos las columnas con datos categoricos
        cat_columns = dfMap.select_dtypes(['category']).columns
        #Dictionary IdProd : nombreProd
        map_prod = dict(enumerate(dfMap.Cod_Prod.cat.categories))
        
        #Transform to number
        dfMap[cat_columns] = dfMap[cat_columns].apply(lambda x: x.cat.codes)
        
        for idGenerado in map_prod.keys():
            mapProduct.mapProd[int(map_prod[idGenerado])]=int(idGenerado)
    
    return dfMap
  
def mapProdByDict(df,mapTest = []):      
    if mapTest == []:
        mapTest = mapProduct.mapPrioris
        
    df = df.replace({"Cod_Prod":mapTest})
    
    return df
  
def addProdAnt(df,num_ant = 1, diasAnt = False, fechaAnt1 = True):
    
    df2 = df.copy().reset_index(drop=True)
    
    
    for ant in range(num_ant):
        
        if ant == 0:
            dfaux = list(df2.Cod_Prod)
            dfFechaAnt1 = list(df2.Cod_Fecha)
            if diasAnt:
                dffec = list(df2.Num_Dias)
        else:
            dfaux = list(df2["Cod_Prod_Ant"+str(ant)])
            if diasAnt:
                dffec = list(df2["Num_Dias_Ant"+str(ant)])

        anteriores = list(df2.groupby("ID_Customer",as_index=False).count().Cod_Prod)
        anteriores.pop(len(anteriores)-1)
        anteriores = np.cumsum(anteriores)
        
        dfaux.insert(0,-1)
        dfaux.pop(len(dfaux)-1)
        dfaux = np.array(dfaux)
        dfaux[anteriores]=-1
        
        df2["Cod_Prod_Ant"+str(ant+1)] = pd.DataFrame(dfaux)
        
        #we add the column of the previous date.
        if fechaAnt1 and ant==0:
            dfFechaAnt1.insert(0,datetime(1950, 1, 1, 0, 0, 0))
            dfFechaAnt1.pop(len(dfFechaAnt1)-1)
            dfFechaAnt1 = np.array(dfFechaAnt1)
            dfFechaAnt1[anteriores]=datetime(1950, 1, 1, 0, 0, 0)
            df2["Cod_Fecha_Ant"] = pd.DataFrame(dfFechaAnt1)
            
        #we add the column of the previous days.
        if diasAnt:
            dffec.insert(0,0)
            dffec.pop(len(dffec)-1)
            dffec = np.array(dffec)
            dffec[anteriores]=0
            
            df2["Num_Dias_Ant"+str(ant+1)] = pd.DataFrame(dffec)

    return df2
  
def createTest (df):
    dfTest2 = df.copy()
    
    #We make a group by user and we keep the last record of each one.
    dfTest2 = dfTest2.groupby("ID_Customer").last().reset_index()
    dfTest = dfTest2.copy()
    
    #We extract the columns that we have modified to obtain information regarding the previous products.
    listaColumnas = list(dfTest.columns)
    
    numProdAnt  = len(list(filter(lambda x: 'Cod_Prod_Ant' in x, listaColumnas)))
   
    dfTest["Cod_Fecha_Ant"]=dfTest2["Cod_Fecha"]
    
    #We modify the columns previously extracted.
    for i in range(numProdAnt):
        nameColProd= 'Cod_Prod_Ant'+str(i+1)
        nameColDias= 'Num_Dias_Ant'+str(i+1)
        
        nameColProd2= 'Cod_Prod_Ant'+str(i)
        nameColDias2= 'Num_Dias_Ant'+str(i)
        if i == 0:
            
            dfTest[nameColProd]=dfTest2["Cod_Prod"]
            dfTest[nameColDias]=dfTest2["Num_Dias"]
         
        else:
            dfTest[nameColProd]=dfTest2[nameColProd2]
            dfTest[nameColDias]=dfTest2[nameColDias2]
        

    return dfTest
  
def subset(dfData, dfLabels, size=.01, seed=45):
    X_train, X_test, y_train, y_test = train_test_split( dfData, dfLabels, train_size=size, random_state=seed)
    return X_train

def classPrune2(dfData, dfLabels, nc = 60, resto = False):
    df = dfData.copy()
    count = dfLabels.value_counts()
    count = pd.DataFrame([count.index, count, count/count.sum()]).transpose()
    quedo = count[[0,1,2]][0:nc]
    poda = count[[0,1,2]][nc:len(count)]
    
    print ("Number of classes:", nc,"\nPercentage of data:",quedo[2].sum()*100,"%\nMinimum of samples:", min(quedo[1]))

    if not resto:
        ind = []
        for d in np.array(poda[0]):
            ind = np.concatenate([ind,df.index[(dfLabels==d)]],axis=0)
        df.drop(ind,axis=0,inplace=True)
    else:
        ind = []
        for d in np.array(poda[0]):
            ind = np.concatenate([ind,df.index[(dfLabels==d)]],axis=0)
        majClass = (np.array(poda)[0,0])
        for i in ind:
            df.loc[i,"Cod_Prod"] = majClass
    return df
  
def expandirVariable(df, nameCol):
    dfNuevo = df.copy()
    numVal = len(dfNuevo[nameCol].unique())
    print("Expanding "+ nameCol + " to dimension " + str(numVal))
    
    for i in range(numVal):
        nameColNuevo = nameCol + "_" + str(i+1)
        
        dfNuevo[nameColNuevo] = np.where(dfNuevo[nameCol]==i, 1, 0)
        
    dfNuevo = dfNuevo.drop(nameCol, 1)
    return dfNuevo
  
def mapAparicionProd(df,productos = []):

    if productos == []:
        productos = list(mapProduct.mapPrioris.values())
        
    mapAparicion = {}
    
    for idP in productos:
        mapAparicion[idP]=df.loc[df['Cod_Prod'] == idP].sort_values(by = ["Cod_Fecha"])["Cod_Fecha"].iloc[0]
    sortedMap = sorted(mapAparicion.items(), key=operator.itemgetter(1))

    marcaTemporal = 0
    mapaFinal = {}
    
    #creamos el mapa
    for idP,fecha in sortedMap:
         mapaFinal[pd.to_datetime(fecha)] = marcaTemporal 
         marcaTemporal += 1
         
    fechaIDtemp = sorted(mapaFinal.items(), key=operator.itemgetter(1))    
         
    mapAparicionProd.mapAparicion = fechaIDtemp
    return fechaIDtemp
  
def addAparicionProdRow(row):
    tiempoAnt = -1
    
    for idF,tiempo in mapAparicionProd.mapAparicion:
        if idF <= row["Cod_Fecha_Ant"]:
            tiempoAnt = tiempo
        else:
            break;
                
        
    return tiempoAnt
  
def addTiempoProdAnt(dfData):
    df = dfData.copy()
    
    df["Aparicion_Prod"] = df.apply(addAparicionProdRow, axis=1)
    
    return df
  
def mapDiasInicio(df):
    dfUsuarioFecha = df[["ID_Customer","Cod_Fecha"]].groupby(["ID_Customer"]).first().reset_index()
    mapUFecha = dict(zip(dfUsuarioFecha.ID_Customer, dfUsuarioFecha.Cod_Fecha))
    
    if hasattr(mapDiasInicio,'mapUsuarioFecha'):
        mapDiasInicio.mapUsuarioFecha.update(mapUFecha)
    else:
        mapDiasInicio.mapUsuarioFecha = mapUFecha
    return mapUFecha
  
def addDiasInicioRow(row):

    if row["Cod_Prod_Ant1"] == -1:
        val = 0
    else:
        val =  (row["Cod_Fecha_Ant"] - mapDiasInicio.mapUsuarioFecha[row["ID_Customer"]]).days
        
    return val
  
def addDiasInicioAnt(dfData):
    df = dfData.copy()
    
    df["DiasDesde_Inicio"] = df.apply(addDiasInicioRow, axis=1)
    
    return df
  
def numProductosComprados(df, test = False):
    nprods = df.groupby("ID_Customer").Cod_Prod.count().reset_index()
    if test:
        return nprods["Cod_Prod"]
    else:
        return nprods["Cod_Prod"] - 1
def ultimoElementoSerie(df):
    df2 = df.copy()
    return df2.groupby("ID_Customer").last().reset_index()
  
def restaFechas(dfData):
    
    df = dfData.copy()
    listaColumnas = list(df.columns)
    numResta  = len(list(filter(lambda x: 'Num_Dias_Ant' in x, listaColumnas)))
    numResta = numResta-1
    for i in range(numResta):
        nameCol = "Diferencia_Fechas_"+str(i+1)
        columna1 = "Num_Dias_Ant"+str(i+1)
        columna2 = "Num_Dias_Ant"+str(i+2)
        df[nameCol] = df[columna1] - df[columna2]
    
    
    return df
  
def modificaColumna(row):
    fecha1 = datetime(2000, 1, 1, 0, 0, 0) #merger of banks in the area of Andalusia and Madrid.
    fecha2 = datetime(2007, 1, 1, 0, 0, 0) #merger of banks in the area of Castilla and Leon.
    fecha3 = datetime(2011, 1, 1, 0, 0, 0) #merger of banks in the area of Valencian community.
    if row["Cod_Fecha_Ant"] == -1:
        val = -1
    elif row["Cod_Fecha_Ant"] < fecha1:
        val = 0
    elif row["Cod_Fecha_Ant"] > fecha1 and row["Cod_Fecha_Ant"] < fecha2:
        val = 1
    elif row["Cod_Fecha_Ant"] > fecha1 and row["Cod_Fecha_Ant"] > fecha2 and row["Cod_Fecha_Ant"] < fecha3 :
        val = 2
    else:
        val = 3
    return val
  
def acontecimiento(dfData):
    df = dfData.copy()
    
    df["AcontecimientoAnt"] = df.apply(modificaColumna, axis=1)
    
    return df
def mapYearPIB():
    dfPIB =  pd.read_csv("PIB.txt",delimiter='\t')

    mapYearPIB.mapPIB = pd.Series(dfPIB.PIB.values,index=dfPIB.Year).to_dict()
    
    return mapYearPIB.mapPIB
  
def addPIBRow(row):
    
    try:
        if row["Cod_Fecha_Ant"].year == 1950: #if equal to 1950, haven't a product before
            val = -1
        elif row["Cod_Fecha_Ant"].year ==2017:
            val = 24000
        else:
            val = mapYearPIB.mapPIB[row["Cod_Fecha_Ant"].year]
            
    except KeyError: #if it is less than 1961 we do not have GDP data so we put it to 0.
        val = 147
    return val
  
def addPIBAnt(dfData):
    df = dfData.copy()
    
    df["PIB_Ant"] = df.apply(addPIBRow, axis=1)
    
    return df