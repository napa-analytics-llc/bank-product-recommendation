#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

#%% 
import pandas as pd
import numpy as np
import operator


from sklearn.model_selection import train_test_split
from datetime import datetime


##
## loadData
## Version: 0.0.1
##
## Description: A file is passed and the result is returned
## in dataframe format sorted by buyer ID and date.
##
## Input:
## - filename: name of the file.
##
## Output:
## - table: description.
##
def loadData (filename):
    
    df = pd.read_csv(filename,delimiter='|')
    df = df.sort_values(by=['ID_Customer', 'Cod_Fecha'], ascending=[1, 1]).reset_index(drop=True)
    return df

##
## getDfMap
## Version: 0.0.1
##
## Description: this function receives the train dictionary
## and the final dataframe and map the product code in the
## output format
##
## Input:
## - df: dataframe with ID_Customer and predicted product code
## - dictTrain: dictionary that we have obtained from training
##
## Output:
## - df: with the mapping done
##
def getDfMap(df,dictTrain=[]):      
    if dictTrain == []:
        dictTrain = mapProduct.mapPrioris
        
    dictFinal={}
    for idGenerado in dictTrain.keys():
        dictFinal[int(dictTrain[idGenerado])]= str(idGenerado).zfill(4)
    
    df = df.replace({"Cod_Prod":dictFinal})
    
    return df

##
## tratamientoFecha
## Version: 0.0.1
##
## Description: A dataframe is passed and a new column with the number is created
## of days that have passed since 1950-01-1 until that user has made the
## purchase of the product
##
##
## Input:
## - df: dataframe to which we are going to create the column number of days.
##
## Output:
## - dfFecha: we return the df with the new column num_dias.
##
def tratamientoFecha(df):
    
    dfFecha=df.copy()
    
    #Conversion of Cod_Fecha column to year-month format
    dfFecha["Cod_Fecha"] =  pd.to_datetime(dfFecha["Cod_Fecha"],format='%Y-%m')
    primerMes = pd.datetime(1950, 1, 1, 0, 0, 0)
    
    # Creation of new column "Num_dias": Subtract from the number of days since 1950.
    dfFecha["Num_Dias"] =  dfFecha["Cod_Fecha"]-primerMes 
    dfFecha["Num_Dias"] = (dfFecha["Num_Dias"] / np.timedelta64(1, 'D')).astype(int)
    
    return dfFecha

##
## mapProduct
## Version: 0.0.7
##
## Description: a train dataframe is passed and the product mapping is obtained
## ordered by priori in case of passing the variable prioris to True.
##
## Input:
## - df: dataframe from which we will obtain the product ids
##
## Output:
## - dfMap: dataframe with the mapping applied.
## 
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

##
## mapProdByDict
## Version: 0.0.1
##
## Description: Maps the product code of a dataframe with the dictionary
## passed by argument.
##
## Input:
## - df: dataframe that contains the column cod_prod to map it.
## - mapTest: dictionary with the mapping of the product code
##
## Output:
## - df: dataframe with the mapping applied.
##  
def mapProdByDict(df,mapTest = []):      
    if mapTest == []:
        mapTest = mapProduct.mapPrioris
        
    df = df.replace({"Cod_Prod":mapTest})
    
    return df

##
## addProdAnt
## Version: 0.0.1
##
## Description: function that extracts products as new columns
## previous purchased by the user, the days that have passed since it was
## have bought the X previous products and the date of the previous product.
##
## Input:
## - df: dataframe from which we are going to extract the information mentioned in the
## description of the function.
## - num_ant: number of previous products to be taken into account
## - diasAnt: if we pass this variable we will also take into account the days that
## have passed since the previous X products were purchased
## - fechaAnt1: if we add this variable to true we add the previous date of the
## previous product contracted by the customer
##
## Output:
## - df2: normalized dataframe.
##  
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

  
##
## createTest
## Version: 0.0.1
##
## Description: this function creates the test dataframe with one record per user
##
## Input:
## - df: dataframe from which we will test.
##
## Output:
## - dfTest: dataframe with one record per user.
##
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

##
## subset
## Version: 0.0.1
##
## Description: extracts a subsampling from the original dataset.
##
## Input:
## - dfData: dataframe with the data
## - dfLabels: dataframe with the class
## - size: percentage that we want to extract from the data
## - seed: seed for subsampling
##
## Output:
## - X_train: dataframe the reduction applied.
##  
def subset(dfData, dfLabels, size=.01, seed=45):
    X_train, X_test, y_train, y_test = train_test_split( dfData, dfLabels, train_size=size, random_state=seed)
    return X_train

##
## classPrune2
## Version: 0.0.1
##
## Description: Eliminate classes given a minimum number of classes
##
## Input:
## - dfData: dataframe with the data
## - dfLabels: dataframe with the class
## - nc: number of classes
## - resto: if set to True, add the rest of the class to a variable and put the label
## of the majority class to that set
##
## Output:
## - df: dataframe with the pruning done.
##
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

##
## expandVariable
## Version: 0.0.1
##
## Description: Expands the variable passed by argument
##
## Input:
## - df: dataframe in recommendation format
## - nameCol: column to expand
##
def expandirVariable(df, nameCol):
    dfNuevo = df.copy()
    numVal = len(dfNuevo[nameCol].unique())
    print("Expanding "+ nameCol + " to dimension " + str(numVal))
    
    for i in range(numVal):
        nameColNuevo = nameCol + "_" + str(i+1)
        
        dfNuevo[nameColNuevo] = np.where(dfNuevo[nameCol]==i, 1, 0)
        
    dfNuevo = dfNuevo.drop(nameCol, 1)
    return dfNuevo

##
## mapAparicionProd
## Version: 0.0.1
##
## Description: create a dictionary with the id of the prouct and its order of appearance
##
## Input:
## - df: dataframe to extract the information of the date of appearance
## - products: list of all products
##
## Output:
## - fechaIDtemp: dictionary with the date and order in which each product appears
##  
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

##
## addAparicionProdRow
## Version: 0.0.1
##
## Description: enter the temporary mark of appearance of each product in a row
##
## Input:
## - row: row a where we are going to add the value
##
## Output:
## - tiempoAnt: value that the row will take
##  
def addAparicionProdRow(row):
    tiempoAnt = -1
    
    for idF,tiempo in mapAparicionProd.mapAparicion:
        if idF <= row["Cod_Fecha_Ant"]:
            tiempoAnt = tiempo
        else:
            break;
                
        
    return tiempoAnt

##
## addTiempoProdAnt
## Version: 0.0.1
##
## Description: we create a column with the temporary mark of appearance of the product,
## from the previous date
##
## Input:
## - dfData: dataframe in which we create the new column
##
## Output:
## - df: modified dataframe
##  
def addTiempoProdAnt(dfData):
    df = dfData.copy()
    
    df["Aparicion_Prod"] = df.apply(addAparicionProdRow, axis=1)
    
    return df

##
## mapDiasInicio
## Version: 0.0.1
##
## Description: we create a map with the buyer's ID and the date of purchase of the
## first product.
##
## Input:
## - df: dataframe from which we extract information from the buyer's ID and the date
## of your first product purchased.
##
## Output:
## - mapUFecha: dictionary with the buyer's id and the date of the first product
##  bought.
##
def mapDiasInicio(df):
    dfUsuarioFecha = df[["ID_Customer","Cod_Fecha"]].groupby(["ID_Customer"]).first().reset_index()
    mapUFecha = dict(zip(dfUsuarioFecha.ID_Customer, dfUsuarioFecha.Cod_Fecha))
    
    if hasattr(mapDiasInicio,'mapUsuarioFecha'):
        mapDiasInicio.mapUsuarioFecha.update(mapUFecha)
    else:
        mapDiasInicio.mapUsuarioFecha = mapUFecha
    return mapUFecha

##
## addDiasInicioRow
## Version: 0.0.1
##
## Description: function that returns the days since the user started buying
##
## Input:
## - row: row from which we will extract the data
##
## Output:
## - val: value of the days that have passed since the user has started to buy
##  
def addDiasInicioRow(row):

    if row["Cod_Prod_Ant1"] == -1:
        val = 0
    else:
        val =  (row["Cod_Fecha_Ant"] - mapDiasInicio.mapUsuarioFecha[row["ID_Customer"]]).days
        
    return val

##
## addDiasInicioAnt
## Version: 0.0.1
##
## Description: we create a column with the days since the user starts buying
## from the previous date.
##
## Input:
## - dfData: dataframe in which we create the new column
##
## Output:
## - df: modified dataframe
##  
def addDiasInicioAnt(dfData):
    df = dfData.copy()
    
    df["DiasDesde_Inicio"] = df.apply(addDiasInicioRow, axis=1)
    
    return df

##
## numProductosComprados
## Version: 0.0.1
##
## Description: Extract the number of products purchased by user
##
## Input:
## - df: dataframe from which we extract the data
## - test: if we pass the test we must subtract 1 from the prod number
##
## Output:
## - nprods: dataframe with the number of products purchased
##  
def numProductosComprados(df, test = False):
    nprods = df.groupby("ID_Customer").Cod_Prod.count().reset_index()
    if test:
        return nprods["Cod_Prod"]
    else:
        return nprods["Cod_Prod"] - 1

##
## ultimoElementoSerie
## Version: 0.0.1
##
## Description: Extract the last element of the series from each user
##
## Input:
## - df: dataframe from which we extract the data
##
## Output:
## - df2: dataframe for each user his last series
##
def ultimoElementoSerie(df):
    df2 = df.copy()
    return df2.groupby("ID_Customer").last().reset_index()

##
## restaFechas
## Version: 0.0.1
##
## Description: we create columns with the differences of purchase dates of the
## previous products in days.
##
## Input:
## - df: dataframe from which we extract the data
##
## Output:
## - df2: dataframe with subtraction of dates applied
##  
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

##
## modificaColumna
## Version: 0.0.1
##
## Description: select the event belonging to the fusion of
## rural boxes based on the previous date
##
## Input:
## - row: row with the data that we are going to modify
##
## Output:
## - val: value that corresponds to an event in the history
##  
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

##
## acontecimiento
## Version: 0.0.1
##
## Description: add the event belonging to the fusion of
## rural boxes based on the previous date
##
## Input:
## - dfData: dataframe in which we create the new column
##
## Output:
## - df: modified dataframe
##  
def acontecimiento(dfData):
    df = dfData.copy()
    
    df["AcontecimientoAnt"] = df.apply(modificaColumna, axis=1)
    
    return df

##
## mapYearPIB
## Version: 0.0.1
##
## Description: create a dictionary with GDP per capita per year.
##
## Input:
##
## Output:
## - mapYearPIB.mapPIB: dictionary with the year associated with its GDP per capita
##
def mapYearPIB():
    dfPIB =  pd.read_csv("PIB.txt",delimiter='\t')

    mapYearPIB.mapPIB = pd.Series(dfPIB.PIB.values,index=dfPIB.Year).to_dict()
    
    return mapYearPIB.mapPIB

##
## addPIBRow
## Version: 0.0.1
##
## Description: select the GDP per capita of the date of the previous purchase
## of a customer
##
## Input:
## - row: row from which we extract the information
##
## Output:
## - val: value of the per capita value of the year of the previous purchase
##  
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

##
## addPIBAnt
## Version: 0.0.1
##
## Description: select the GDP per capita of the date of the previous purchase
## of a customer
##
## Input:
## - dfData: dataframe in which we create the new column
##
## Output:
## - df: modified dataframe
##  
def addPIBAnt(dfData):
    df = dfData.copy()
    
    df["PIB_Ant"] = df.apply(addPIBRow, axis=1)
    
    return df
