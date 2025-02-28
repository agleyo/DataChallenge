#Importations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
import holidays
import os


#Lecture des fichiers de données
meteo=pd.read_parquet("meteo.parquet")
popDepartements = pd.read_csv("Departements.csv",sep=";")
popRegions = pd.read_csv("Regions.csv",sep=";")


#Création des ratios de population
dep_present =['50', '12', '83', '22', '65', '80', '35', '51', '67', '40', '56',
       '18', '86', '17', '63', '66', '13', '44', '61', '33', '59', '68',
       '46', '34', '29', '10', '76', '26', '21', '54', '05', '37', '69',
       '91', '43', '06', '14', '09', '87', '31']

popDepartements = popDepartements[popDepartements["CODDEP"].isin(dep_present)]
popDepartements = popDepartements[["CODREG","CODDEP","PTOT"]]

popDepartements=popDepartements.merge(popDepartements.groupby("CODREG")["PTOT"].sum(),on="CODREG")
popDepartements["ratio"]=popDepartements["PTOT_x"]/popDepartements["PTOT_y"]


#Sélection des colonnes utiles/traitables
meteo.drop(columns=["phenspe1","phenspe2","phenspe3","phenspe4","nnuage1","ctype1","hnuage1","ctype2","nnuage2","hnuage2","nnuage3",
                    "ctype3","hnuage3","nnuage4","ctype4","hnuage4","coordonnees","tn12","tn24","tx12","tx24"
                    ,"tn12c","tn24c","tx12c","tx24c","tminsol","type_de_tendance_barometrique","temps_passe_1",
                    "temps_present","cod_tend","ssfrai","perssfrai","w1","w2","cl","cm","ch","niv_bar","geop",
                    "sw","tw","hbas","etat_sol","rr12","rr24","dd","longitude","latitude","tminsolc","rr6","rafper","raf10","rr3","per"],inplace=True)

#Fusion des ratios de population avec les données météo
popDepartements.rename(columns={'CODDEP':'code_dep'}, inplace=True)
meteo = meteo.merge(popDepartements[['code_dep', 'ratio']], on='code_dep', how='left')


#Séparation des données météos par région
Liste_region = meteo["nom_reg"].unique()
reg_dict = dict() 

for region in Liste_region:
    reg_dict[region]=meteo.query("nom_reg == @region").copy()
    reg_dict[region].drop(columns=["nom_reg","code_reg"],inplace=True)
    for col in reg_dict[region]:
      #On vérifie qu'on a pas plus de 30% de NaN pour chaque colonne
      #Si c'est le cas, on drop la colonne
      if reg_dict[region][col].isna().sum()>reg_dict[region].count()["numer_sta"]*0.15:
        reg_dict[region].drop(columns=[col],inplace=True)
        print("Dropped column ",col," in region",region)
    reg_dict[region]=reg_dict[region].astype({'numer_sta': 'string'})


#Création d'un tableau des vacances
vacances = pd.read_csv("data.csv",sep=",")
vacances = vacances.loc[vacances["date"]>="2017-02-13"]
vacances=vacances.drop(["nom_vacances"],axis=1)
vacances["date"]= pd.to_datetime(vacances["date"])
vacances=vacances.rename(columns={'date': 'date_temp'})


#Lecture des dataframes d'entrainement et de test
traindf = pd.read_csv("train.csv",sep=",")
traindf["date"]=pd.to_datetime(traindf["date"],utc=True)
testdf = pd.read_csv("test.csv",sep=",")
testdf["date"]=pd.to_datetime(testdf["date"],utc=True)




#Traitement des données régionales

for region in Liste_region:
    #On extrait les mesures de load dans la région concernée
    traindfReg = traindf.loc[:,(region,'date')]
    traindfReg.rename(columns={region:'load'}, inplace=True)

    #On ne sélectionne que les données numériques mais on garde la date et le numéro de station
    reg=reg_dict[region].select_dtypes(include=np.number)
    reg=reg.join(reg_dict[region][['date','numer_sta']])

    #On fait un dataframe avec toutes les dates données et demandées
    all_timestamps = pd.concat([traindfReg["date"],testdf])["date"]

    #On fait un produit cartésien pour avoir, pour chaque station, une mesure toutes les 30 minutes (qu'on va trouver par interpolation)
    all_stations = reg["numer_sta"].unique()
    df_reg = pd.MultiIndex.from_product([all_timestamps, all_stations], names=["date", "numer_sta"])
    df_reg = pd.DataFrame(index=df_reg).reset_index()
    df_reg = pd.merge(df_reg, reg ,on=["date","numer_sta"], how="left")
    interpolatedReg = pd.merge(df_reg, traindfReg ,on=["date"], how="left")
    interpolatedReg.sort_values("date",inplace=True)
    interpolatedReg["numer_sta"]= interpolatedReg["numer_sta"].astype("string")

    #Pour chaque station, on fait une interpolation des données
    liste_station = interpolatedReg["numer_sta"].unique()
    for station in liste_station :
        interpolation = interpolatedReg.query("numer_sta == @station")
        for column in interpolation.columns :
            if column!="date" and column!="numer_sta" and column != "load" :
                x=interpolation[column].interpolate('linear', limit_direction='both')
                interpolatedReg[column]=interpolatedReg[column].fillna(x)
    
    #On converti les dates pour enlever la timezone
    if not(interpolatedReg["date"].dt.tz is None):
        interpolatedReg["date"]=pd.to_datetime(interpolatedReg["date"]).dt.tz_localize(None)

    
    #On effectue une moyenne pondérée des données météorologiques entre les stations d'une même région
    weatherData=[e for e in interpolatedReg.columns if e not in ["numer_sta","date","cyclic_day","cyclic_month","ratio","mois_de_l_annee","day","hour","minute","year","week","estimated_load","loadDiff"]]
    interpolatedRegRatio = interpolatedReg[weatherData].multiply(interpolatedReg["ratio"],axis=0)
    interpolatedRegRatio=interpolatedRegRatio.merge(interpolatedReg.drop(weatherData,axis=1),how='outer',left_index=True,right_index=True)
    interpolatedRegRatio = interpolatedRegRatio.groupby(["date"],as_index=False)[weatherData].sum()
    

    #Création des données calendaires
    #Conversion des dates en array numpy
    dates= np.array(interpolatedRegRatio["date"],dtype='datetime64').astype('datetime64[s]')
    year =pd.to_datetime(dates).year
    month=pd.to_datetime(dates).month
    day=pd.to_datetime(dates).dayofweek+1
    week=pd.to_datetime(dates).isocalendar().week.reset_index(drop=True)
    hour =pd.to_datetime(dates).hour
    minute=pd.to_datetime(dates).minute
    heureHiver = np.logical_or(np.array(month)>=11,np.array(month)<=4)
    interpolatedRegRatio=pd.concat([interpolatedRegRatio,pd.DataFrame({"day":day,"hour":hour,"minute":minute}),week],axis=1)

    #Données calendaires cycliques pour supprimer la relation d'ordre
    cyclic_hour = np.sin(2 * np.pi * hour/24)
    cyclic_day = np.sin(2 * np.pi * day/7)
    cyclic_week = np.sin(2 * np.pi * week/52)
    cyclic_month = np.sin(2 * np.pi * month/12)
    
    isSaturday= day==6
    isSunday= day==7
    year = (year-np.min(year))/(np.max(year)-np.min(year))

    isHoliday = vacances[["date_temp","vacances_zone_a","vacances_zone_b","vacances_zone_c"]]
    interpolatedRegRatio["date_temp"]=pd.to_datetime(dates.astype('datetime64[D]'))
    interpolatedRegRatio=interpolatedRegRatio.merge(isHoliday,on="date_temp",how='left')
    interpolatedRegRatio=interpolatedRegRatio.drop(["date_temp"],axis=1)

    interpolatedRegRatio=pd.concat([interpolatedRegRatio,pd.DataFrame({"cyclic_hour":cyclic_hour,"cyclic_day":cyclic_day,"cyclic_week":cyclic_week,"cyclic_month":cyclic_month,
                                         "year":year,"isSaturday":isSaturday,"isSunday":isSunday,"heureHiver":heureHiver})],axis=1)
    
    
    #Estimation statistique des loads futures (juste une moyenne)
    estimatedLoad=interpolatedRegRatio.groupby(["week","day","hour","minute"])['load'].mean()
    estimatedLoad=estimatedLoad.to_frame()
    estimatedLoad.rename(columns={'load':'estimated_load'}, inplace=True)
    
    interpolatedRegRatio=pd.merge(interpolatedRegRatio,estimatedLoad ,on=["week","day","hour","minute"], how="left")
    interpolatedRegRatio=interpolatedRegRatio.drop(["day","hour","minute","week"],axis=1)

    
    
    #On cherchera à estimer la différence seulement, on a plus besoin de la valeur originale de la load
    interpolatedRegRatio['loadDiff']=interpolatedRegRatio['load']-interpolatedRegRatio['estimated_load']
    interpolatedRegRatio=interpolatedRegRatio.drop(["load"],axis=1)

    #Sauvegarde
    path = './interpolatedData/'+region+".csv"
    interpolatedRegRatio.to_csv(path)


#Gestion des données des métropoles
dict_metro = dict()
dict_metro["Montpellier Méditerranée Métropole"]= ["Occitanie","07643"]
dict_metro["Métropole Européenne de Lille"]= ["Hauts-de-France","07015"]
dict_metro["Métropole Grenoble-Alpes-Métropole"]= ["Auvergne-Rhône-Alpes","07481"]
dict_metro["Métropole Nice Côte d'Azur"]= ["Provence-Alpes-Côte d'Azur","07690"]
dict_metro["Métropole Rennes Métropole"]= ["Bretagne","07130"]
dict_metro["Métropole Rouen Normandie"]= ["Normandie","07037"]
dict_metro["Métropole d'Aix-Marseille-Provence"]= ["Provence-Alpes-Côte d'Azur","07650"]
dict_metro["Métropole de Lyon"]= ["Auvergne-Rhône-Alpes","07481"]
dict_metro["Métropole du Grand Nancy"]= ["Grand Est","07181"]
dict_metro["Métropole du Grand Paris"]= ["Île-de-France","07149"]
dict_metro["Nantes Métropole"]= ["Pays de la Loire","07222"]
dict_metro["Toulouse Métropole"]= ["Occitanie","07630"]


for metropole in dict_metro:
    
    #On extrait les mesures de load dans la métropole concernée
    region = dict_metro[metropole][0]
    numer_sta = dict_metro[metropole][1]
    load_df_sta = traindf.loc[:,(metropole,'date')]
    load_df_sta.rename(columns={metropole:'load'}, inplace=True)
    print(metropole)
    print(region)
    print(numer_sta)

    #On obtient tous le timestamps donnés et demandés de la métropole
    all_timestamps = pd.concat([load_df_sta,testdf])

    #On y associe les données météorologiques de la station associée
    all_timestamps = pd.merge(all_timestamps,reg_dict[region].loc[reg_dict[region]["numer_sta"]==numer_sta], on=["date"], how="left")

    #Interpolation des données 
    interpolate=all_timestamps.select_dtypes(include=['number','datetime64[ns, UTC]'])
    for column in interpolate.columns :
            if column!="date" and column != "load" and column != "numer_sta":
                interpolate[column]=interpolate[column].interpolate('linear', limit=10) #Interpolation sur 5 heures max
    
    #Suppression du la timezone
    all_timestamps = interpolate.merge(all_timestamps['date'],how='left')
    if not(all_timestamps["date"].dt.tz is None):
        all_timestamps["date"]=pd.to_datetime(all_timestamps["date"]).dt.tz_localize(None)
    
    #Données calendaires
    dates= np.array(all_timestamps["date"],dtype='datetime64').astype('datetime64[s]')

    year =pd.to_datetime(dates).year
    month=pd.to_datetime(dates).month
    day=pd.to_datetime(dates).dayofweek+1
    week=pd.to_datetime(dates).isocalendar().week.reset_index(drop=True)
    hour =pd.to_datetime(dates).hour
    minute=pd.to_datetime(dates).minute
    cyclic_hour = np.sin(2 * np.pi * hour/24)
    cyclic_day = np.sin(2 * np.pi * day/7)
    cyclic_week = np.sin(2 * np.pi * week/52)
    cyclic_month = np.sin(2 * np.pi * month/12)
    
    isSaturday= day==6
    isSunday= day==7


    heureHiver = np.logical_or(np.array(month)>=11,np.array(month)<=4)

    isHoliday = vacances[["date_temp","vacances_zone_a","vacances_zone_b","vacances_zone_c"]]
    all_timestamps["date_temp"]=pd.to_datetime(dates.astype('datetime64[D]'))
    all_timestamps=all_timestamps.merge(isHoliday,on="date_temp",how='left')
    all_timestamps=all_timestamps.drop(["date_temp"],axis=1)

    all_timestamps=pd.concat([all_timestamps,pd.DataFrame({"cyclic_hour":cyclic_hour,"cyclic_day":cyclic_day,"cyclic_week":cyclic_week,"cyclic_month":cyclic_month,
                                         "year":year,"isSaturday":isSaturday,"isSunday":isSunday,"heureHiver":heureHiver})],axis=1)

    
    #Estimations statistiques
    all_timestamps=pd.concat([all_timestamps,pd.DataFrame({"day":day,"hour":hour,"minute":minute}),week],axis=1)
    estimatedLoad=all_timestamps.groupby(["week","day","hour","minute"])['load'].mean()
    estimatedLoad=estimatedLoad.to_frame()
    estimatedLoad.rename(columns={'load':'estimated_load'}, inplace=True)
    all_timestamps=pd.merge(all_timestamps,estimatedLoad ,on=["week","day","hour","minute"], how="left")

    all_timestamps=all_timestamps.drop(["day","hour","minute","week"],axis=1)
    
    
    
    #On ne garde que la différence et l'estimation
    all_timestamps['loadDiff']=all_timestamps['load']-all_timestamps['estimated_load']
    all_timestamps=all_timestamps.drop(["load"],axis=1)

    #Sauvegarde
    path = './interpolatedData/'+metropole+".csv"
    all_timestamps.to_csv(path)
