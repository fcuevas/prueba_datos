#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 16:36:49 2018
 
@author: fcuevas
"""
from math import pi

import xarray as xr
import numpy as np
import pandas as pd
import holoviews as hv
import geoviews as gv
import geoviews.feature as gf

import cartopy
from cartopy import crs as ccrs
from colorcet import cm_n

from bokeh.io import curdoc
from bokeh.tile_providers import STAMEN_TONER
from bokeh.plotting import Figure
from bokeh.models import WMTSTileSource, OpenURL, TapTool, HoverTool, FactorRange, DatetimeTickFormatter
#from bokeh.models.callbacks import CustomJS

renderer = hv.renderer('bokeh').instance(mode='server')
hv.extension('bokeh')
########################################################################################
tiles = {'OpenMap': WMTSTileSource(url='http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png'),
         'ESRI': WMTSTileSource(url='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}.jpg'),
         'Wikipedia': WMTSTileSource(url='https://maps.wikimedia.org/osm-intl/{Z}/{X}/{Y}@2x.png'),
         'Stamen Toner': STAMEN_TONER}
########################################################################################
#header = ['ID_Central','ID_Propietario','ID_Localidad','ID_Comuna','ID_CentralTipo','ID_Fuente',
#          'GrupoNombreNemo','NombrePropietario','NemoPropietario','NombreComuna','NombrePlantaTipo', 	
#          'NombreFuente','centralNumero','centralNemo','centralNombre',
#          'Estado','Punto_conexion','Potencia','Capacidad_maxima','Fecha_entrada',
#          'Clasificacion','Distribuidora','Region','Numero_comunas','Coordenada_este','Coordenada_norte','Zona']
##solar_tot = pd.read_excel('../datos/centrales/1_resumen_centrales/centrales_4.xlsx', sheet_name='Solar',encoding="ISO-8859-1", 
##                      names=header,skiprows=1)
#
#solar_tot = pd.read_csv('../datos/centrales/1_resumen_centrales/plantaSolar.csv', encoding="ISO-8859-1", 
#                      names=header,skiprows=1,sep=';',decimal=',')
#solar = solar_tot[solar_tot.Coordenada_este.notna()]
#wl = []
#for pl in solar.ID_Central:
#    print (pl)
#    try:
#        wl_tmp = "http://localhost:6003/solarPlants_dev" + "?idPlanta=" + str(int(pl))
#    except (ValueError):
#        wl_tmp = "http://localhost:6003/solarPlants_dev" + "?idPlanta="
#    wl.append(wl_tmp)
#    
    
solar = pd.read_csv('eclipse2019/infoFV_ft.csv', encoding="ISO-8859-1")
pt_size = np.log(solar.Potencia+1)
solar['pt_size'] = pt_size
wl = []
for pl in solar.idPlanta:
    try:
       wl_tmp = "http://localhost:5509/dash" + "?idPlanta=" + str(int(pl))
        # wl_tmp = "http://www.programaenergiasolar.cl/dash" + "?idPlanta=" + str(int(pl))
    except (ValueError):
       wl_tmp = "http://localhost:5509/dash" + "?idPlanta="
        # wl_tmp = "http://www.programaenergiasolar.cl/dash" + "?idPlanta="
    wl.append(wl_tmp)
    
solar['weblink'] = wl

from pyproj import Proj, transform

outProj = Proj(init='epsg:4326')
inProj = Proj(init='epsg:32719')
lt = []
lg = []
#for utm_e, utm_o in zip(solar.Coordenada_este, solar.Coordenada_norte):
for utm_e, utm_o in zip(solar.UTM_este, solar.UTM_oeste):
    x2,y2 = transform(inProj,outProj,utm_e,utm_o)
    lg.append(x2)
    lt.append(y2)
    
solar['Longitud'] = lg
solar['Latitud'] = lt
########################################################################################
ecl = pd.read_csv('eclipse2019/plantas_tabla.txt', encoding="ISO-8859-1")
ecl.c1 = pd.to_datetime(ecl.c1)
#ecl.c2 = pd.to_datetime(ecl.c2)
#ecl.c3 = pd.to_datetime(ecl.c3)
ecl.c4 = pd.to_datetime(ecl.c4)
ecl.cMax = pd.to_datetime(ecl.cMax)
ecl.sunset = pd.to_datetime(ecl.sunset)


df = pd.concat([ecl,solar],axis=1)
df = df.dropna()


fv = df.Nombre

TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select,save"
p1 = Figure(name="p1", tools=TOOLS, sizing_mode="scale_width", plot_width=1200, plot_height=450, x_range=FactorRange(*fv)) 
p1.vbar(x=fv,top=df.oscuridad, width=0.6, line_color='orangered',
        fill_color='orangered')
p1.xaxis.major_label_orientation = pi/3
p1.yaxis.axis_label= "Porcentaje oscuridad"



p2 = Figure(name="p2", tools=TOOLS, plot_width=1200, plot_height=900, y_range=FactorRange(factors=list(df.Nombre)), sizing_mode="scale_width") #list(data2.index)
p2.xaxis.formatter=DatetimeTickFormatter(hours = ['%d/%m %H:00'],days = ['%F'])

p2.hbar(y=list(df.Nombre),left=df.c1, right=df.c4, height=0.4, 
         color='orangered')

########################################################################################
hst = df[['oscuridad','Potencia']]

v1 = hst.Potencia[hst.oscuridad > 95].sum()
v2 = hst.Potencia[(hst.oscuridad > 90) & (hst.oscuridad < 95)].sum()
v3 = hst.Potencia[(hst.oscuridad > 85) & (hst.oscuridad < 90)].sum()
v4 = hst.Potencia[(hst.oscuridad > 80) & (hst.oscuridad < 85)].sum()
v5 = hst.Potencia[(hst.oscuridad > 75) & (hst.oscuridad < 80)].sum()
v6 = hst.Potencia[(hst.oscuridad > 70) & (hst.oscuridad < 75)].sum()
v7 = hst.Potencia[(hst.oscuridad > 65) & (hst.oscuridad < 70)].sum()
v8 = hst.Potencia[(hst.oscuridad > 60) & (hst.oscuridad < 65)].sum()
v9 = hst.Potencia[(hst.oscuridad > 55) & (hst.oscuridad < 60)].sum()
v10 = hst.Potencia[(hst.oscuridad > 50) & (hst.oscuridad < 55)].sum()

ac = np.array([v10,v9,v8,v7,v6,v5,v4,v3,v2,v1])
edges=np.arange(50,101,5)

p3 = Figure(tools=TOOLS)
p3.quad(top=ac, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="orangered", line_color="white", alpha=1.0)

p3.xaxis.axis_label= "Porcentaje oscuridad (%)"
p3.yaxis.axis_label= "Potencia (MW)"
########################################################################################
# est_minEner = pd.read_csv('../datos/estaciones_MinEner/infoEstaciones.csv', sep=';', encoding="ISO-8859-1")
# est_minEner['Mandante'] = 'Min Energia'
# #est_minEner["weblink"] = "http://localhost:6010/est_minEner"

# wl = []
# for pl in est_minEner.index:
#    try:
#        wl_tmp = "http://localhost:6010/est_minEner" + "?idEst=" + str(int(pl))
#    except (ValueError):
#        wl_tmp = "http://localhost:6010/est_minEner" + "?idEst="
#    wl.append(wl_tmp)
   
# est_minEner['weblink'] = wl
#########################################################################################
est_DMC = pd.read_csv('eclipse2019/estacionesDMC.csv', sep=';', encoding="ISO-8859-1")
est_DMC['Ejecutor'] = 'Direccion Metereologica de Chile'

wl = []
for cd in est_DMC.Codigo:
    try:
       wl_tmp = "http://localhost:5504/est_DMC_dev" + "?idEst=" + str(int(cd))
        # wl_tmp = "http://www.programaenergiasolar.cl/est_DMC_dev" + "?idEst=" + str(int(cd))
    except (ValueError):
       wl_tmp = "http://localhost:5504/est_DMC_dev" + "?idEst="
        # wl_tmp = "http://www.programaenergiasolar.cl/est_DMC_dev" + "?idEst="
    wl.append(wl_tmp)

est_DMC['weblink'] = wl    

lt = []
lg = []
for val in est_DMC.index:
    lt_gr = float(str(est_DMC['Latitud'][val])[1:3])
    lt_mn = float(str(est_DMC['Latitud'][val])[4:6])/60
    lt_sc = float(str(est_DMC['Latitud'][val])[6:8])/3600
    lg_gr = float(str(est_DMC['Longitud'][val])[1:3])
    lg_mn = float(str(est_DMC['Longitud'][val])[4:6])/60
    lg_sc = float(str(est_DMC['Longitud'][val])[6:8])/3600
    
    lt_dc = -(lt_gr + lt_mn + lt_sc)
    lt.append(lt_dc)
    lg_dc = -(lg_gr + lg_mn + lg_sc)
    lg.append(lg_dc)

est_DMC['Latitud'] = lt
est_DMC['Longitud'] = lg
#########################################################################################
# est_esp = pd.read_csv('../datos/medicion_espectro/estaciones_espectro.csv', sep=';', encoding="ISO-8859-1")
# est_esp['Ejecutor'] = 'USACH'
# est_esp['weblink'] = "http://localhost:6011/med_espectro"
# est_esp['Proyecto'] = '15BPE2-47233'

# lt = []
# lg = []
# for val in est_esp.index:
#    lt_gr = float(str(est_esp['Latitud'][val])[1:3])
#    lt_mn = float(str(est_esp['Latitud'][val])[4:6])/60
#    lt_sc = float(str(est_esp['Latitud'][val])[6:8])/3600
#    lg_gr = float(str(est_esp['Longitud'][val])[1:3])
#    lg_mn = float(str(est_esp['Longitud'][val])[4:6])/60
#    lg_sc = float(str(est_esp['Longitud'][val])[6:8])/3600
   
#    lt_dc = -(lt_gr + lt_mn + lt_sc)
#    lt.append(lt_dc)
#    lg_dc = -(lg_gr + lg_mn + lg_sc)
#    lg.append(lg_dc)

# est_esp['Latitud'] = lt
# est_esp['Longitud'] = lg
#########################################################################################
# est_soil = pd.read_csv('../datos/soiling_fv/estaciones_soiling.csv', sep=';', encoding="ISO-8859-1")
# est_soil['Ejecutor'] = 'USACH'

# wl = []
# for cd in est_soil.index:
#     try:
#         # wl_tmp = "http://www.programaenergiasolar.cl/est_soiling_dev" + "?idEst=" + str(int(cd))
#        wl_tmp = "http://localhost:5505/est_soiling_dev" + "?idEst=" + str(int(cd))
#     except (ValueError):
#         # wl_tmp = "http://www.programaenergiasolar.cl/est_soiling_dev" + "?idEst="
#        wl_tmp = "http://localhost:5505/est_soiling_dev" + "?idEst=" 
#     wl.append(wl_tmp)

# est_soil['weblink'] = wl  

# est_soil['Proyecto'] = '15BP-45364'
# est_soil['Periodo'] = 2017
##########################################################################################
est_DTS = pd.read_csv('eclipse2019/DTS.csv', sep=';', encoding="ISO-8859-1")
est_DTS['size'] = 45

est_DTS['Mandante'] = 'Comite Solar'
est_DTS['Ejecutor'] = 'Fraunhofer CSET'
est_DTS['Proyecto'] = '15PEDN-57256'

wl = ["http://localhost:5507/recursoSolarDA_dev",
     "http://localhost:5508/soilingDA_dev",
     "http://localhost:5506/perf_ratioDA_dev"]

# wl = ["http://www.programaenergiasolar.cl/recursoSolarDA_dev",
#       "http://www.programaenergiasolar.cl/soilingDA_dev",
#       "http://www.programaenergiasolar.cl/perf_ratioDA_dev"]

est_DTS['weblink'] = wl

lt = []
lg = []
for val in est_DTS.index:
    lt_gr = float(str(est_DTS['Latitud'][val])[1:3])
    lt_mn = float(str(est_DTS['Latitud'][val])[4:6])/60
    lt_sc = float(str(est_DTS['Latitud'][val])[6:8])/3600
    lg_gr = float(str(est_DTS['Longitud'][val])[1:3])
    lg_mn = float(str(est_DTS['Longitud'][val])[4:6])/60
    lg_sc = float(str(est_DTS['Longitud'][val])[6:8])/3600
    
    lt_dc = -(lt_gr + lt_mn + lt_sc)
    lt.append(lt_dc)
    lg_dc = -(lg_gr + lg_mn + lg_sc)
    lg.append(lg_dc)

est_DTS['Latitud'] = lt
est_DTS['Longitud'] = lg
#########################################################################################
solar_gv = gv.Dataset(solar, kdims=['centralNombre'])
est_DMC_gv = gv.Dataset(est_DMC, kdims=['Estacion'])
est_DTS_gv = gv.Dataset(est_DTS, kdims=['Estacion'])
# est_minEner_gv = gv.Dataset(est_minEner, kdims=['Estacion'])
# est_esp_gv = gv.Dataset(est_esp, kdims=['Estacion'])
# est_soil_gv = gv.Dataset(est_soil, kdims=['Estacion'])
########################################################################################
#TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,reset,save"
tile_options = dict(width=700,height=800,xaxis=None,yaxis=None,bgcolor='black',show_grid=False,tools=['pan','wheel_zoom'])
#point_options = dict(alpha = 0.6, tools=['hover','tap','box_select'])

layer_1 = gv.WMTS(tiles['ESRI']).opts(plot=tile_options)

#####################
url2 = "@weblink"
#url2 = "@weblink" + "?idPlanta=" + "@idPlanta"

callback2 = OpenURL(url=url2)
tap2 = TapTool(callback=callback2)

ns_alpha=0.75
mt_alpha=0.0
#####################
hover2 = HoverTool(tooltips=[("Nombre central: ", "@centralNombre"),
                             ("Potencia: ", "@Potencia{(0.0)}"),
                             ("Nombre comuna: ","@NombreComuna")]) 

layer_2 = solar_gv.to(gv.Points , kdims=['Longitud', 'Latitud'],
              vdims=['pt_size', 'Potencia','centralNombre','NombreComuna','weblink'], 
              crs=ccrs.PlateCarree()).options(color='orangered',size_index=2, size=4.5, tools=[hover2, tap2,'pan','wheel_zoom'],
                                  nonselection_alpha=ns_alpha, nonselection_color='orangered', legend_position='top_left',
                                  muted_alpha=mt_alpha)
#, 
#                    'ID_Central','Punto_conexion','NombrePropietario','Estado','Region'
#("Región: ", "@Region"
#,
# ("Punto conexión: ", "@Punto_conexion"),
# ("Nombre propietario: ", "@NombrePropietario"),
# ("Estado: ", "@Estado"),

######################
# hover3 = HoverTool(tooltips=[("Estacion: ", "@Estacion"),
#                             ("Mandante: ", "@Mandante"),
#                             ("Fecha inicio: ","@Fecha_inicio"),
#                             ("Datos hasta: ","@Fecha_fin")])
  
# layer_3 = est_minEner_gv.to(gv.Points, kdims=['Longitud', 'Latitud'],
#                 vdims=['Estacion', 'Mandante','Fecha_inicio','Fecha_fin','weblink'], 
#                 crs=ccrs.PlateCarree()).options(color='darkcyan',size=10, tools=[hover3, tap2],
#                                  nonselection_alpha=ns_alpha, nonselection_color='darkcyan')

######################
hover4 = HoverTool(tooltips=[("Estacion: ", "@Estacion"),
                             ("Código DMC estación: ","@Codigo"),
                             ("Ejecutor: ", "@Ejecutor"),
                             ("Fecha inicio: ","@Fecha_inicio"),
                             ("Variables medidas: ","@Variables_medidas")])
layer_4 = est_DMC_gv.to(gv.Points, kdims=['Longitud', 'Latitud'],
                 vdims=['Estacion', 'Ejecutor','Fecha_inicio','Variables_medidas','weblink','Codigo'], 
                 crs=ccrs.PlateCarree()).options(color='blue',size=10, tools=[hover4, tap2,'pan','wheel_zoom'],
                                  nonselection_alpha=ns_alpha, nonselection_color='blue',
                                  muted_alpha=mt_alpha) 

######################
hover5 = HoverTool(tooltips=[("Estacion: ", "@Estacion"),
                             ("Proyecto: ", "@Proyecto"),
                             ("Mandante: ", "@Mandante"),
                             ("Ejecutor: ", "@Ejecutor"),
                             ("Fecha inicio: ","@Fecha_inicio"),
                             ("Variables medidas: ","@Variables")])
layer_5 = est_DTS_gv.to(gv.Points, kdims=['Longitud', 'Latitud'],
                 vdims=['Estacion','Proyecto','Mandante','Ejecutor','Fecha_inicio','Variables','weblink'], 
              crs=ccrs.PlateCarree()).options(color='green',size=10, tools=[hover5, tap2,'pan','wheel_zoom'],
                                  nonselection_alpha=ns_alpha, nonselection_color='green',
                                  muted_alpha=mt_alpha) 

######################
# hover6 = HoverTool(tooltips=[("Estacion: ", "@Estacion"),
#                             ("Altura (msnm): ", "@Altura"),
#                             ("Proyecto: ", "@Proyecto"),
#                             ("Ejecutor: ", "@Ejecutor"),
#                             ("Fecha inicio: ","@Fecha_inicio"),
#                             ("Fecha fin: ","@Fecha_fin"),
#                             ("Variables medidas: ","@Variable")])
# layer_6 = est_esp_gv.to(gv.Points, kdims=['Longitud', 'Latitud'],
#                 vdims=['Estacion','Altura','Proyecto','Ejecutor','Fecha_inicio','Fecha_fin', 'Variable','weblink'], 
#              crs=ccrs.PlateCarree()).options(color='yellow',size=11, tools=[hover6, tap2],
#                                  nonselection_alpha=ns_alpha, nonselection_color='yellow') 

######################
# hover7 = HoverTool(tooltips=[("Estacion: ", "@Estacion"),
#                              ("Altura (msnm): ", "@Altura"),
#                              ("Proyecto: ", "@Proyecto"),
#                              ("Ejecutor: ", "@Ejecutor"),
#                              ("Período medición: ","@Periodo"),
#                              ("Variables medidas: ","@Variable")])

# layer_7 = est_soil_gv.to(gv.Points, kdims=['Longitud', 'Latitud'],
#                  vdims=['Estacion','Altura', 'Proyecto','Ejecutor','Periodo','Variable','weblink'], 
#               crs=ccrs.PlateCarree()).options(color='magenta', size=10, tools=[hover7, tap2,'pan','wheel_zoom'],
#                                   nonselection_alpha=ns_alpha, nonselection_color='magenta',
#                                   muted_alpha=mt_alpha)
#
######################

#mapa = layer_1 * hv.NdOverlay({'Plantas FV':layer_2, 'Estaciones Min Energía':layer_3, 
#                               'Estaciones DMC':layer_4, 'DTS':layer_5, 
#                               'Medición espectro solar':layer_6, 'Estaciones soiling':layer_7})  ,'Estaciones Min Energía':layer_3


#shapefile = 'ecl/eclipsefinal.shp'
#layer_gis = gv.Shape.from_shapefile(shapefile, crs=ccrs.PlateCarree())

# shapefile = 'eclipse2019/boundaries/boundaries2.shp'
# layer_gis = gv.Shape.from_shapefile(shapefile, crs=ccrs.PlateCarree())

shapefile = 'eclipse2019/boundaries/eclipse.shp'
# gv.Shape.from_shapefile(shapefile, crs=ccrs.PlateCarree())

shapes = cartopy.io.shapereader.Reader(shapefile)

referendum = pd.read_csv('eclipse2019/eclipse.csv')
referendum = hv.Dataset(referendum)
layer_gis = gv.Shape.from_records(shapes.records(), referendum, on='code', value='Magnitud').opts(tools=['hover'],width=900, height=900, cmap='Wistia',alpha=0.5,colorbar=True)

#gis = xr.open_rasterio('eclipse2019/ecl/eclipse1kmv2_EPSG3857.tif').load()[0]
#np.place(gis.values,gis.values<-9998.0,[np.nan])
#layer_gis = hv.Image(gis).redim(x='Longitude', y='Latitude').options(cmap=cm_n["rainbow"], colorbar=True,
#                           clipping_colors={'NaN': (0, 0, 0, 0)}, tools=['hover']) 


#layer_gis = hv.Image(gis).redim(x='Longitude', y='Latitude').options(cmap=['#0000ff', '#8888ff', '#ffffff', '#ff8888', '#ff0000'],
#                    colorbar=True,clipping_colors={'NaN': (0, 0, 0, 0)}) 

mapa = layer_gis * layer_1 *  hv.NdOverlay({'Plantas Solares SEN':layer_2, 
                               'Estaciones meteorológicas DMC':layer_4,
                               'Estación Diego de Almagro':layer_5}) 

#mapa.toolbar.reset = None
#mapa.toolbar.
########################################################################################
doc = renderer.server_doc(mapa)
doc.title = 'Mapa'

curdoc().add_root(p1)
curdoc().add_root(p2)
curdoc().add_root(p3)