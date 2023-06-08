# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2022.HF5 replay file
# Internal Version: 2022_09_26-18.03.59 176852
# Run by zopev1 on Mon May 22 15:07:10 2023
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=351.333312988281, 
    height=172.66667175293)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from viewerModules import *
from driverUtils import executeOnCaeStartup
#==========================Initialise a working director===============================
working_directory = r"C:\Users\zopev1\Downloads\Copy-Macro Optimization"
# ========================= Select a folder for all files ===============================
img_Folder=r'Exported_Images/'
txt_Folder=r'Exported_Texts/'

executeOnCaeStartup()
o2 = session.openOdb(name='ndb50.odb')
#: Model: C:/SIMULIA/temp/New_ndb50/ndb50.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       10
#: Number of Node Sets:          6
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(
    visibleEdges=FEATURE)
session.viewports['Viewport: 1'].view.fitView()
session.printToFile(fileName=working_directory+'/'+img_Folder+'Deformed Model', format=TIFF, canvasObjects=(
    session.viewports['Viewport: 1'], ))
#working directory
odb = session.odbs[working_directory+r"\ndb50.odb"]
session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', 
    NODAL, ((COMPONENT, 'RF2'), )), ), operator=ADD, nodeSets=("DISP", ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=114.103, 
    farPlane=166.442, width=22.9678, height=9.09539, viewOffsetX=6.08261, 
    viewOffsetY=15.6999)
odb = session.odbs[working_directory+r"\ndb50.odb"]
session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('U', 
    NODAL, ((COMPONENT, 'U2'), )), ), nodePick=(('MODEL', 1, ('[#0:984 #40 ]', 
    )), ), )
xy1 = session.xyDataObjects['U:U2 PI: MODEL N: 30837']
xy2 = session.xyDataObjects['ADD_RF:RF2']
xy3 = combine(xy1, xy2)
xyp = session.XYPlot('XYPlot-1')
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
c1 = session.Curve(xyData=xy3)
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
session.printToFile(fileName=img_Folder+'F-D Curve', format=TIFF, canvasObjects=(
    session.viewports['Viewport: 1'], ))
session.xyDataObjects.changeKey(fromName='ADD_RF:RF2', toName='Force')
session.xyDataObjects.changeKey(fromName='U:U2 PI: MODEL N: 30837', 
    toName='Displacement')
x0 = session.xyDataObjects['Displacement']
x1 = session.xyDataObjects['Force']
session.writeXYReport(fileName=txt_Folder+'F-D data'+'.txt', appendMode=OFF, xyData=(x0, x1))
