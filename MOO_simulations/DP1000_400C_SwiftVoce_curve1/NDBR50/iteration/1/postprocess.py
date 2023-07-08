# -*- coding: mbcs -*-
# Abaqus/Viewer Release 2022.HF5 replay file
# Author : Vedant Zope

import os
from abaqus import *
from abaqusConstants import *
from viewerModules import *
from driverUtils import executeOnCaeStartup

# Initialisation
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=351.333312988281, 
    height=172.66667175293)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()

# Define working directory
working_directory = os.getcwd()  

# Execute Abaqus CAE startup
executeOnCaeStartup()

# Open output database
o2 = session.openOdb(name='geometry.odb')

# Set the viewport parameters
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF,))
session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(visibleEdges=FEATURE)
session.viewports['Viewport: 1'].view.fitView()

# Print deformed shape to file
session.printToFile(
    fileName=os.path.join(working_directory,"Deformed_Specimen"), 
    format=TIFF, 
    canvasObjects=(session.viewports['Viewport: 1'], )
)

# Get nodal reaction forces and store them as xy data
odb = session.odbs[working_directory + r"\geometry.odb"]
session.xyDataListFromField(
    odb=odb, 
    outputPosition=NODAL, 
    variable=(('RF', NODAL, ((COMPONENT, 'RF2'), )), ), 
    operator=ADD, 
    nodeSets=("DISP", )
)

# Adjust the view parameters
session.viewports['Viewport: 1'].view.setValues(
    nearPlane=114.103, 
    farPlane=166.442, 
    width=22.9678, 
    height=9.09539, 
    viewOffsetX=6.08261, 
    viewOffsetY=15.6999
)

# Get nodal displacements and store them as xy data
session.xyDataListFromField(
    odb=odb, 
    outputPosition=NODAL, 
    variable=(('U', NODAL, ((COMPONENT, 'U2'), )), ), 
    operator=AVERAGE_ALL, 
    nodeSets=("DISP", )
)

# Rename xy data objects
session.xyDataObjects.changeKey(fromName='ADD_RF:RF2', toName='Force')
session.xyDataObjects.changeKey(fromName='AVERAGE_U:U2', toName='Displacement')

# Double the force values
force = session.xyDataObjects['Force']
force_double_data = [(x[0], 2.0*x[1]) for x in force.data]
force_double = session.XYData(name='Force_double', data=force_double_data, 
                              xValuesLabel=force.xValuesLabel, yValuesLabel=force.yValuesLabel)
session.xyDataObjects.changeKey(fromName='Force', toName='Force_original')
session.xyDataObjects.changeKey(fromName='Force_double', toName='Force')

# Create a new XY plot with the doubled force and displacement data
xy1 = session.xyDataObjects['Displacement']
xy2 = session.xyDataObjects['Force']
session.XYData(name='Force-Displacement', objectToCopy=xy2)
session.xyDataObjects['Force-Displacement'].setValues(sourceDescription=xy1.description)
xy3 = combine(xy1, xy2)
xyp = session.XYPlot('XYPlot-1')
chartName = list(session.charts.keys())[0]
chart = xyp.charts[chartName]
c1 = session.Curve(xyData=xy3)
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)

# Save the force-displacement curve
session.printToFile(
    fileName=os.path.join(working_directory,"FD_Curve_Plot"), 
    format=TIFF, 
    canvasObjects=(session.viewports['Viewport: 1'], )
)

# Write xy data to a text file
x0 = session.xyDataObjects['Displacement']
x1 = session.xyDataObjects['Force']  # This now refers to the doubled force
session.writeXYReport(
    fileName=os.path.join(working_directory,"FD_Curve.txt"), 
    appendMode=OFF, 
    xyData=(x0, x1)
)
