# Import the necessary libraries/modules from Abaqus
from abaqus import *
from abaqusConstants import *
import os

# Setup a new viewport with given parameters
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=351.333312988281, height=172.66667175293)
# Set the current viewport to be 'Viewport: 1'
session.viewports['Viewport: 1'].makeCurrent()
# Maximize the current viewport
session.viewports['Viewport: 1'].maximize()

# Import additional necessary modules
from viewerModules import *
from driverUtils import executeOnCaeStartup

# Define the current working directory
working_directory = os.getcwd()
# Create folders for storing exported images and texts
os.makedirs('Exported_Images')
os.makedirs('Exported_Texts')

# Define folders for storing images and text files
img_Folder='Exported_Images'
txt_Folder='Exported_Texts'

# Execute Abaqus startup
executeOnCaeStartup()

# Open the output database (odb) file 'ndb50.odb'
o2 = session.openOdb(name='ndb50.odb')

# Set the current viewport's displayed object to the odb
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(visibleEdges=FEATURE)
# Fit the model to the viewport window
session.viewports['Viewport: 1'].view.fitView()

# Save the deformation image
session.printToFile(fileName=f"{working_directory}/{img_Folder}/Deformed_Model", format=TIFF, canvasObjects=(session.viewports['Viewport: 1'], ))

# Open the odb file for further operations
odb = session.odbs[working_directory+r"\ndb50.odb"]

# Extract the reaction force (RF) from the odb and add it to session's xy data
session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', NODAL, ((COMPONENT, 'RF2'), )), ), operator=ADD, nodeSets=("DISP", ))

# Modify the viewport's viewing parameters
session.viewports['Viewport: 1'].view.setValues(nearPlane=114.103, farPlane=166.442, width=22.9678, height=9.09539, viewOffsetX=6.08261, viewOffsetY=15.6999)

# Extract the displacement field (U) from the odb and add it to session's xy data
session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('U', NODAL, ((COMPONENT, 'U2'), )), ), nodePick=(('MODEL', 1, ('[#0:984 #40 ]', )), ), )

# Create a new combined xy data object by combining displacement and force
xy1 = session.xyDataObjects['U:U2 PI: MODEL N: 30837']
xy2 = session.xyDataObjects['ADD_RF:RF2']
xy3 = combine(xy1, xy2)

# Create a new XY plot and plot the combined data
xyp = session.XYPlot('XYPlot-1')
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
c1 = session.Curve(xyData=xy3)
chart.setValues(curvesToPlot=(c1, ), )

# Automatically assign color to lines and symbols in the chart
session.charts[chartName].autoColor(lines=True, symbols=True)

# Set the current viewport's displayed object to the xy plot
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)

# Save the force-displacement curve as an image
session.printToFile(fileName=f"{img_Folder}/F-D_Curve", format=TIFF, canvasObjects=(session.viewports['Viewport: 1'], ))

# Rename the xy data objects for better readability
session.xyDataObjects.changeKey(fromName='ADD_RF:RF2', toName='Force')
session.xyDataObjects.changeKey(fromName='U:U2 PI: MODEL N: 30837', toName='Displacement')

# Define the xy data objects for displacement and force
x0 = session.xyDataObjects['Displacement']
x1 = session.xyDataObjects['Force']

# Write the force and displacement data to a text file
session.writeXYReport(fileName=f"{txt_Folder}/F-D_data.txt", appendMode=OFF, xyData=(x0, x1))
