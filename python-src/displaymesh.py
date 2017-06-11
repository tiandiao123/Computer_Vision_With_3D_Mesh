'''
You can translate a STEP file into an OCCT shape in the following steps:
 1.load the file,
 2.check file consistency,
 3.set the translation parameters,
 4.perform the translation,
 5.fetch the results.
'''

import sys
from OCC.Display.SimpleGui import init_display
from OCC.IFSelect import IFSelect_RetDone,IFSelect_ItemsByEntity

# Reads STEP files, checks them and translates their contents into Open CASCADE models
from OCC.STEPControl import STEPControl_Reader


# Creates a reader object with an empty STEP mode
step_reader = STEPControl_Reader()

# Loads a file and returns the read status
status = step_reader.ReadFile('Drone.step')

# check status 
if status == IFSelect_RetDone:  # RetDone : normal execution with a result
    # Checking the STEP file
    # Error messages are displayed if there are invalid or incomplete STEP entities
    step_reader.PrintCheckLoad(True, IFSelect_ItemsByEntity)

    # Performing the STEP file translation
    step_reader.TransferRoot()

    # Each successful translation operation outputs one shape
    # Returns the shape resulting from a translation
    shape = step_reader.Shape()
else:
    print("Error: can't read file.")
    sys.exit(0)
          
# initializes the display
display, start_display, add_menu, add_function_to_menu = init_display()

# Then the shape is sent to the renderer
display.DisplayShape(shape, update=True)

# enter the gui mainloop
start_display()