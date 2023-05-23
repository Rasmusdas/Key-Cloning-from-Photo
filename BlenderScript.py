import bpy
import math
from pathlib import Path
from bpy import context
import os
import sys

bpy.ops.object.select_all(action='SELECT')

bpy.ops.object.delete(use_global=False, confirm=False)


dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir )
    
from main import generate_key_process
argv = sys.argv
image_name = argv[argv.index("--imgName") + 1]
heightProfile = float(argv[argv.index("--bladeHeight") + 1])
width = float(argv[argv.index("--keyThick") + 1])
heightHead = float(argv[argv.index("--headHeight") + 1])
length = float(argv[argv.index("--keyLength") + 1])

kernelSize = int(argv[argv.index("--kernelSize") + 1])
medianBlur = int(argv[argv.index("--medianBlur") + 1])
noiseRemoval = int(argv[argv.index("--noiseRemoval") + 1])
lineThresh = int(argv[argv.index("--lineThresh") + 1])
erosionSize = int(argv[argv.index("--erosionSize") + 1])
scaleFactor = int(argv[argv.index("--scaleFactor") + 1])


print(f"Creating Key of image: {image_name}")

#Hjælpe funktioner
def ImportSVG(path):
    return bpy.ops.import_curve.svg(filepath=path)

def SetObjLocation(object, loc):
    object.location = loc

def SetObjRotation(object, rot):
    obj.rotation_euler[0] = math.radians(rot[0])
    obj.rotation_euler[1] = math.radians(rot[1])
    obj.rotation_euler[2] = math.radians(rot[2])

def Editmode(bool):
    if bool:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='VERT')
    else:
        bpy.ops.object.mode_set(mode='OBJECT')

def Extrude(amount):
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.extrude_region_move(MESH_OT_extrude_region={"use_normal_flip":False, "use_dissolve_ortho_edges":False, "mirror":False}, TRANSFORM_OT_translate={"value":amount, "orient_axis_ortho":'X', "orient_type":'NORMAL', "orient_matrix":((0, 4.37114e-08, -1), (1, 0, 0), (0, -1, -4.37114e-08)), "orient_matrix_type":'NORMAL', "constraint_axis":(False, False, True), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_target":'CLOSEST', "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})

def BooleanMod(objInter, mode = 0):
    bpy.ops.object.modifier_add(type='BOOLEAN')
    bpy.context.object.modifiers["Boolean"].object = objInter
    if mode == 0:
        bpy.context.object.modifiers["Boolean"].operation = 'INTERSECT'
    else:
         bpy.context.object.modifiers["Boolean"].operation = 'DIFFERENCE'
    bpy.ops.object.modifier_apply(modifier="Boolean") 

def SelectObj(name):
    obj = bpy.context.scene.objects[name] 
    bpy.ops.object.select_all(action='DESELECT') 
    bpy.context.view_layer.objects.active = obj   
    obj.select_set(True)
    return obj   

def DeletObj(name):
    obj = SelectObj(name)
    bpy.ops.object.delete(use_global=False, confirm=False)
    
def Export(name, path):
    SelectObj(name)
    path = Path(path)
    stl_path = path / f"key.stl"
    bpy.ops.export_mesh.stl(filepath=str(stl_path), use_selection=True)
    
    
#Side dimensions
generate_key_process("",image_name,scaleFactor,kernelSize,1,medianBlur,noiseRemoval,lineThresh,erosionSize,heightHead*100)

offset = -0.0
# defalt -0.088
# RU14Farfar -0.092
# RU14Ras -0.094
script_directory = os.path.dirname(os.path.abspath(__file__))

#Profile dimensions

exportPath = script_directory+"\\out\\"
svgPath = script_directory+"\\out\\"
profilePath = script_directory+"\\key_profiles\\"
sideSVG = "generated_out.svg"
profileSVG = "RUKO_ru14.svg"

#Import af sideview svg   
obj = ImportSVG(svgPath + sideSVG) 
obj = SelectObj("Curve")  
bpy.context.active_object.name = 'Side'

#Sætter dimensionerne til den rigtige størelse 
dims = bpy.context.object.dimensions
bpy.context.object.dimensions = length, heightHead, dims[2]

#Reducer antal vertesis (er muligvis ikke nødvengit)
bpy.context.object.data.resolution_u = 6
bpy.ops.object.convert(target='MESH')

#Flytter og rotere i 3D space
SetObjLocation(obj, (0,1,offset))
SetObjRotation(obj, (90,0,0))
Editmode(True)
Extrude((0,0,2))
Editmode(False)
bpy.ops.transform.resize(value=(1, 1, 1), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)

#Import af profile svg 
obj = ImportSVG(profilePath + profileSVG) 
obj = SelectObj("Curve")  
bpy.context.active_object.name = 'Front'

#Sætter dimensionerne til den rigtige størelse 
dims = bpy.context.object.dimensions
bpy.context.object.dimensions = heightProfile, width, dims[2]

#Reducer antal vertesis (er muligvis ikke nødvengit)
bpy.context.object.data.resolution_u = 6
bpy.ops.object.convert(target='MESH')

#Flytter og rotere i 3D space
SetObjLocation(obj, (1,0,0))
SetObjRotation(obj, (0,-90,0))
Editmode(True)
Extrude((0,-1.5,0))
Editmode(False)

#Finder intersectionen mellem de to svg'ere
objSide = bpy.context.scene.objects["Side"] 
BooleanMod(objSide)
bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

#Sletere unødvendige objecter
DeletObj("Side")

#Cutter noget af hoved
bpy.ops.mesh.primitive_cube_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(0.125, 0.125, 0.125))
objCube = bpy.context.scene.objects["Cube"] 
obj = SelectObj("Front")  
BooleanMod(objCube,1)
DeletObj("Cube")

#Laver nyt hovde og joiner dem
bpy.ops.mesh.primitive_cube_add(enter_editmode=False, align='WORLD', location=(0.19, 0.0135, 0.048), scale=(0.070, 0.014, 0.048))
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.join()
SelectObj("Cube")  
bpy.context.active_object.name = 'key'

#Exporter nøgle
Export("key", exportPath)