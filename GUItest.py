from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import tkinter
import vtkplotlib as vpl
from stl.mesh import Mesh
from PIL import ImageTk,Image
from main import generate_key
import os
import numpy as np
import cv2 as cv

root = Tk()
root.geometry("400x800")
frm = Frame(root)
frm.pack(side=TOP)
frmM = Frame(root)
frmM.pack(side=TOP)
frmBot = Frame(root)
frmBot.pack(side= TOP)
img_name = ""

def view3D():
    mesh = Mesh.from_file("out/key.stl")

    mesh = vpl.mesh_plot(mesh,color=(255,205,0))
    
    mesh.label = ""
    
    vpl.show()

def openfn():
    filename = filedialog.askopenfilename(title='open', filetypes = (("Image files","*.png*"),("all files","*.*")))
    inputLabel.config(text = "Choosen image: " + filename.split('/')[-1])
    return filename

def open_img():
    x = openfn()
    img = Image.open(x)
    global img_name
    img_name = x
    
    scale = img.size[0]//200
    
    temp = (int(img.size[0]//scale),int(img.size[1]//scale))
    img = img.resize(temp, Image.LANCZOS)
    img = ImageTk.PhotoImage(img)   
    panel.configure(image=img)
    panel.image = img


def show_img(fileName):
    # img = Image.open(image)
    # temp = [int(img.size[0]/4),int(img.size[1]/4)]
    # img = img.resize((temp[0],temp[1]), Image.LANCZOS)
    # img = ImageTk.PhotoImage(img)
    
    img= Image.open(fileName)
    
    temp = (int(img.size[0]/8),int(img.size[1]/8))
    
    
    img = img.resize(temp, Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    
    label = Label(root, text= "Key view" )
    panel = Label(root, image=img)
    
    panel.image = img
    panel.label = label
    label.pack(side=TOP)
    panel.pack(side=TOP)

def showKeySP():
    os.system(f'blender --background --python BlenderScript.py --imgName {img_name} --bladeHeight {profilHin.get()} --keyThick {profilWin.get()} --headHeight {sideHin.get()} --keyLength {sideWin.get()} --kernelSize {kernelSizeSettingInput.get()} --medianBlur {medianBlurSettingInput.get()} --noiseRemoval {noiseRemovalSettingInput.get()} --lineThresh {lineThreshSettingInput.get()} --scaleFactor {downFactorSettingInput.get()} --erosionSize {erosionSizeSettingInput.get()}')

def getMeasurements(key):
    #indlæs målene for "key" og gem dem i et array [ProfilHøjde, profilBrede, sideHøjde, sideBrede].
    updateMeasurements(data[key])

def updateMeasurements(value):

    profilHin.delete(0,100)
    profilHin.insert(0,value[0])
    profilHin.pack(side=LEFT)

    profilWin.delete(0,100)
    profilWin.insert(0,value[1])
    profilWin.pack(side=LEFT)

    sideHin.delete(0,100)
    sideHin.insert(0,value[2])
    sideHin.pack(side=LEFT)

    sideWin.delete(0,100)
    sideWin.insert(0,value[3])
    sideWin.pack(side=LEFT)

# Dropdown menu options
options = []

#Read data from data.txt
data = {}
datafile = open('data.txt', 'r')
contents = datafile.read()
print(contents)
temp = contents.split('\n')
for f in temp:
    temp2 = f.split(',')
    data[temp2[0]] = [temp2[1],temp2[2],temp2[3],temp2[4]]
    options.append(temp2[0])


inputImage = ""

# datatype of menu text
clicked = StringVar()
  
Label(frm, text="Key Type:").pack(side= LEFT)
# Create Dropdown menu
drop = OptionMenu( frm , clicked , *options, command=getMeasurements)
drop.pack(side= LEFT)
  
profilLabel = Label(frmM, text = "Profile measurements: ").pack(side=TOP)

pHeight = Frame(frmM)
pHeight.pack(side=TOP)
pWidth = Frame(frmM)
pWidth.pack(side=TOP)

profilLabel = Label(frmM, text = "Side measurements: ").pack(side=TOP)

sHeight = Frame(frmM)
sHeight.pack(side=TOP)
sWidth = Frame(frmM)
sWidth.pack(side=TOP)

profilHLabel = Label(pHeight, text = "Blade height: ").pack(side=LEFT)
profilHin = Entry(pHeight, width= 8)

profilWLabel = Label(pWidth, text = "Key thickness: ").pack(side=LEFT)
profilWin = Entry(pWidth, width= 8)

sideHLabel = Label(sHeight, text = "Head height: ").pack(side=LEFT)
sideHin = Entry(sHeight, width= 8)

sideWLabel = Label(sWidth, text = "Key lenght:  ").pack(side=LEFT)
sideWin = Entry(sWidth, width= 8)



updateMeasurements(["","","",""])

spacer = Frame(frmM)
spacer.pack(side=TOP)
Label(spacer, text="").pack(side= TOP)
Label(spacer, text="Settings").pack(side= TOP)

kernelSizeSetting = Frame(frmM)
kernelSizeSetting.pack(side=TOP)

medianBlurSetting = Frame(frmM)
medianBlurSetting.pack(side=TOP)

noiseRemovalSetting = Frame(frmM)
noiseRemovalSetting.pack(side=TOP)

lineThreshSetting = Frame(frmM)
lineThreshSetting.pack(side=TOP)

erosionSizeSetting = Frame(frmM)
erosionSizeSetting.pack(side=TOP)

downFactorSetting = Frame(frmM)
downFactorSetting.pack(side=TOP)

kernelSizeSettingLable = Label(kernelSizeSetting, text = "Kernel Size: ").pack(side=LEFT)
kernelSizeSettingInput = Entry(kernelSizeSetting, width= 8)
kernelSizeSettingInput.insert(0,"7")
kernelSizeSettingInput.pack(side=LEFT)

medianBlurSettingLable = Label(medianBlurSetting, text = "Median Blur Size: ").pack(side=LEFT)
medianBlurSettingInput = Entry(medianBlurSetting, width= 8)
medianBlurSettingInput.insert(0,"9")
medianBlurSettingInput.pack(side=LEFT)

noiseRemovalSettingLable = Label(noiseRemovalSetting, text = "Noise Removal: ").pack(side=LEFT)
noiseRemovalSettingInput = Entry(noiseRemovalSetting, width= 8)
noiseRemovalSettingInput.insert(0,"2")
noiseRemovalSettingInput.pack(side=LEFT)

lineThreshSettingLable = Label(lineThreshSetting, text = "Line Detection Threshold: ").pack(side=LEFT)
lineThreshSettingInput = Entry(lineThreshSetting, width= 8)
lineThreshSettingInput.insert(0,"1000")
lineThreshSettingInput.pack(side=LEFT)

erosionSizeSettingLable = Label(erosionSizeSetting, text = "Erosion Size: ").pack(side=LEFT)
erosionSizeSettingInput = Entry(erosionSizeSetting, width= 8)
erosionSizeSettingInput.insert(0,"-1")
erosionSizeSettingInput.pack(side=LEFT)

downFactorSettingLable = Label(downFactorSetting, text = "Downscaling Factor: ").pack(side=LEFT)
downFactorSettingInput = Entry(downFactorSetting, width= 8)
downFactorSettingInput.insert(0,"2")
downFactorSettingInput.pack(side=LEFT)

# Just af spacer to make the GUI look nice.
label = Label( frmBot , text = " " )
label.pack(side= TOP)

panel = Label(root, image=None)
panel.pack(side=TOP) 
# Create button, it will change label text
button_explore = Button(frmBot, text='Open image', command=open_img, width= 15, height= 1).pack(side=TOP)
button = Button( frmBot , text = "Make key" , command=showKeySP, width= 15, height= 1 ).pack(side=TOP)
button3D = Button( frmBot , text = "View 3D-key" , command = view3D, width= 15, height= 1 ).pack(side=TOP)

inputImageLable = Label(frmBot)
inputImageLable.pack(side=TOP)

inputLabel = Label(root, text = "Choosen image: ")
inputLabel.pack(side=TOP)

root.wm_title("Key Generator")
# Execute tkinter
root.mainloop()



