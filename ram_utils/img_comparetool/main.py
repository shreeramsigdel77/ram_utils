import PySimpleGUI as sg
import os
from PySimpleGUI.PySimpleGUI import FolderBrowse, HorizontalSeparator
from natsort import natsorted
sg.theme('DarkAmber')   # Add a touch of color

def_value  = 0


def global_counter_both(val):
    global def_value
    if val == "next":
        def_value  += 1
    elif val == "prev":
       def_value  -= 1

img_folder1 = [
    [
        # sg.Text("Browse"),
        sg.In(size=(50,2),enable_events=True,key="-FOLDER1-"),
        sg.FolderBrowse(),
    ]
] 

img_folder2 = [
    [
        # sg.Text("Browse"),
        sg.In(size=(50,2),enable_events=True,key="-FOLDER2-"),
        sg.FolderBrowse(),
    ]
] 


preview_img1 = [
    [
      sg.Image(key="-IMAGE1-")  
    ]
]


preview_img2 = [
    [
      sg.Image(key="-IMAGE2-")  
    ]
]


# All the stuff inside your window.
layout = [  
    [
        sg.Column(img_folder1),
        sg.VerticalSeparator(),
        sg.Column(img_folder2),

        
    ],
    [
        sg.Column(preview_img1),
        sg.VerticalSeparator(),
        sg.Column(preview_img2),
    ],
    
    [
        sg.Button('Previous' ,enable_events=True),
        sg.Button('Next',enable_events=True)
    ],
    
    
    
    
    [
        sg.Button('Quit')
    ]
]
















# Create the Window
window = sg.Window('Preview Images', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Quit': # if user closes window or clicks cancel
        break
    # pick filenames from first
    elif event == "-FOLDER1-" :
        folder_pth1 = values["-FOLDER1-"]
        try:
            img_list1 = os.listdir(folder_pth1)
        except:
            img_list1 = []
        
        fnames1 = []
        for f in img_list1:
            if f.split('.')[-1] in ["png","jpg"]:
                fnames1.append(os.path.join(folder_pth1,f))
        fnames1 = natsorted(fnames1)
        window["-IMAGE1-"].update(fnames1[def_value])

    #pick filenames from second
    elif event == "-FOLDER2-" :
        folder_pth2 = values["-FOLDER2-"]
        try:
            img_list2 = os.listdir(folder_pth2)
        except:
            img_list2 = []
        
        fnames2 = []
        for f in img_list2:
            if f.split('.')[-1] in ["png","jpg"]:
                fnames2.append(os.path.join(folder_pth2,f))
        fnames2 = natsorted(fnames2)
        window["-IMAGE2-"].update(fnames2[def_value])
        
    elif event == "Previous":
        global_counter_both("prev")
        if def_value>=0:
            # print(def_value)
            window["-IMAGE1-"].update(fnames1[def_value])
            window["-IMAGE2-"].update(fnames2[def_value])
        elif def_value <0:
            def_value = (len(fnames1)-1)
            window["-IMAGE1-"].update(fnames1[def_value])
            window["-IMAGE2-"].update(fnames2[def_value])
    elif event == "Next":
        global_counter_both("next")
        if def_value<len(fnames1):
            window["-IMAGE1-"].update(fnames1[def_value])
            window["-IMAGE2-"].update(fnames2[def_value])
        elif def_value == len(fnames1):
                def_value =0
                window["-IMAGE1-"].update(fnames1[def_value])
                window["-IMAGE2-"].update(fnames2[def_value])

window.close()
