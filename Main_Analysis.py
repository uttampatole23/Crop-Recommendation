from SubFunctions.Functions import preprocessing, All_Analysis, complete_graph
import torch
import PySimpleGUI as sg
# ensuring device is gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device : ",device)
choose = sg.PopupYesNo("Do You Want Full Execution ?")
if choose == "Yes":
    # preprocessing(1)
    # preprocessing(2)
    All_Analysis()
    ## Graph Functions
    complete_graph(1, 1)
    complete_graph(1, 2)
else:
    ## Graph Functions
    complete_graph(1, 1)
    complete_graph(1, 2)

