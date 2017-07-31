def zoning_classification(zoning,zone):
    if zone == "A":
        if zoning == "A":
            return 1
        else:
            return 0
    elif zone == "C":
        if zoning == "C":
            return 1
        else:
            return 0
    elif zone == "FV":
        if zoning == "FV":
            return 1
        else:
            return 0
    elif zone == "I":
        if zoning == "I":
            return 1
        else:
            return 0
    elif zone == "RH":
        if zoning == "RH":
            return 1
        else:
            return 0
    elif zone == "RL":
        if zoning == "RL":
            return 1
        else:
            return 0
    elif zone == "RP":
        if zoning == "RP":
            return 1
        else:
            return 0
    elif zone == "RM":
        if zoning == "RM":
            return 1
        else:
            return 0


def street (street,type):
    if street == "Grvl":
        if type == "Grvl":
            return 1
        else:
            return 0
    elif street == "Pave":
        if type == "Pave":
            return 1
        else:
            return 0

def Alley (street,type):
    if street == "Grvl":
        if type == "Grvl":
            return 1
        else:
            return 0
    elif street == "Pave":
        if type == "Pave":
            return 1
        else:
            return 0
    elif street == "NA":
        if type == "NA":
            return 1
        else:
            return 0

def LotShape (street,type):
    if street == "Reg":
        if type == "Reg":
            return 1
        else:
            return 0
    elif street == "IR1":
        if type == "IR1":
            return 1
        else:
            return 0
    elif street == "IR2":
        if type == "IR2":
            return 1
        else:
            return 0
    elif street == "IR3":
        if type == "IR3":
            return 1
        else:
            return 0



