import numpy as np

#================ defining the hardening laws =============================

def Swift(c1,c2,c3, truePlasticStrain):
    trueStress = c1 * (c2 + truePlasticStrain) ** c3
    return trueStress

def Voce(c1,c2,c3,truePlasticStrain):
    trueStress = c1 + c2 * (1- np.exp(-c3 * truePlasticStrain))
    return trueStress

def SwiftVoce(c1,c2,c3,c4,c5,c6,c7,truePlasticStrain):
    trueStressSwift = Swift(c2,c3,c4,truePlasticStrain)
    trueStressVoce = Voce(c5,c6,c7,truePlasticStrain)
    trueStress = c1 * trueStressSwift + (1 - c1) * trueStressVoce
    return trueStress

def calculate_flowCurve(parameters, hardeningLaw, truePlasticStrain):
    # We assume that parameters is a dictionary
    if hardeningLaw == "Swift":
        c1, c2, c3 = parameters["c1"], parameters["c2"], parameters["c3"]
        trueStress = Swift(c1, c2, c3, truePlasticStrain)
    elif hardeningLaw == "Voce":
        c1, c2, c3 = parameters["c1"], parameters["c2"], parameters["c3"]
        trueStress = Voce(c1, c2, c3, truePlasticStrain)
    elif hardeningLaw == "SwiftVoce":
        c1, c2, c3, c4, c5, c6, c7 = parameters["c1"], parameters["c2"], parameters["c3"], parameters["c4"], parameters["c5"], parameters["c6"], parameters["c7"]
        trueStress = SwiftVoce(c1, c2, c3, c4, c5, c6, c7, truePlasticStrain)
    return trueStress
        
