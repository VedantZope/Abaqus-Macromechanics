import numpy as np

#================ defining the hardening laws =============================

def Swift(c1,c2,c3, truePlasticStrain):
    trueStress = c1 * (c2 + truePlasticStrain) ** c3
    return trueStress

def Voce(c1,c2,c3,truePlasticStrain):
    trueStress = c1 * (c2 - c1) * np.exp(-c3 * truePlasticStrain)
    return trueStress

def SwiftVoce(c1,c2,c3,c4,c5,c6,c7,truePlasticStrain):
    trueStressSwift = Swift(c2,c3,c4,truePlasticStrain)
    trueStressVoce = Voce(c5,c6,c7,truePlasticStrain)
    trueStress = c1 * trueStressSwift + (1-c1) * trueStressVoce
    return trueStress