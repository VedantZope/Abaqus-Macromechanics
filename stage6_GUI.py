import streamlit as st
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from modules.preprocessing import *
import math 
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import stage0_config
from optimizers.ANN import *
import torch
from optimizers.scaler import *

def main_GUI(info):
    st.title('Crystal plasticity Application')
    st.text("Author: Nguyen Xuan Binh \nInstitution: Aalto University \nCourse: Computational Engineering Project")
    st.markdown("This is an online tool that plots stress-strain curves in the crystal plasticity model and analyzes the fitting parameter optimization. Crystal plasticity studies the plastic deformation of polycrystalline materials")
    st.image("GUI/pictures/CP_illustration.png")


    # Using "with" notation
    with st.sidebar:
        st.header('Please specify your choice')
        #########
        materials = ("RVE_1_40_D", "512grains512")
        material = st.radio("Please select the material", materials)
        #########
        CPLaws = ("DB", "PH")
        CPLaw = st.radio("Please select the crystal plasticity law", CPLaws)
        #########
        curveIndex = st.text_input('Please select the curve index', '1')
        #########
        optimizerNames = ("MOO_NSGA_II", "SOO_GA")
        optimizerName = st.radio("Please select the optimizing strategy", optimizerNames)
        #########
        # All common data 

        loadingsOption = ["linear_uniaxial_RD", 
                        "linear_uniaxial_TD", 
                        "nonlinear_biaxial_RD", 
                        "nonlinear_biaxial_TD",
                        "nonlinear_planestrain_RD",  
                        "nonlinear_planestrain_TD",
                        "nonlinear_uniaxial_RD",   
                        "nonlinear_uniaxial_TD"]

        loadingsName = {"linear_uniaxial_RD": "linear UAT-RD", 
                    "linear_uniaxial_TD": "linear UAT-TD",
                    "nonlinear_biaxial_RD": "nonlinear BAT-RD",
                    "nonlinear_biaxial_TD": "nonlinear BAT-TD",
                    "nonlinear_planestrain_RD": "nonlinear PST-RD",  
                    "nonlinear_planestrain_TD": "nonlinear PST-TD",
                    "nonlinear_uniaxial_RD": "nonlinear UAT-RD",   
                    "nonlinear_uniaxial_TD": "nonlinear UAT-TD"
                    }
        
        loadings = st.multiselect('Select the loadings:', loadingsOption)  
        #########
        roundContinuousDecimals = info["roundContinuousDecimals"]
        #########

        yieldingPoints = info["yieldingPoints"]

    param_info_filtered = info['param_info_filtered']
    paramsFormatted = {
        "PH": {
            "tau0": "τ₀", 
            "a": "a", 
            "gdot0": "γ̇₀", 
            "h0": "h₀", 
            "n": "n", 
            "tausat": "τₛₐₜ",
            "self": "self", 
            "coplanar": "coplanar", 
            "collinear": "collinear", 
            "orthogonal": "orthogonal", 
            "glissile": "glissile", 
            "sessile": "sessile", 
        },
        "DB": {
            "dipole": "dα", 
            "islip": "iₛₗᵢₚ", 
            "omega": "Ω", 
            "p": "p", 
            "q": "q", 
            "tausol": "τₛₒₗ",
            "Qs": "Qs",
            "Qc": "Qc",
            "v0": "v₀",
            "rho_e": "ρe",
            "rho_d": "ρd",   
            "self": "self", 
            "coplanar": "coplanar", 
            "collinear": "collinear", 
            "orthogonal": "orthogonal", 
            "glissile": "glissile", 
            "sessile": "sessile", 
        },
    }

    # parameterRows = {
    #     "PH":
    #         [r"$\tau_0$",
    #         r"$a$", 
    #         r"$h_0$", 
    #         r"$\tau_{sat}$", 
    #         r"$self$", 
    #         r"$coplanar$", 
    #         r"$collinear$", 
    #         r"$orthogonal$", 
    #         r"$glissile$", 
    #         r"$sessile$"],
    #     "DB": 
    #         [r"$d^\alpha$", 
    #         r"$i_{slip}$", 
    #         r"$\Omega$", 
    #         r"$p$", 
    #         r"$q$", 
    #         r"$\tau_{sol}$", 
    #         r"$Q_s$", 
    #         r"$Q_c$", 
    #         r"$v_0$", 
    #         r"$\rho_e$",
    #         r"$self$", 
    #         r"$coplanar$", 
    #         r"$collinear$", 
    #         r"$orthogonal$", 
    #         r"$glissile$", 
    #         r"$sessile$"],
    # }
       
    parameterRows = {
        "PH":
            [r"$\tau_0$",
            r"$a$", 
            r"$h_0$", 
            r"$\tau_{sat}$", 
            r"$self$", 
            r"$coplanar$", 
            r"$collinear$", 
            r"$orthogonal$", 
            r"$glissile$", 
            r"$sessile$"],
        "DB": 
            [
            r"$\tau_{sol}$", 
            r"$d^\alpha$", 
            r"$i_{slip}$", 
            r"$\Omega$", 
            r"$Q_c$", 
            r"$D_0$", 
            r"$self$", 
            r"$coplanar$", 
            r"$collinear$", 
            r"$orthogonal$", 
            r"$glissile$", 
            r"$sessile$"],
    }


    standardColors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    
    # Starting the tabs
    tab1, tab2, tab3, tab4 = st.tabs(["A. Preprocessing", "B. Simulation results", "C. Plotting error", "D. Regressor prediction"])#, "D. Parameter analysis"])

    with tab1:

        st.header('Preprocessing stage')
    
        st.markdown("Please select the curve types that you want to plot")

        targetTrueCheck = st.checkbox("Plot true curve", value=True)
        targetProcessedCheck = st.checkbox("Plot processed curve", value=True)
        yieldCheck = st.checkbox("Plot yielding point", value=True)
        initialSimCheck = st.checkbox("Plot initial simulations", value=False)
        initialSimTypes = ("True curves", "Processed curves")
        initialSimType = st.radio("Please select the initial simulation curve type", initialSimTypes)
        for loading in loadings:
            title = ""
            size = 15
            figure(figsize=(6, 4))
            if initialSimCheck:
                if initialSimType == "True curves":
                    initial_curves = np.load(f'results/{material}/{CPLaw}/universal/{loading}/initial_trueCurves.npy', allow_pickle=True).tolist()
                elif initialSimType == "Processed curves":
                    initial_curves = np.load(f'results/{material}/{CPLaw}/universal/{loading}/initial_processCurves.npy', allow_pickle=True).tolist()
                for curve in initial_curves.values():
                    #st.write(curve["stress"])
                    strain = curve["strain"] 
                    stress = curve["stress"]
                    plt.plot(strain, stress, c='orange', alpha=0.07)
                plt.plot(strain, stress, label = f"Initial simulations x 500",c='orange', alpha=0.3)
                title += f" | Universal initial simulations\n({CPLaw} law)"

            currentPath = f"targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}.xlsx"
            if targetTrueCheck:
                trueCurve = preprocessExperimentalTrue(currentPath, Pa=False)
                trueStrain = trueCurve["strain"]
                trueStress = trueCurve["stress"] 
                plt.plot(trueStrain, trueStress, c='blue', label="Experimental", alpha = 1)
            if targetProcessedCheck:
                processCurve = preprocessExperimentalFitted(currentPath, Pa=False)
                processStrain = processCurve["strain"]
                processStress = processCurve["stress"]
                plt.plot(processStrain, processStress, c='black', label="Swift - Voce fitting")
            title += f"Target curve " 
            
            if yieldCheck:
                yieldingPoint = yieldingPoints[CPLaw][loading]
                plt.axvline(x = yieldingPoint, color = 'black', label = f"Yielding point = {yieldingPoint}", alpha=0.5)
            
            title += f"\n({loading})"
            #plt.title(f"{loading} | {CPLaw} model" , size=size + 4)
            plt.title(f"{loading} | {CPLaw} model" , size=size + 3)
            
            plt.xticks(fontsize=size-2)
            plt.yticks(fontsize=size-2)

            if CPLaw == "PH": 
                plt.xlim([0, 0.27])
                plt.ylim([0, 1750])
                plt.xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
                plt.yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750])
            elif CPLaw == "DB":
                plt.xlim([0, 0.27])
                plt.ylim([0, 1000])
                plt.xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
                plt.yticks([0, 250, 500, 750, 1000])
            # elif convertUnit == "Pa":
            #     if CPLaw == "PH": 
            #         plt.xlim([0, 0.27])
            #         plt.ylim([0, 1750 * 1e6])
            #         plt.xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
            #         plt.yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750]  * 1e6)
            #     elif CPLaw == "DB":
            #         plt.xlim([0, 0.27])
            #         plt.ylim([0, 1000]  * 1e6)
            #         plt.xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
            #         plt.yticks([0, 250, 500, 750, 1000] * 1e6) 
            #plt.xlim([0, 0.27])
            #plt.ylim([0, 500])
            # plt.xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
            # plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450])
            plt.ylabel(f'True stress, MPa', size=size + 1)
            plt.xlabel("True strain, -", size=size + 1)
            legend = plt.legend(loc=2, frameon=False, fontsize=size-2, ncol=1, facecolor='white')
            legend.get_frame().set_linewidth(0.0)
            st.pyplot(plt)


    with tab2:
        st.header('Simulation result stage')

        stageNumbers = ("All stages", "default", "1st stage", "2nd stage", "3rd stage")
        stageNumber = st.radio("Please select the simulation stage", stageNumbers)

        st.subheader("Plot all stages")

        curveTypes = ("True curves", "Processed curves", "Interpolated curves")
        curveType = st.radio("Please select the curve type", curveTypes)
        if curveType == "True curves":
            curveType = "true"
        if curveType == "Processed curves":
            curveType = "process"
        if curveType == "Interpolated curves":
            curveType = "interpolate"
        plotDirection = ("vertical", "horizontal")
        plotDirection = st.radio("Please select the plotting direction", plotDirection, key="plotDirection1")

        if plotDirection == "vertical":
            indexLoading = {
                'tableParam': (0,0),
                "linear_uniaxial_RD": (1,0), 
                "linear_uniaxial_TD": (1,1), 
                "nonlinear_biaxial_RD": (2,0), 
                "nonlinear_biaxial_TD": (2,1),     
                "nonlinear_planestrain_RD": (3,0),     
                "nonlinear_planestrain_TD": (3,1),     
                "nonlinear_uniaxial_RD": (4,0), 
                "nonlinear_uniaxial_TD": (4,1),
            }
        else:
            indexLoading = {
                'tableParam': (0,0),
                "linear_uniaxial_RD": (1,0), 
                "linear_uniaxial_TD": (2,0), 
                "nonlinear_biaxial_RD": (1,1), 
                "nonlinear_biaxial_TD": (2,1),     
                "nonlinear_planestrain_RD": (1,2),     
                "nonlinear_planestrain_TD": (2,2),     
                "nonlinear_uniaxial_RD": (1,3), 
                "nonlinear_uniaxial_TD": (2,3),
            }

        resultPath = f"results/{material}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}"


        if plotDirection == "vertical":
            fig = plt.figure(figsize=(22,38))#, constrained_layout=True) # tight_layout=True, 
            #fig.tight_layout(rect=[0, 0.2, 1, 0.8])
            #fig.subplots_adjust(top=0.85)
            gs = gridspec.GridSpec(5, 2)
            gs.update(top=0.95)
        if plotDirection == "horizontal":
            fig = plt.figure(figsize=(40,21))#, constrained_layout=True) # tight_layout=True, 
            #fig.tight_layout(rect=[0, 0.2, 1, 0.8])
            #fig.subplots_adjust(top=0.85)
            gs = gridspec.GridSpec(3, 4)
            gs.update(top=0.92) # Distance between main title and param table
        
        
        
        if stageNumber == "All stages":

            # Extracting the results

            stage_CurvesList = np.load(f"{resultPath}/common/stage_CurvesList.npy", allow_pickle=True).tolist()
            
            # col2a, col2b = st.columns(2)
            
            # with col2a:
            #     startingIter = st.number_input("Starting iteration", min_value=1, max_value=len(stage_CurvesList), value=len(stage_CurvesList)-1, key="Result1")
            #     #startingIter = st.number_input("Starting iteration", value=23)
            # with col2b:
            #     endingIter = st.number_input("Ending iteration", min_value=1, max_value=len(stage_CurvesList), value=len(stage_CurvesList), key="Result2")
            #     #endingIter = st.number_input("Ending iteration", value=26)

            #col2 = st.columns(1)
            
            #with col2:
            startingIter = st.number_input("Starting iteration", min_value=1, max_value=len(stage_CurvesList), value=len(stage_CurvesList), key="Result1")
            startingIter = int(startingIter)
            #startingIter = st.number_input("Starting iteration", value=23)

            stage_CurvesList = [stage_CurvesList[startingIter - 1]]

            #stage_CurvesList = [stage_CurvesList[67]]
            parameterValues = list([stageCurves["parameters_tuple"] for stageCurves in stage_CurvesList])
            #st.write(stage_CurvesList)
            iterationColumns = []
            paramValues2D = []

            numberOfIterations = len(parameterValues) 
            repeatedCycles = math.ceil(numberOfIterations/10) 
            columnColors = standardColors * repeatedCycles

            iteration = startingIter
            iterationColumns.append(f"Iter {iteration}")
            #for iteration in range(startingIter, endingIter + 1):
            #    iterationColumns.append(f"Iter {iteration}")

            for tupleParams in parameterValues:
                paramValues = []
                dictParams = dict(tupleParams)
                for param in dictParams:
                    paramValues.append(round(dictParams[param], roundContinuousDecimals))
                paramValues2D.append(paramValues)

            # transposing the matrix
            paramValues2D = np.array(paramValues2D).T
            
            size = 28
            #iTable = indexLoading["tableParam"][0]
            #jTable = indexLoading["tableParam"][1]
            axisPlot = fig.add_subplot(gs[0, :])
            axisPlot.axis('tight')
            axisPlot.axis('off')
            table = axisPlot.table(cellText=paramValues2D, 
                                    colLabels=iterationColumns, 
                                    rowLabels=parameterRows[CPLaw], 
                                    loc='upper center', 
                                    cellLoc='center', 
                                    colLoc="center",
                                    rowLoc="center",
                                    colWidths=[len(iterationColumns) * 0.05 for x in iterationColumns],
                                    colColours= columnColors, 
                                    fontsize=40)
            #ax[iTable][jTable].set_title(f"Parameter values", size= 5/4 * size)
            table.auto_set_column_width(col=iterationColumns)
            table.auto_set_font_size(False)
            table.set_fontsize(25)
            table.scale(2.3, 2.3)
            currentStage = ""
            for loading in loadings:
                iteration = startingIter
                pathTarget = f"targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_{curveType}.npy"
                target_Curve = np.load(pathTarget, allow_pickle=True).tolist()
                i = indexLoading[loading][0]
                j = indexLoading[loading][1]
                axisPlot = fig.add_subplot(gs[i, j])
                axisPlot.plot([], [], label=f"{loadingsName[loading]}", color="white")
                axisPlot.plot(target_Curve["strain"], target_Curve["stress"], color = "k", linewidth=3, alpha=1, label=f"Exp")
                
                for stageCurves in stage_CurvesList:
                    axisPlot.plot(stageCurves[curveType][loading]["strain"], stageCurves[curveType][loading]["stress"], linewidth=3, alpha=1, label=f"Sim")
                    iteration += 1
                    
                    axisPlot.set_xlim(right = 0.27)
                    axisPlot.set_ylim(top = 370)
                    axisPlot.tick_params(axis='x', labelsize= size)
                    axisPlot.tick_params(axis='y', labelsize= size)
                    axisPlot.set_ylabel('Stress, MPa', size= size)
                    axisPlot.set_yticks([0, 50, 100, 150, 200, 250, 300, 350])
                    axisPlot.set_xlabel("Strain, -", size= size)
                    
                    legend = axisPlot.legend(loc=4, frameon=False, fontsize= size-2, ncol=1) #, shadow =True, framealpha=1)
                    # plt.grid()
                    legend.get_frame().set_linewidth(0.0)

            if stage_CurvesList[-1]["stage"] == 0:
                currentStage = "yielding"
            else:
                currentStage = "hardening"
            fig.suptitle(f"Parameter calibration result | Curve {CPLaw}{curveIndex} \nOptimizer: {optimizerName} | {currentStage} stage", fontsize=35)
            fig.tight_layout()
            st.pyplot(plt)
        else:
            parameterTypes = {
                "PH": {
                    "default parameters": [],
                    "yieldingParams":  ["tau0"],
                    "linearHardeningParams": ["a", "h0", "tausat"],
                    "nonlinearHardeningParams": ["self", "coplanar", "collinear", "orthogonal", "glissile", "sessile"]
                },
                "DB": {
                    "default parameters": [],
                    "yieldingParams":  ["tausol"],
                    "linearHardeningParams": ["dipmin", "islip", "omega"],
                    "nonlinearHardeningParams": ["self", "coplanar", "collinear", "orthogonal", "glissile", "sessile"]
                }
            }
            if stageNumber == "default":
                stage_curves = np.load(f"{resultPath}/common/default_curve.npy", allow_pickle=True).tolist()
                targetParams = parameterTypes[CPLaw]["default parameters"]
                fig.suptitle(f'Default parameters | Curve {CPLaw}{curveIndex} | Optimizer: {optimizerName}', fontsize=35)
                color = "white"
            if stageNumber == "1st stage":
                stage_curves = np.load(f"{resultPath}/common/stage1_curve.npy", allow_pickle=True).tolist()
                targetParams = parameterTypes[CPLaw]["yieldingParams"]
                fig.suptitle(f'Yielding parameters calibration | Curve {CPLaw}{curveIndex}  | Optimizer: {optimizerName} | Allowed Δᵧ_ₗᵢₙₑₐᵣ = 0.2%', fontsize=35)
                color = "lightgreen"
            if stageNumber == "2nd stage":
                stage_curves = np.load(f"{resultPath}/common/stage2_curve.npy", allow_pickle=True).tolist()
                targetParams = parameterTypes[CPLaw]["linearHardeningParams"]
                fig.suptitle(f'Linear hardening parameters calibration | Curve {CPLaw}{curveIndex} model \n Optimizer: {optimizerName} | Allowed Δₕ_ₗᵢₙₑₐᵣ = 1%', fontsize=35)
                color = "lightskyblue"
            if stageNumber == "3rd stage":
                stage_curves = np.load(f"{resultPath}/common/stage3_curve.npy", allow_pickle=True).tolist()
                targetParams = parameterTypes[CPLaw]["nonlinearHardeningParams"]
                fig.suptitle(f'Nonlinear hardening parameters calibration | Curve {CPLaw}{curveIndex} model \n Optimizer: {optimizerName} | Allowed Δₕ_ₙₒₙₗᵢₙₑₐᵣ = 3%', fontsize=35)
                color = "lightcoral"

            tupleParams = stage_curves["parameters_tuple"]
     
            columnColors = [color]
            rowColors = []


            iterationColumns = ["Parameters"]

            paramValues = []
            paramValues2D = []
            dictParams = dict(tupleParams)
            for param in dictParams:
                if param in targetParams:
                    paramValues.append(f"{round(dictParams[param], roundContinuousDecimals)} (target)")
                    rowColors.append(color)
                else:
                    paramValues.append(round(dictParams[param], roundContinuousDecimals))
                    rowColors.append("white")
            paramValues2D.append(paramValues)

            # st.write(paramValues2D)
            # transposing the matrix
            paramValues2D = np.array(paramValues2D).T

            size = 28

            axisPlot = fig.add_subplot(gs[0, :])
            axisPlot.axis('tight')
            axisPlot.axis('off')
            table = axisPlot.table(cellText=paramValues2D, 
                                    colLabels=iterationColumns, 
                                    rowLabels=parameterRows[CPLaw], 
                                    loc='upper center', 
                                    cellLoc='center', 
                                    colLoc="center",
                                    rowLoc="center",
                                    colWidths=[len(iterationColumns) * 0.08 for x in iterationColumns],
                                    colColours= columnColors, 
                                    fontsize=40)
            #ax[iTable][jTable].set_title(f"Parameter values", size= 5/4 * size)
            table.auto_set_column_width(col=iterationColumns)
            table.auto_set_font_size(False)
            table.set_fontsize(25)
            table.scale(2.3, 2.3)
            currentStage = ""
            for loading in loadings:
                iteration = stage_curves['iteration']
                pathTarget = f"targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_{curveType}.npy"
                target_Curve = np.load(pathTarget, allow_pickle=True).tolist()
                i = indexLoading[loading][0]
                j = indexLoading[loading][1]
                axisPlot = fig.add_subplot(gs[i, j])
                axisPlot.plot(target_Curve["strain"], target_Curve["stress"], color = "k", linewidth=3, alpha=1, label=f"Target")
                #axisPlot.plot(stage_curves[curveType][loading]["strain"], stage_curves[curveType][loading]["stress"], linewidth=3, alpha=1, label=f"Iter {iteration}")
                axisPlot.plot(stage_curves[curveType][loading]["strain"], stage_curves[curveType][loading]["stress"], linewidth=3, alpha=1, label=f"Sim")
                axisPlot.set_xlim(right = 0.27)
                axisPlot.set_ylim(top = 370)
                axisPlot.tick_params(axis='x', labelsize= size)
                axisPlot.tick_params(axis='y', labelsize= size)
                axisPlot.set_ylabel('Stress, MPa', size= size)
                axisPlot.set_yticks([0, 50, 100, 150, 200, 250, 300, 350])
                axisPlot.set_xlabel("Strain, -", size= size)
                axisPlot.set_title(f"{loadingsName[loading]}", size= 5/4 * size)
                legend = axisPlot.legend(loc=4, frameon=False, fontsize= size-2, ncol=2) #, shadow =True, framealpha=1)
                # plt.grid()
                legend.get_frame().set_linewidth(0.0)

            #if stage_CurvesList[-1]["stage"] == 0:
            #    currentStage = "yielding"
            #else:
            #    currentStage = "hardening"
            #fig.suptitle(f"Parameter calibration result | Curve {CPLaw}{curveIndex} \nOptimizer: {optimizerName} | {currentStage} stage", fontsize=35)
            st.pyplot(plt)


    with tab3:
        st.header('Plotting iteration errors')

        # Extracting the results
        
        stage_CurvesList = np.load(f"{resultPath}/common/stage_CurvesList.npy", allow_pickle=True).tolist()

        col2c, col2d = st.columns(2)
        with col2c:
            startingIter2 = st.number_input("Starting iteration", min_value=1, max_value=len(stage_CurvesList), value=1, key="Result3")
        with col2d:
            endingIter2 = st.number_input("Ending iteration", min_value=1, max_value=len(stage_CurvesList), value=len(stage_CurvesList), key="Result4")
        
        stage_CurvesList = stage_CurvesList[startingIter2:endingIter2+1]
        
        #curveTypes = ("Total error", "P", "Interpolated curves")
        lossTypes = ("True hardening loss", "True yielding loss",  "Predicted hardening loss", "Predicted yielding loss",)
        lossType = st.radio("Please select the curve type", lossTypes)
        lossTypeDict = {
            "True yielding loss": "true_yielding_loss",
            "True hardening loss": "true_hardening_loss",
            "Predicted yielding loss": "predicted_yielding_loss",
            "Predicted hardening loss": "predicted_hardening_loss"
        }
        lossType2 = lossTypeDict[lossType]
        #st.write([curve[lossType][loading]["H1"] for curve in stage_CurvesList])
        
        # for loading in loadings:
        #     #st.write(lossType)
        #     iters = list(range(startingIter2, endingIter2)) 
        #     if lossType2 == "true_yielding_loss" or lossType2 == "predicted_yielding_loss":
        #         Y1Loss = [curve[lossType2][loading]["Y1"] for curve in stage_CurvesList]
        #         Y2Loss = [curve[lossType2][loading]["Y2"] for curve in stage_CurvesList]
        #         plt.figure(figsize=(6,4))
        #         plt.plot(iters, Y1Loss, label="H1")
        #         plt.plot(iters, Y2Loss, label="H2")
        #         plt.title(f"{loadingsName[loading]} - {lossType} - {CPLaw}")
        #         plt.xlabel("Iteration")
        #         plt.ylabel("RMSE loss")
        #         plt.legend()
        #         st.pyplot(plt)
        #     else:
        #         H1Loss = [curve[lossType2][loading]["H1"] for curve in stage_CurvesList]
        #         H2Loss = [curve[lossType2][loading]["H2"] for curve in stage_CurvesList]
        #         plt.figure(figsize=(6,4))
        #         plt.plot(iters, H1Loss, label="H1 - Distance loss")
        #         plt.plot(iters, H2Loss, label="H2 - Slop loss")
        #         plt.title(f"{loadingsName[loading]} - {lossType} - {CPLaw}")
        #         plt.xlabel("Iteration")
        #         plt.ylabel("RMSE loss")
        #         plt.legend()
        #         st.pyplot(plt)
        
        
        iters = list(range(startingIter2, endingIter2)) 
        if lossType2 == "true_yielding_loss" or lossType2 == "predicted_yielding_loss":
            plt.figure(figsize=(6,4))
            for loading in loadings:

                Y1Loss = [curve[lossType2][loading]["Y1"] for curve in stage_CurvesList]
                Y2Loss = [curve[lossType2][loading]["Y2"] for curve in stage_CurvesList]
                
                plt.plot(iters, Y1Loss, label="Y1")
                plt.plot(iters, Y2Loss, label="Y2")
                plt.title(f"{loadingsName[loading]} - {lossType} - {CPLaw}")
                plt.xlabel("Iteration")
                plt.ylabel("RMSE loss")
                plt.legend()
                st.pyplot(plt)
        else:
            plt.figure(figsize=(6,4))
            for loading in loadings:
                H1Loss = [curve[lossType2][loading]["H1"] for curve in stage_CurvesList]
                
                plt.plot(iters, H1Loss, label=loadingsName[loading])

                plt.title(f"{loadingsName[loading]} - H1 {lossType} - {CPLaw}")
                plt.xlabel("Iteration")
                plt.ylabel("RMSE loss")
                plt.legend()
            st.pyplot(plt)

            plt.figure(figsize=(6,4))
            for loading in loadings:
                H2Loss = [curve[lossType2][loading]["H2"] for curve in stage_CurvesList]
                
                plt.plot(iters, H2Loss, label=loadingsName[loading])
                plt.title(f"{loadingsName[loading]} - H2 {lossType} - {CPLaw}")
                plt.xlabel("Iteration")
                plt.ylabel("RMSE loss")
                plt.legend()
            st.pyplot(plt)
            #print(lossList)

    with tab4:
        st.header('Regressor prediction')

        plotDirection = ("vertical", "horizontal")
        plotDirection = st.radio("Please select the plotting direction", plotDirection, key="plotDirection2")

        curveType = "process"

        if plotDirection == "vertical":
            indexLoading = {
                'tableParam': (0,0),
                "linear_uniaxial_RD": (1,0), 
                "linear_uniaxial_TD": (1,1), 
                "nonlinear_biaxial_RD": (2,0), 
                "nonlinear_biaxial_TD": (2,1),     
                "nonlinear_planestrain_RD": (3,0),     
                "nonlinear_planestrain_TD": (3,1),     
                "nonlinear_uniaxial_RD": (4,0), 
                "nonlinear_uniaxial_TD": (4,1),
            }
        else:
            indexLoading = {
                'tableParam': (0,0),
                "linear_uniaxial_RD": (1,0), 
                "linear_uniaxial_TD": (2,0), 
                "nonlinear_biaxial_RD": (1,1), 
                "nonlinear_biaxial_TD": (2,1),     
                "nonlinear_planestrain_RD": (1,2),     
                "nonlinear_planestrain_TD": (2,2),     
                "nonlinear_uniaxial_RD": (1,3), 
                "nonlinear_uniaxial_TD": (2,3),
            }

        resultPath = f"results/{material}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}"


        if plotDirection == "vertical":
            fig = plt.figure(figsize=(22,38))#, constrained_layout=True) # tight_layout=True, 
            #fig.tight_layout(rect=[0, 0.2, 1, 0.8])
            #fig.subplots_adjust(top=0.85)
            gs = gridspec.GridSpec(5, 2)
            gs.update(top=0.95)
        if plotDirection == "horizontal":
            fig = plt.figure(figsize=(40,21))#, constrained_layout=True) # tight_layout=True, 
            #fig.tight_layout(rect=[0, 0.2, 1, 0.8])
            #fig.subplots_adjust(top=0.85)
            gs = gridspec.GridSpec(3, 4)
            gs.update(top=0.92) # Distance between main title and param table


        trainingParams = pd.read_excel("configs/ANN_config.xlsx", engine="openpyxl")
        trainingParams = dict(zip(trainingParams.iloc[:,0], trainingParams.iloc[:,1]))
        trainingParams['numberOfHiddenLayers'] = round(trainingParams['numberOfHiddenLayers'])
        trainingParams['step_size'] = round(trainingParams['step_size'])
        trainingParams['tolerance'] = round(trainingParams['tolerance'])
        trainingParams['patience'] = round(trainingParams['patience'])
        trainingParams['startingEpoch'] = round(trainingParams['startingEpoch'])
        trainingParams['batch_size'] = round(trainingParams['batch_size'])
        trainingParams['max_epochs'] = round(trainingParams['max_epochs'])
        hiddenSizes = np.round([trainingParams['hiddenSize'] for _ in range(trainingParams['numberOfHiddenLayers'])])
        trainingParams['hiddenSizes'] = hiddenSizes

        stage_CurvesList = np.load(f"{resultPath}/common/stage_CurvesList.npy", allow_pickle=True).tolist()
        
        iteration = st.number_input("Iteration", min_value=1, max_value=len(stage_CurvesList), value=len(stage_CurvesList), key="Result2")
        iteration = int(iteration)
        #startingIter = st.number_input("Starting iteration", value=23)

        stage_CurvesList = [stage_CurvesList[iteration - 1]]

        #stage_CurvesList = [stage_CurvesList[67]]
        parameterValues = list([stageCurves["parameters_tuple"] for stageCurves in stage_CurvesList])

        
        iterationColumns = []
        paramValues2D = []

        numberOfIterations = len(parameterValues) 
        repeatedCycles = math.ceil(numberOfIterations/10) 
        columnColors = standardColors * repeatedCycles

        iterationColumns.append(f"Iter {iteration}")
        #for iteration in range(startingIter, endingIter + 1):
        #    iterationColumns.append(f"Iter {iteration}")

        for tupleParams in parameterValues:
            paramValues = []
            dictParams = dict(tupleParams)
            for param in dictParams:
                paramValues.append(round(dictParams[param], roundContinuousDecimals))
            paramValues2D.append(paramValues)

        # transposing the matrix
        paramValues2D = np.array(paramValues2D).T

        regressors = {}
        # Define the MLP model using PyTorch
        # Load the saved model's state dictionary
        for loading in loadings:
            checkpoint = torch.load(f'trained_models/{material}/{CPLaw}/model_{loading}.pth')
            paramFeatures = np.load(f'trained_models/{material}/{CPLaw}/paramFeatures_{loading}.npy')
            #print(paramFeatures.shape)
            stressLabels = np.load(f'trained_models/{material}/{CPLaw}/stressLabels_{loading}.npy')
            regressors[loading] = NeuralNetwork(paramFeatures.shape[1], stressLabels.shape[1], trainingParams=trainingParams)
            regressors[loading].load_state_dict(checkpoint['model_state_dict'])

        predictedParams = [list(stageCurves["parameters_dict"].values()) for stageCurves in stage_CurvesList]
        predictedParams = np.array(predictedParams)

        featureMatrixScaling = np.zeros((2, len(list(param_info_filtered.keys()))))
        #st.write(featureMatrixScaling.shape)
        powerList = np.zeros(len(list(param_info_filtered.keys())))
        for index, parameter in enumerate(list(param_info_filtered.keys())):
            featureMatrixScaling[:, index] = np.array([param_info_filtered[parameter]["generalLow"], param_info_filtered[parameter]["generalHigh"]])
            powerList[index] = param_info_filtered[parameter]["power"]
        scaler = CustomScaler(featureMatrixScaling, powerList)
        #st.write(predictedParams)
        #st.write(predictedParams.shape)
        paramFeatures = scaler.transform(predictedParams)
        #st.write(paramFeatures)
        #st.write(paramFeatures.shape)
        # Extracting the results

        size = 28
        #iTable = indexLoading["tableParam"][0]
        #jTable = indexLoading["tableParam"][1]
        axisPlot = fig.add_subplot(gs[0, :])
        axisPlot.axis('tight')
        axisPlot.axis('off')
        table = axisPlot.table(cellText=paramValues2D, 
                                colLabels=iterationColumns, 
                                rowLabels=parameterRows[CPLaw], 
                                loc='upper center', 
                                cellLoc='center', 
                                colLoc="center",
                                rowLoc="center",
                                colWidths=[len(iterationColumns) * 0.05 for x in iterationColumns],
                                colColours= columnColors, 
                                fontsize=40)
        #ax[iTable][jTable].set_title(f"Parameter values", size= 5/4 * size)
        table.auto_set_column_width(col=iterationColumns)
        table.auto_set_font_size(False)
        table.set_fontsize(25)
        table.scale(2.3, 2.3)
        currentStage = ""
        for loading in loadings:
            iteration = startingIter
            pathTarget = f"targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_interpolate.npy"
            target_Curve = np.load(pathTarget, allow_pickle=True).tolist()
            interpolatedStrain = target_Curve["strain"]
            predictedStress = regressors[loading].predictOneDimension(paramFeatures).reshape(-1)
            i = indexLoading[loading][0]
            j = indexLoading[loading][1]
            axisPlot = fig.add_subplot(gs[i, j])
            axisPlot.plot([], [], label=f"{loadingsName[loading]}", color="white")
            axisPlot.plot(interpolatedStrain, predictedStress, color = "k", linewidth=3, alpha=1, label=f"Prediction")
            
            for stageCurves in stage_CurvesList:
                axisPlot.plot(stageCurves[curveType][loading]["strain"], stageCurves[curveType][loading]["stress"], linewidth=3, alpha=1, label=f"Sim")
                iteration += 1
                
                axisPlot.set_xlim(right = 0.27)
                #axisPlot.set_ylim(top = 370)
                axisPlot.tick_params(axis='x', labelsize= size)
                axisPlot.tick_params(axis='y', labelsize= size)
                axisPlot.set_ylabel('Stress, MPa', size= size)
                axisPlot.set_yticks([0, 50, 100, 150, 200, 250, 300, 350])
                axisPlot.set_xlabel("Strain, -", size= size)
                
                legend = axisPlot.legend(loc=4, frameon=False, fontsize= size-2, ncol=1) #, shadow =True, framealpha=1)
                # plt.grid()
                legend.get_frame().set_linewidth(0.0)

        if stage_CurvesList[-1]["stage"] == 0:
            currentStage = "yielding"
        else:
            currentStage = "hardening"
        fig.suptitle(f"Regressor prediction | Curve {CPLaw}{curveIndex} \nOptimizer: {optimizerName} | {currentStage} stage", fontsize=35)
        fig.tight_layout()
        st.pyplot(plt)  
        #print(iters)
        

# # Run the program with the following command:
# # python -m streamlit run stage6_GUI.py
# # python3 -m streamlit run stage6_GUI.py

if __name__  == "__main__":
    info = stage0_config.main_config()
    main_GUI(info)
