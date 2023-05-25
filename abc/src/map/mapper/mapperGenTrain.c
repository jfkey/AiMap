/**CFile****************************************************************

  FileName    [abcMap.c]

  SystemName  [ABC: Logic synthesis and verification system.]

  PackageName [Network and node package.]

  Synopsis    [Interface with the SC mapping package.]

  Author      [Alan Mishchenko]

  Affiliation [UC Berkeley]

  Date        [Ver. 1.0. Started - June 20, 2005.]

  Revision    [$Id: abcMap.c,v 1.00 2005/06/20 00:00:00 alanmi Exp $]

***********************************************************************/

#include "base/abc/abc.h"
#include "base/main/main.h"
#include "map/mio/mio.h"
#include "map/mapper/mapper.h"
#include "map/scl/sclLib.h"
#include "map/scl/sclSize.h"
#include "base/abci/abcMap.c"
#include "map/mapper/mapperInt.h"


ABC_NAMESPACE_IMPL_START


////////////////////////////////////////////////////////////////////////
///                        DECLARATIONS                              ///
////////////////////////////////////////////////////////////////////////

//extern Map_Man_t *  Abc_NtkToMap( Abc_Ntk_t * pNtk, double DelayTarget, int fRecovery, float * pSwitching, int fVerbose );
//extern Abc_Ntk_t *  Abc_NtkFromMap( Map_Man_t * pMan, Abc_Ntk_t * pNtk, int fUseBuffs );
//static Abc_Obj_t *  Abc_NodeFromMap_rec( Abc_Ntk_t * pNtkNew, Map_Node_t * pNodeMap, int fPhase );
//static Abc_Obj_t *  Abc_NodeFromMapPhase_rec( Abc_Ntk_t * pNtkNew, Map_Node_t * pNodeMap, int fPhase );


////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////

/**Function*************************************************************

  Synopsis    [Interface with the mapping package.]

  Description []

  SideEffects []

  SeeAlso     []

***********************************************************************/
Abc_Ntk_t * Abc_genTrain(Abc_Frame_t * pAbc, Abc_Ntk_t * pNtk,  char* nodeFileStr, char* cutFileStr, char* cellFileStr, char* labelFileStr, int itera)
{
    // default config
    double DelayTarget =-1;
    double AreaMulti   = 0;
    double DelayMulti  = 0;
    float LogFan = 0;
    float Slew = 0; // choose based on the library
    float Gain = 250;
    int nGatesMin = 0;
    int fAreaOnly   = 0;
    int fRecovery   = 1;        // 1 -> do area recovery
    int fSweep      = 0;
    int fSwitching  = 0;
    int fSkipFanout = 0;
    int fUseProfile = 0;
    int fUseBuffs   = 0;
    int fVerbose    = 0;

    static int fUseMulti = 0;
    int fShowSwitching = 1;
    Abc_Ntk_t * pNtkRes;
    Map_Man_t * pMan;
    Vec_Int_t * vSwitching = NULL;
    float * pSwitching = NULL;
//    abctime clk, clkTotal = Abc_Clock();
    Mio_Library_t * pLib = (Mio_Library_t *)Abc_FrameReadLibGen();

    float minDelay = MAP_FLOAT_LARGE;
    char* preDelayStr = "";
    // init Map_Train_t parameters.
    Map_Train_t * pPara = (Map_Train_t *) malloc(sizeof(Map_Train_t));
    pPara->isTrain  = 1;
    pPara->slew     = 0;
    pPara->gain     = 0;
    pPara->alpha    = 0;
    pPara->gamma    = 0;
    pPara->tau      = 0;
    pPara->beta     = 0;
    pPara->constF   = 0;

    float alphaArr[] = {0.6, 0.8, 1.0, 1.2, 1.4};           // 5
    float gainArr[] = {100, 150, 200, 250, 300, 350};       // 6
    float slewArr[] = {1, 10, 20, 50, 100, 150, 200};       // 7

    char * emptyStr = "";
    Abc_Ntk_t *  pNtkInput =Abc_NtkDup(pNtk);


    // default result of ABC
    pNtkRes = Abc_NtkMap( pNtk, DelayTarget, AreaMulti, DelayMulti, LogFan, Slew, Gain, nGatesMin, fRecovery, fSwitching,
                          fSkipFanout, fUseProfile, fUseBuffs, fVerbose, emptyStr, emptyStr, emptyStr, emptyStr, NULL);
    Abc_FrameReplaceCurrentNetwork( pAbc, pNtkRes );

    Abc_SclTimePerform( (SC_Lib *)pAbc->pLibScl, Abc_FrameReadNtk(pAbc),0, 0, 0, 0, 0, emptyStr);
    float dfDelay = Abc_NtkReportDelay(Abc_FrameReadNtk(pAbc));
    float dfArea = Abc_NtkGetMappedArea(Abc_FrameReadNtk(pAbc));
    printf("default delay:%5.2f, default area:%5.2f\n", dfDelay, dfArea);

    // strategy 1. Different cell delay computation parameters
    float bestgain = 0.0; int besti = 0, bestj = 0, bestk = 0;
    for (int i = 0; i < 5; i ++ ) {             // alphaArr  5
        for (int j = 0; j < 6; j++) {           // gainArr   6
            for (int k = 0; k < 7; k++) {       // slewArr   7
                pPara->alpha = alphaArr[i];
                pPara->gain = gainArr[j];
                pPara->slew = slewArr[k];
//                Abc_FrameReplaceCurrentNetwork( pAbc, pNtk);
                // technology mapping and STA
                pNtkRes = Abc_NtkMap(pNtkInput, DelayTarget, AreaMulti, DelayMulti, LogFan, Slew, Gain, nGatesMin,
                                     fRecovery, fSwitching,
                                     fSkipFanout, fUseProfile, fUseBuffs, fVerbose, emptyStr, emptyStr, emptyStr,
                                     emptyStr, pPara);
                Abc_FrameReplaceCurrentNetwork(pAbc, pNtkRes);
                Abc_SclTimePerform((SC_Lib *) pAbc->pLibScl, Abc_FrameReadNtk(pAbc), 0, 0, 0, 0, 0, emptyStr);
                float curDelay = Abc_NtkReportDelay(Abc_FrameReadNtk(pAbc));
                float curArea = Abc_NtkGetMappedArea(Abc_FrameReadNtk(pAbc));
                float gainD = (dfDelay - curDelay) / dfDelay;
                float gainA = (dfArea - curArea) / dfArea;
                if (gainD + gainA > 0.01 && gainD + gainA > bestgain) {
                    bestgain = gainA + gainD;
                    besti = i;
                    bestj = j;
                    bestk = k;
                }
            }
        }
    }
    if (bestgain > 0) {
        printf("alpha:%3.2f, gain:%3.2f, slew:%3.2f \n", alphaArr[besti], gainArr[bestj], slewArr[bestk]);
        pPara->alpha = alphaArr[besti];
        pPara->gain = gainArr[bestj];
        pPara->slew = slewArr[bestk];
//                Abc_FrameReplaceCurrentNetwork( pAbc, pNtk);
        // technology mapping and STA
        pNtkRes = Abc_NtkMap(pNtkInput, DelayTarget, AreaMulti, DelayMulti, LogFan, Slew, Gain, nGatesMin,
                             fRecovery, fSwitching,
                             fSkipFanout, fUseProfile, fUseBuffs, fVerbose, emptyStr, emptyStr, emptyStr,
                             emptyStr, pPara);
        Abc_FrameReplaceCurrentNetwork(pAbc, pNtkRes);
        Abc_SclTimePerform((SC_Lib *) pAbc->pLibScl, Abc_FrameReadNtk(pAbc), 0, 0, 0, 0, 0, emptyStr);
        dfDelay = Abc_NtkReportDelay(Abc_FrameReadNtk(pAbc));
        dfArea = Abc_NtkGetMappedArea(Abc_FrameReadNtk(pAbc));
    }

    // to remove
//    pPara->alpha = 0.8, pPara->gain = 150, pPara->slew = 100;
//    pNtkRes = Abc_NtkMap( pNtk, DelayTarget, AreaMulti, DelayMulti, LogFan, Slew, Gain, nGatesMin, fRecovery, fSwitching,
//                          fSkipFanout, fUseProfile, fUseBuffs, fVerbose, emptyStr, emptyStr, emptyStr, emptyStr, pPara);
//    Abc_FrameReplaceCurrentNetwork( pAbc, pNtkRes );
//
//    Abc_SclTimePerform( (SC_Lib *)pAbc->pLibScl, Abc_FrameReadNtk(pAbc),0, 0, 0, 0, 0, emptyStr);
//    float dfDelay = Abc_NtkReportDelay(Abc_FrameReadNtk(pAbc));
//    float dfArea = Abc_NtkGetMappedArea(Abc_FrameReadNtk(pAbc));
    printf("after strategy 1  delay:%5.2f, after strategy 1 area:%5.2f\n", dfDelay, dfArea);
    bestgain = 0;

    // strategy 2. Different delay computation.
    float betaArr[]   = {0.9, 1.15, 1.25, 1.35, 1.45};
    float gammaArr[]  = {0.9, 1.0, 1.1, 1.2, 1.3};
    float tauArr[]    = {0.1, 0.2, 0.3, 0.4, 0.5};
    float constFArr[] = {0.7, 0.8, 0.9, 1.0, 1.1};
    for (int i = 0; i < 5; i ++ ) {             // betaArr    5
        for (int j = 0; j < 5; j++) {           // gammaArr   5
            for (int k = 0; k < 5; k++) {       // tauArr     5
                for (int l = 0; l < 5; l ++){   // constFArr  5
                    pPara->beta  = betaArr[i];
                    pPara->gamma = gammaArr[j];
                    pPara->tau   = slewArr[k];
                    pPara->constF= constFArr[l];
    //                Abc_FrameReplaceCurrentNetwork( pAbc, pNtk);
                    // technology mapping and STA
                    pNtkRes = Abc_NtkMap(pNtkInput, DelayTarget, AreaMulti, DelayMulti, LogFan, Slew, Gain, nGatesMin,
                                         fRecovery, fSwitching,
                                         fSkipFanout, fUseProfile, fUseBuffs, fVerbose, emptyStr, emptyStr, emptyStr,
                                         emptyStr, pPara);
                    Abc_FrameReplaceCurrentNetwork(pAbc, pNtkRes);
                    Abc_SclTimePerform((SC_Lib *) pAbc->pLibScl, Abc_FrameReadNtk(pAbc), 0, 0, 0, 0, 0, emptyStr);
                    float curDelay = Abc_NtkReportDelay(Abc_FrameReadNtk(pAbc));
                    float curArea = Abc_NtkGetMappedArea(Abc_FrameReadNtk(pAbc));
                    float gainD = (dfDelay - curDelay) / dfDelay;
                    float gainA = (dfArea - curArea) / dfArea;
                    if (gainD + gainA > 0.01 && gainD + gainA > bestgain) {
                        printf("--------------------------------------%3.2f, %3.2f\n", curDelay, curArea);
                        bestgain = gainA + gainD;
                    }
                }
            }
        }
    }




//    pNtkRes = Abc_NtkMap( pNtk, DelayTarget, AreaMulti, DelayMulti, LogFan, Slew, Gain, nGatesMin, fRecovery, fSwitching,
//                          fSkipFanout, fUseProfile, fUseBuffs, fVerbose, nodeFileStr, cutFileStr, cellFileStr, preDelayStr, NULL);
//    Abc_FrameReplaceCurrentNetwork( pAbc, pNtkRes );
//
//    Abc_SclTimePerform( (SC_Lib *)pAbc->pLibScl, Abc_FrameReadNtk(pAbc),0, 0, 0, 0, 0, labelFileStr);


//
//    for (int i = 0; i < itera; i ++ ) {
//        pNtkRes = Abc_NtkMap( pNtk, DelayTarget, AreaMulti, DelayMulti, LogFan, Slew, Gain, nGatesMin, fRecovery, fSwitching,
//                              fSkipFanout, fUseProfile, fUseBuffs, fVerbose, nodeFileStr, cutFileStr, cellFileStr, preDelayStr, isTrain);
//        Abc_FrameReplaceCurrentNetwork( pAbc, pNtkRes );
//
//        Abc_SclTimePerform( (SC_Lib *)pAbc->pLibScl, Abc_FrameReadNtk(pAbc),0, 0, 0, 0, 0, labelFileStr);
//        float reportDelay = Abc_NtkReportDelay(Abc_FrameReadNtk(pAbc));
//        if (reportDelay < minDelay)
//            minDelay = reportDelay;
//         printf("%.3f\n", minDelay);
//    }
    // write into trainFileStr.




}

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////


ABC_NAMESPACE_IMPL_END