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
    pPara->randCutCount = 250;          // Default 250
    pPara->cutCoutItera = 0;



    int paraSize = 5;
    int blockSize = 15;
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
    float initDelay = dfDelay, initArea = dfArea;
    float curBestDelay = dfDelay, curBestArea = dfArea;
    printf("Before tuning: (Area:%5.2f, Delay:%5.2f)\n", dfArea, dfDelay);

    // strategy 1. Different cell delay computation parameters
    float bestgain1 = 0.0, bestgain2 = 0.0, bestgain3 = 0.0; int besti = 0, bestj = 0, bestk = 0, bestl = 0;

    for (int i = 0; i < paraSize; i ++ ) {               // alphaArr  5
        for (int j = 0; j < paraSize+1; j++) {           // gainArr   6
            for (int k = 0; k < paraSize+2; k++) {       // slewArr   7
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
                if (gainD + gainA > 0.01 && gainD + gainA > bestgain1) {
                    curBestDelay = curDelay;
                    curBestArea = curArea;
                    bestgain1 = gainA + gainD;
                    besti = i;
                    bestj = j;
                    bestk = k;
                }
            }
        }
    }
    if (bestgain1 > 0) {
        printf("Effective After Strategy 1 Gain:%3.2f, (Area:%5.2f, Delay:%5.2f) with  (alpha:%3.2f, gain:%3.2f, slew:%3.2f) \n", bestgain1,curBestArea, curBestDelay,  alphaArr[besti], gainArr[bestj], slewArr[bestk]);
        pPara->alpha = alphaArr[besti];
        pPara->gain = gainArr[bestj];
        pPara->slew = slewArr[bestk];
    } else {
        pPara->alpha = 1;
        pPara->gain = 250;
        pPara->slew = 20;
        printf("Invalid After Strategy 1  Gain:%3.2f,  (Area:%5.2f, Delay:%5.2f) with (alpha:%3.2f, gain:%3.2f, slew:%3.2f) \n", bestgain1, curBestArea, curBestDelay,  pPara->alpha, pPara->gain, pPara->slew);
    }

    // strategy 2. Different delay computation .
    float betaArr[]   = {1.25, 1.35, 1.45, 1.5, 1.6};
    float gammaArr[]  = {0.9, 1.0, 1.1, 1.2, 1.3};
    float tauArr[]    = {0.1, 0.2, 0.3, 0.4, 0.5};
    float constFArr[] = {0.7, 0.8, 0.9, 1.0, 1.1};

    for (int i = 0; i < paraSize; i ++ ) {              // betaArr    5
        for (int j = 0; j < paraSize; j++) {            // gammaArr   5
            for (int k = 0; k < paraSize ; k++) {       // tauArr     5
                for (int l = 0; l < paraSize; l ++){    // constFArr  5
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
                    if (gainD + gainA > 0.01 && gainD + gainA > bestgain1) {
                        bestgain2 = gainA + gainD;
                        besti = i; bestj = j; bestk = k; bestl = l;
                        curBestDelay = curDelay;
                        curBestArea = curArea;
                    }
                }
            }
        }
    }
    if (bestgain2 > 0) {
        pPara->beta  = betaArr[besti];
        pPara->gamma = gammaArr[bestj];
        pPara->tau   = slewArr[bestk];
        pPara->constF= constFArr[bestl];
        printf("Effective After Strategy 2 Gain:%3.2f, (Area:%5.2f, Delay:%5.2f) with  (beta:%3.2f, gamma:%3.2f, tau:%3.2f, constF:%3.2f) \n", bestgain2, curBestArea, curBestDelay,  pPara->beta, pPara->gamma, pPara->tau, pPara->constF);
    } else {
        pPara->beta  = 0;
        pPara->gamma = 0;
        pPara->tau   = 0;
        pPara->constF= 0;
        printf("Invalid  After Strategy 2 Gain:%3.2f, (Area:%5.2f, Delay:%5.2f) with  (beta:%3.2f, gamma:%3.2f, tau:%3.2f, constF:%3.2f) \n", bestgain2, curBestArea, curBestDelay,  pPara->beta, pPara->gamma, pPara->tau, pPara->constF);
    }

//    // strategy 3. sample cuts.
    int  randTypeArr[] = {1, 2, 3, 4, 5};
    for (int i = 0; i < paraSize; i ++ ) {
        for (int j = 0; j < blockSize; j= j+1 ) {
//            pPara->randCutCount = cutSizeArr[j];
            pPara->randType = i;
            pPara->randCutCount = j+1;
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
            if (gainD + gainA > 0.01 && gainD + gainA > bestgain2 && gainD + gainA > bestgain1 ) {
                bestgain3 = gainA + gainD;
                besti = i; bestj = j+1;
                curBestDelay = curDelay;
                curBestArea = curArea;
            }
        }
    }
    if (bestgain3 > 0) {
        pPara->randType = randTypeArr[besti];
        pPara->randCutCount = bestj;
        printf("Effective After Strategy 3 Gain:%3.2f, (Area:%5.2f, Delay:%5.2f) with  (randType:%d, blockSize:%d) \n", bestgain3, curBestArea, curBestDelay, pPara->randType, pPara->randCutCount);
    } else {
        pPara->randType = 1;
        pPara->randCutCount = 250;
        printf("Invalid After Strategy 3 Gain:%3.2f, (Area:%5.2f, Delay:%5.2f) with  (randType:%d, blockSize:%d) \n", bestgain3, curBestArea, curBestDelay,  pPara->randType, pPara->randCutCount);
    }
    //// adder
//    pPara->alpha = 1.4;
//    pPara->gain = 250;
//    pPara->slew = 10;
//    pPara->beta  = 1.6;
//    pPara->gamma = 1.1;
//    pPara->tau   = 1.0;
//    pPara->constF= 0.7;
//    pPara->randType = 5;
//    pPara->randCutCount = 4;
//
    //// max
//    pPara->alpha = 1.4;
//    pPara->gain = 350;
//    pPara->slew = 50;
//    pPara->beta  = 1.5;
//    pPara->gamma = 1.1;
//    pPara->tau   = 10;
//    pPara->constF= 1.10;
//    pPara->randType = 5;
//    pPara->randCutCount = 11;

    //// sin
//    pPara->alpha = 1.2;
//    pPara->gain = 350;
//    pPara->slew = 10;
//    pPara->beta  = 0;
//    pPara->gamma = 0;
//    pPara->tau   = 0;
//    pPara->constF= 0;
//    pPara->randType = 1;
//    pPara->randCutCount = 250;

    //// bar
//    pPara->alpha = 0.8;
//    pPara->gain = 150;
//    pPara->slew = 100;
//    pPara->beta  = 0;
//    pPara->gamma = 0;
//    pPara->tau   = 0;
//    pPara->constF= 0;
//    pPara->randType = 1;
//    pPara->randCutCount = 250;

    //// router ****
//    pPara->alpha = 1.2;
//    pPara->gain = 150;
//    pPara->slew = 1;
//    pPara->beta  = 1.6;
//    pPara->gamma = 0.9;
//    pPara->tau   = 1.0;
//    pPara->constF= 0.9;
//    pPara->randType = 3;
//    pPara->randCutCount = 16;

    //// i2c
//    pPara->alpha = 1.2;
//    pPara->gain = 250;
//    pPara->slew = 150;
//    pPara->beta  = 0;
//    pPara->gamma = 0;
//    pPara->tau   = 0;
//    pPara->constF= 0;
//    pPara->randType = 1;
//    pPara->randCutCount = 250;

    //// i2c
//    pPara->alpha = 1.4;
//    pPara->gain = 150;
//    pPara->slew = 1;
//    pPara->beta  = 1.6;
//    pPara->gamma = 1.2;
//    pPara->tau   = 1.0;
//    pPara->constF= 0.7;
//    pPara->randType = 1;
//    pPara->randCutCount = 250;

    printf("After tuning, write the labels to the file:");
    pNtkRes = Abc_NtkMap( pNtkInput, DelayTarget, AreaMulti, DelayMulti, LogFan, Slew, Gain, nGatesMin, fRecovery, fSwitching,
                              fSkipFanout, fUseProfile, fUseBuffs, fVerbose, nodeFileStr, cutFileStr, cellFileStr, preDelayStr, pPara);
        Abc_FrameReplaceCurrentNetwork( pAbc, pNtkRes );
        Abc_SclTimePerform( (SC_Lib *)pAbc->pLibScl, Abc_FrameReadNtk(pAbc),0, 0, 0, 0, 0, labelFileStr);
        float reportDelay = Abc_NtkReportDelay(Abc_FrameReadNtk(pAbc));

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