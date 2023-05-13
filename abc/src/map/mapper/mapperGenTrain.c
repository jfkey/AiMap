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
    abctime clk, clkTotal = Abc_Clock();
    Mio_Library_t * pLib = (Mio_Library_t *)Abc_FrameReadLibGen();


    float minDelay = 1000000000.0;
    char* preDelayStr = "";

   for (int i = 0; i < itera; i ++ ) {
        pNtkRes = Abc_NtkMap( pNtk, DelayTarget, AreaMulti, DelayMulti, LogFan, Slew, Gain, nGatesMin, fRecovery, fSwitching,
                              fSkipFanout, fUseProfile, fUseBuffs, fVerbose, nodeFileStr, cutFileStr, cellFileStr, preDelayStr );
        Abc_FrameReplaceCurrentNetwork( pAbc, pNtkRes );
        // topo
//        pNtkRes = Abc_NtkDupDfs( pNtkRes );
//        Abc_FrameReplaceCurrentNetwork( pAbc, pNtkRes );
        // stime
        // char* tmpStr = "";
        Abc_SclTimePerform( (SC_Lib *)pAbc->pLibScl, Abc_FrameReadNtk(pAbc),0, 0, 0, 0, 0, labelFileStr);
        float reportDelay = Abc_NtkReportDelay(Abc_FrameReadNtk(pAbc));
        if (reportDelay < minDelay)
            minDelay = reportDelay;
//        printf("%.3f\n", minDelay);
    }
    // write into trainFileStr.




}

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////


ABC_NAMESPACE_IMPL_END

