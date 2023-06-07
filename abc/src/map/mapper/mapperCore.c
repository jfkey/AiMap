/**CFile****************************************************************

  FileName    [mapperCore.c]

  PackageName [MVSIS 1.3: Multi-valued logic synthesis system.]

  Synopsis    [Generic technology mapping engine.]

  Author      [MVSIS Group]
  
  Affiliation [UC Berkeley]

  Date        [Ver. 2.0. Started - June 1, 2004.]

  Revision    [$Id: mapperCore.c,v 1.7 2004/10/01 23:41:04 satrajit Exp $]

***********************************************************************/

#include "mapperInt.h"
//#include "resm.h"
#include "map/mapper/mapperPreMatch.c"

ABC_NAMESPACE_IMPL_START


////////////////////////////////////////////////////////////////////////
///                        DECLARATIONS                              ///
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////

/**Function*************************************************************

  Synopsis    [Performs technology mapping for the given object graph.]

  Description [The object graph is stored in the mapping manager.
  First, the AND nodes that fanout into POs are collected in the DFS order.
  Two preprocessing steps are performed: the k-feasible cuts are computed 
  for each node and the truth tables are computed for each cut. Next, the 
  delay-optimal matches are assigned for each node, followed by several 
  iterations of area recoveryd: using area flow (global optimization) 
  and using exact area at a node (local optimization).]
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
int Map_Mapping( Map_Man_t * p, char * nodeFileStr, char * cutFileStr, char* preDelayStr, Map_Train_t *para)
{
    int fShowSwitching         = 0;
    int fUseAreaFlow           = 1;
    int fUseExactArea          = !p->fSwitching;
    int fUseExactAreaWithPhase = !p->fSwitching;
    abctime clk;


    //////////////////////////////////////////////////////////////////////
    // perform pre-mapping computations
    if ( p->fVerbose )
        Map_MappingReportChoices( p ); 
    Map_MappingSetChoiceLevels( p ); // should always be called before mapping!
//    return 1;

    // compute the cuts of nodes in the DFS order
    clk = Abc_Clock();
    Map_MappingCuts( p, para );
    p->timeCuts = Abc_Clock() - clk;
    // derive the truth tables 
    clk = Abc_Clock();
    // 计算每个cut对应的TT，并为每个Cut找到对应的supergate的实现，初始化pCut->M
    Map_MappingTruths( p, para);
    p->timeTruth = Abc_Clock() - clk;
    //////////////////////////////////////////////////////////////////////
//ABC_PRT( "Truths", Abc_Clock() - clk );

    // write cut; node
//    Map_printSuperLib((Map_SuperLib_t *)Abc_FrameReadLibSuper());
    Map_MappingComputeFanouts(p);
    Map_MappingWriteNodeCut(p, (Map_SuperLib_t *)Abc_FrameReadLibSuper(), nodeFileStr, cutFileStr);
    Map_MappingLoadNodeCutDelay(p, (Map_SuperLib_t *)Abc_FrameReadLibSuper(), preDelayStr);


    //////////////////////////////////////////////////////////////////////
    // compute the minimum-delay mapping
    clk = Abc_Clock();
    p->fMappingMode = 0;
    if ( strcmp(preDelayStr, "")){
        if (!Map_MappingMatchesPre( p )) return 0;
    } else{
        if (!Map_MappingMatches( p, para) ) return 0;
    }


    p->timeMatch = Abc_Clock() - clk;
    // compute the references and collect the nodes used in the mapping
    Map_MappingSetRefs( p );
    p->AreaBase = Map_MappingGetArea( p );
if ( p->fVerbose )
{
printf( "Delay    : %s = %8.2f  Flow = %11.1f  Area = %11.1f  %4.1f %%   ",
                    fShowSwitching? "Switch" : "Delay",
                    fShowSwitching? Map_MappingGetSwitching(p) : p->fRequiredGlo,
                    Map_MappingGetAreaFlow(p), p->AreaBase, 0.0 );
ABC_PRT( "Time", p->timeMatch );
}
    //////////////////////////////////////////////////////////////////////

    if ( !p->fAreaRecovery )
    {
        if ( p->fVerbose )
            Map_MappingPrintOutputArrivals( p );
        return 1;
    }

    //////////////////////////////////////////////////////////////////////
    // perform area recovery using area flow
    clk = Abc_Clock();
    if ( fUseAreaFlow )
    {

        p->fMappingMode = 1;
        if ( strcmp(preDelayStr, "")){
            // compute the required times
            Map_TimeComputeRequiredGlobalPre( p );
            // recover area flow
            if (!Map_MappingMatchesPre( p )) return 0;
        } else{
            // compute the required times
            Map_TimeComputeRequiredGlobal( p );
            // recover area flow
            if (!Map_MappingMatches( p, para ) ) return 0;
        }
        // compute the references and collect the nodes used in the mapping
        Map_MappingSetRefs( p );
        p->AreaFinal = Map_MappingGetArea( p );
if ( p->fVerbose )
{
printf( "AreaFlow : %s = %8.2f  Flow = %11.1f  Area = %11.1f  %4.1f %%   ",
                    fShowSwitching? "Switch" : "Delay",
                    fShowSwitching? Map_MappingGetSwitching(p) : p->fRequiredGlo,
                    Map_MappingGetAreaFlow(p), p->AreaFinal,
                    100.0*(p->AreaBase-p->AreaFinal)/p->AreaBase );
ABC_PRT( "Time", Abc_Clock() - clk );
}
    }
    p->timeArea += Abc_Clock() - clk;
    //////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////
    // perform area recovery using exact area
    clk = Abc_Clock();
    if ( fUseExactArea )
    {
        p->fMappingMode = 2;
        if ( strcmp(preDelayStr, "")){
            // compute the required times
            Map_TimeComputeRequiredGlobalPre( p );
            // recover area flow
            if (!Map_MappingMatchesPre( p )) return 0;
        } else{
            // compute the required times
            Map_TimeComputeRequiredGlobal( p );
            // recover area flow
            if (!Map_MappingMatches( p, para) ) return 0;
        }
        // compute the references and collect the nodes used in the mapping
        Map_MappingSetRefs( p );
        p->AreaFinal = Map_MappingGetArea( p );
if ( p->fVerbose )
{
printf( "Area     : %s = %8.2f  Flow = %11.1f  Area = %11.1f  %4.1f %%   ",
                    fShowSwitching? "Switch" : "Delay",
                    fShowSwitching? Map_MappingGetSwitching(p) : p->fRequiredGlo,
                    0.0, p->AreaFinal,
                    100.0*(p->AreaBase-p->AreaFinal)/p->AreaBase );
ABC_PRT( "Time", Abc_Clock() - clk );
}
    }
    p->timeArea += Abc_Clock() - clk;
    //////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////
    // perform area recovery using exact area
    clk = Abc_Clock();
    if ( fUseExactAreaWithPhase )
    {
        p->fMappingMode = 3;
        if ( strcmp(preDelayStr, "")){
            // compute the required times
            Map_TimeComputeRequiredGlobalPre( p );
            // recover area flow
            if (!Map_MappingMatchesPre( p )) return 0;
        } else{
            // compute the required times
            Map_TimeComputeRequiredGlobal( p );
            // recover area flow
            if (!Map_MappingMatches( p, para) ) return 0;
        }
        // compute the references and collect the nodes used in the mapping
        Map_MappingSetRefs( p );
        p->AreaFinal = Map_MappingGetArea( p );
if ( p->fVerbose )
{
printf( "Area     : %s = %8.2f  Flow = %11.1f  Area = %11.1f  %4.1f %%   ",
                    fShowSwitching? "Switch" : "Delay",
                    fShowSwitching? Map_MappingGetSwitching(p) : p->fRequiredGlo,
                    0.0, p->AreaFinal,
                    100.0*(p->AreaBase-p->AreaFinal)/p->AreaBase );
ABC_PRT( "Time", Abc_Clock() - clk );
}
    }
    p->timeArea += Abc_Clock() - clk;
    //////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////
    // perform area recovery using exact area
    clk = Abc_Clock();
    if ( p->fSwitching )
    {
        p->fMappingMode = 4;
        if ( strcmp(preDelayStr, "")){
            // compute the required times
            Map_TimeComputeRequiredGlobalPre( p );
            // recover area flow
            if (!Map_MappingMatchesPre( p )) return 0;
        } else{
            // compute the required times
            Map_TimeComputeRequiredGlobal( p );
            // recover area flow
            if (!Map_MappingMatches( p, para) ) return 0;
        }

        // compute the references and collect the nodes used in the mapping
        Map_MappingSetRefs( p );
        p->AreaFinal = Map_MappingGetArea( p );
if ( p->fVerbose )
{
printf( "Switching: %s = %8.2f  Flow = %11.1f  Area = %11.1f  %4.1f %%   ",
                    fShowSwitching? "Switch" : "Delay",
                    fShowSwitching? Map_MappingGetSwitching(p) : p->fRequiredGlo,
                    0.0, p->AreaFinal,
                    100.0*(p->AreaBase-p->AreaFinal)/p->AreaBase );
ABC_PRT( "Time", Abc_Clock() - clk );
}

        p->fMappingMode = 4;
        if ( strcmp(preDelayStr, "")){
            // compute the required times
            Map_TimeComputeRequiredGlobalPre( p );
            // recover area flow
            if (!Map_MappingMatchesPre( p )) return 0;
        } else{
            // compute the required times
            Map_TimeComputeRequiredGlobal( p );
            // recover area flow
            if (!Map_MappingMatches( p, para) ) return 0;
        }
        // compute the references and collect the nodes used in the mapping
        Map_MappingSetRefs( p );
        p->AreaFinal = Map_MappingGetArea( p );
if ( p->fVerbose )
{
printf( "Switching: %s = %8.2f  Flow = %11.1f  Area = %11.1f  %4.1f %%   ", 
                    fShowSwitching? "Switch" : "Delay", 
                    fShowSwitching? Map_MappingGetSwitching(p) : p->fRequiredGlo, 
                    0.0, p->AreaFinal, 
                    100.0*(p->AreaBase-p->AreaFinal)/p->AreaBase );
ABC_PRT( "Time", Abc_Clock() - clk );
}
    }
    p->timeArea += Abc_Clock() - clk;
    //////////////////////////////////////////////////////////////////////

    // print the arrival times of the latest outputs
    if ( p->fVerbose )
        Map_MappingPrintOutputArrivals( p );


    return 1;
}
ABC_NAMESPACE_IMPL_END

