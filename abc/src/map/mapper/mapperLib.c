/**CFile****************************************************************

  FileName    [mapperLib.c]

  PackageName [MVSIS 1.3: Multi-valued logic synthesis system.]

  Synopsis    [Generic technology mapping engine.]

  Author      [MVSIS Group]
  
  Affiliation [UC Berkeley]

  Date        [Ver. 2.0. Started - June 1, 2004.]

  Revision    [$Id: mapperLib.c,v 1.6 2005/01/23 06:59:44 alanmi Exp $]

***********************************************************************/
//#define _BSD_SOURCE

#ifndef WIN32
#define _DEFAULT_SOURCE
#include <unistd.h>
#endif

#include "mapperInt.h"
#include "map/super/super.h"
#include "map/mapper/mapperInt.h"
#include "map/mio/mio.h"
//#include "mapperFanout.c"

ABC_NAMESPACE_IMPL_START


////////////////////////////////////////////////////////////////////////
///                        DECLARATIONS                              ///
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////

/**Function*************************************************************

  Synopsis    [Reads in the supergate library and prepares it for use.]

  Description [The supergates library comes in a .super file. This file
  contains descriptions of supergates along with some relevant information.
  This procedure reads the supergate file, canonicizes the supergates,
  and constructs an additional lookup table, which can be used to map
  truth tables of the cuts into the pair (phase, supergate). The phase
  indicates how the current truth table should be phase assigned to 
  match the canonical form of the supergate. The resulting phase is the
  bitwise EXOR of the phase needed to canonicize the supergate and the
  phase needed to transform the truth table into its canonical form.]
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
Map_SuperLib_t * Map_SuperLibCreate( Mio_Library_t * pGenlib, Vec_Str_t * vStr, char * pFileName, char * pExcludeFile, int fAlgorithm, int fVerbose )
{
    Map_SuperLib_t * p;
    abctime clk;

    // start the supergate library
    p = ABC_ALLOC( Map_SuperLib_t, 1 );
    memset( p, 0, sizeof(Map_SuperLib_t) );
    p->pName     = Abc_UtilStrsav(pFileName);
    p->fVerbose  = fVerbose;
    p->mmSupers  = Extra_MmFixedStart( sizeof(Map_Super_t) );
    p->mmEntries = Extra_MmFixedStart( sizeof(Map_HashEntry_t) );
    p->mmForms   = Extra_MmFlexStart();
    Map_MappingSetupTruthTables( p->uTruths );

    // start the hash table
    p->tTableC = Map_SuperTableCreate( p );
    p->tTable  = Map_SuperTableCreate( p );

    // read the supergate library from file
clk = Abc_Clock();
    if ( vStr != NULL )
    {
        // read the supergate library from file
        int Status = Map_LibraryReadFileTreeStr( p, pGenlib, vStr, pFileName );
        if ( Status == 0 )
        {
            Map_SuperLibFree( p );
            return NULL;
        }
        // prepare the info about the library
        Status = Map_LibraryDeriveGateInfo( p, NULL );
        if ( Status == 0 )
        {
            Map_SuperLibFree( p );
            return NULL;
        }
        assert( p->nVarsMax > 0 );
    }
    else if ( fAlgorithm )
    {
        if ( !Map_LibraryReadTree( p, pGenlib, pFileName, pExcludeFile ) )
        {
            Map_SuperLibFree( p );
            return NULL;
        }
    }
    else
    {
        if ( pExcludeFile != 0 )
        {
            Map_SuperLibFree( p );
            printf ("Error: Exclude file support not present for old format. Stop.\n");
            return NULL;
        }
        if ( !Map_LibraryRead( p, pFileName ) )
        {
            Map_SuperLibFree( p );
            return NULL;
        }
    }
    assert( p->nVarsMax > 0 );

    // report the stats
    if ( fVerbose ) 
    {
        printf( "Loaded %d unique %d-input supergates from \"%s\".  ", 
            p->nSupersReal, p->nVarsMax, pFileName );
        ABC_PRT( "Time", Abc_Clock() - clk );
    }

    // assign the interver parameters
    p->pGateInv        = Mio_LibraryReadInv( p->pGenlib );
    p->tDelayInv.Rise  = Mio_LibraryReadDelayInvRise( p->pGenlib );
    p->tDelayInv.Fall  = Mio_LibraryReadDelayInvFall( p->pGenlib );
    p->tDelayInv.Worst = MAP_MAX( p->tDelayInv.Rise, p->tDelayInv.Fall );
    p->AreaInv         = Mio_LibraryReadAreaInv( p->pGenlib );
    p->AreaBuf         = Mio_LibraryReadAreaBuf( p->pGenlib );

    // assign the interver supergate
    p->pSuperInv = (Map_Super_t *)Extra_MmFixedEntryFetch( p->mmSupers );
    memset( p->pSuperInv, 0, sizeof(Map_Super_t) );
    p->pSuperInv->Num         = -1;
    p->pSuperInv->nGates      =  1;
    p->pSuperInv->nFanins     =  1;
    p->pSuperInv->nFanLimit   = 10;
    p->pSuperInv->pFanins[0]  = p->ppSupers[0];
    p->pSuperInv->pRoot       = p->pGateInv;
    p->pSuperInv->Area        = p->AreaInv;
    p->pSuperInv->tDelayMax   = p->tDelayInv;
    p->pSuperInv->tDelaysR[0].Rise = MAP_NO_VAR;
    p->pSuperInv->tDelaysR[0].Fall = p->tDelayInv.Rise;
    p->pSuperInv->tDelaysF[0].Rise = p->tDelayInv.Fall;
    p->pSuperInv->tDelaysF[0].Fall = MAP_NO_VAR;
    return p;
}


/**Function*************************************************************

  Synopsis    [Deallocates the supergate library.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Map_SuperLibFree( Map_SuperLib_t * p )
{
    if ( p == NULL ) return;
    if ( p->pGenlib )
    {
        if ( p->pGenlib != Abc_FrameReadLibGen() )
            Mio_LibraryDelete( p->pGenlib );
        p->pGenlib = NULL;
    }
    if ( p->tTableC )
        Map_SuperTableFree( p->tTableC );
    if ( p->tTable )
        Map_SuperTableFree( p->tTable );
    Extra_MmFixedStop( p->mmSupers );
    Extra_MmFixedStop( p->mmEntries );
    Extra_MmFlexStop( p->mmForms );
    ABC_FREE( p->ppSupers );
    ABC_FREE( p->pName );
    ABC_FREE( p );
}

/**Function*************************************************************

  Synopsis    [Derives the library from the genlib library.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
int Map_SuperLibDeriveFromGenlib( Mio_Library_t * pLib, int fVerbose )
{
    Map_SuperLib_t * pLibSuper;
    Vec_Str_t * vStr;
    char * pFileName;
    if ( pLib == NULL )
        return 0;

    // compute supergates
    vStr = Super_PrecomputeStr( pLib, 5, 1, 100000000, 10000000, 10000000, 100, 1, 0 );
    if ( vStr == NULL )
        return 0;

    // create supergate library
    pFileName = Extra_FileNameGenericAppend( Mio_LibraryReadName(pLib), ".super" );
    pLibSuper = Map_SuperLibCreate( pLib, vStr, pFileName, NULL, 1, 0 );
    // liujf
//    FILE *pFile = fopen( "supergate.lib", "w" );;
//    fprintf( pFile, "%s", Vec_StrArray(vStr) );

    Vec_StrFree( vStr );

    // replace the library
    Map_SuperLibFree( (Map_SuperLib_t *)Abc_FrameReadLibSuper() );
    Abc_FrameSetLibSuper( pLibSuper );


    return 1;
}

/**Function*************************************************************

  Synopsis    [Derives the library from the genlib library.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
int Map_SuperLibDeriveFromGenlib2( Mio_Library_t * pLib, int fVerbose )
{
    Abc_Frame_t * pAbc = Abc_FrameGetGlobalFrame();
    char * pFileName;
    if ( pLib == NULL )
        return 0;
    // compute supergates
    pFileName = Extra_FileNameGenericAppend(Mio_LibraryReadName(pLib), ".super");
    Super_Precompute( pLib, 5, 1, 100000000, 10000000, 10000000, 100, 1, 0, pFileName );
    // assuming that it terminated successfully
    if ( Cmd_CommandExecute( pAbc, pFileName ) )
    {
        fprintf( stdout, "Cannot execute command \"read_super %s\".\n", pFileName );
        return 0;
    }
    return 1;
}


/**Function*************************************************************

  Synopsis    [print the supergate library.]

  Description []

  SideEffects []

  SeeAlso     []

***********************************************************************/
void Map_printSuperLib( Map_SuperLib_t * p ){
    Map_Super_t * pSuper;
    int nSupers = 0, i, Counter;
    for (i = 0;i <p->nSupersAll; i ++){
        pSuper = p->ppSupers[i];
        printf("supergate with new function.");
        if (pSuper->nUsed == 0){
            nSupers += 1;
            for ( Counter = 0; pSuper; pSuper = pSuper->pNext, Counter++ ) {
                if (pSuper->pRoot == NULL) continue;
                pSuper->nUsed = 1;
                printf( "%s",            Mio_GateReadName(pSuper->pRoot));
                printf( "%5d   ",        Counter );
                printf( "%5d   ",        pSuper->Num );
                printf( "A = %5.2f   ",  pSuper->Area );
                printf( "D = %5.2f   ",  pSuper->tDelayMax.Rise );
                printf( "%s",            pSuper->pFormula );
                printf( "\n" );
            }
        }
    }
    printf("\nIn total %d different supergates\n", nSupers);
}

void Map_MappingComputeFanouts(Map_Man_t * pMan ){
    Map_Node_t * pNode, *f1, *f2;
    int i = 0;
    for ( i = 0; i < pMan->vMapObjs->nSize; i++ )
    {
        pNode = pMan->vMapObjs->pArray[i];
        pNode ->nFanouts = 0;
    }

    for ( i = 0; i < pMan->vMapObjs->nSize; i++ )
    {
        // for each node
        pNode = pMan->vMapObjs->pArray[i];
        if ( Map_NodeIsBuf(pNode) )
        {
            assert( pNode->p2 == NULL );
            pNode->tArrival[0] = Map_Regular(pNode->p1)->tArrival[ Map_IsComplement(pNode->p1)];
            pNode->tArrival[1] = Map_Regular(pNode->p1)->tArrival[!Map_IsComplement(pNode->p1)];
            continue;
        }
        // skip primary inputs and secondary nodes if mapping with choices
//        if ( !Map_NodeIsAnd( pNode ) || pNode->pRepr )
//            continue;
        if (pNode->p1 != NULL) {
            f1 = Map_Regular(pNode->p1);
            f1->nFanouts += 1;
        }
        if (pNode->p2 != NULL) {
            f2 = Map_Regular(pNode->p2);
            f2->nFanouts += 1;
        }
    }
}

void Map_MappingWriteNodeFeatures(Map_Man_t * pMan, FILE * pNodeFile) {
    Map_Node_t * pNode;
    int i = 0;
    float totalLevel = Map_MappingGetMaxLevel(pMan) *1.0;
    fprintf(pNodeFile, "node_id,fanout,level,inv,fanout_p1,level_p1,inv_p1,fanout_p2,level_p2,inv_p2,re_level\n");
    for ( i = 0; i < pMan->vMapObjs->nSize; i++ )
    {
        // for each node
        pNode = pMan->vMapObjs->pArray[i];
        if ( Map_NodeIsBuf(pNode) )
        {
            assert( pNode->p2 == NULL );
            pNode->tArrival[0] = Map_Regular(pNode->p1)->tArrival[ Map_IsComplement(pNode->p1)];
            pNode->tArrival[1] = Map_Regular(pNode->p1)->tArrival[!Map_IsComplement(pNode->p1)];
            continue;
        }
        // skip primary inputs and secondary nodes if mapping with choices
//        if ( !Map_NodeIsAnd( pNode ) || pNode->pRepr )
//            continue;

        // make sure that at least one non-trival cut is present
        if ( pNode->pCuts->pNext == NULL && ( Map_NodeIsAnd( pNode ) && !pNode->pRepr))
        {
            // Extra_ProgressBarStop( pProgress );
            printf( "\nError: A node in the mapping graph does not have feasible cuts.\n" );
            return ;
        }

        float relativeLevel = (totalLevel - pNode->Level)/totalLevel;
        if (pNode->p1 != NULL && pNode->p2 != NULL) {
            fprintf(pNodeFile, "%d,%d,%.3f,%u,%d,%.3f,%u,%d,%.3f,%u,%.3f\n",
                    pNode->Num,  pNode->nFanouts, pNode->Level*1.0/totalLevel, pNode->fInv,
                    Map_Regular(pNode->p1)->nFanouts,  1.0*pNode->p1->Level/totalLevel, pNode->p1->fInv,
                    Map_Regular(pNode->p2)->nFanouts,  1.0*pNode->p2->Level/totalLevel, pNode->p2->fInv, relativeLevel);
        } else{
            fprintf(pNodeFile, "%d,%d,%.3f,%u,%d,%.3f,%u,%d,%.3f,%u,%.3f\n",
                    pNode->Num,  pNode->nFanouts, pNode->Level*1.0/totalLevel, pNode->fInv,
                    0, 0.0, 0, 0, 0.0, 0, relativeLevel);
        }

    }
}

int hash_string( const char *str) {
    int hash_max = 5003;
    unsigned int hash = 0;
    int p;
    for(p = 0; p<strlen(str); p++)
        hash += str[p]*p;
    return hash % hash_max;
}

int isVecContainName( Vec_Ptr_t * vecName, char* tmpName){
    int res = 0, i = 0;
    char * pEntry;
    Vec_PtrForEachEntry(char*, vecName, pEntry, i )
        if (tmpName == pEntry) {
            res = 1;
            break;
        }
    return res;
}


void Map_MappingUpdateDelay(Map_Man_t * pMan, Map_SuperLib_t * p, FILE * preDelayFile) {
    // load the predicted delay
    Vec_Int_t * nodeIdVec = Vec_IntAlloc(1000);
    Vec_Ptr_t * cellNameVec = Vec_PtrAlloc(1000);
    Vec_Int_t * phaseVec = Vec_IntAlloc(1000);
    Vec_Flt_t * preDelayVec = Vec_FltAlloc(1000);
    char line[1024];
    while (fgets(line, 1024, preDelayFile) != NULL) {
        char *token;
        token = strtok(line, ",");
        if (token == NULL)
            printf( "\nError: load the predicted delay with NULL value.\n" );
        if (token != NULL)
            Vec_IntPush(nodeIdVec, atoi(token));
        token = strtok(NULL, ",");
        if (token != NULL) {
            char* name = (char*) malloc(strlen(token) + 1);
            strcpy(name, token);
            Vec_PtrPush(cellNameVec, name);
        }
        token = strtok(NULL, ",");
        if (token != NULL)
            Vec_IntPush(phaseVec, atoi(token));
        token = strtok(NULL, ",");
        if (token != NULL)
            Vec_FltPush(preDelayVec, atof(token));
    }
    assert(nodeIdVec->nSize == cellNameVec->nSize);
    assert(nodeIdVec->nSize == phaseVec->nSize);
    assert(nodeIdVec->nSize == preDelayVec->nSize);
//    char * pEntry; int y = 0;
//    Vec_PtrForEachEntry(char*, cellNameVec, pEntry, y )
//        printf("%s\n", pEntry);

    // update super node with the predicted delay
    Map_Node_t * pNode;
    int i = 0, idx = 0;
    for ( i = 0; i < pMan->vMapObjs->nSize; i++ )
    {
        // for each node
        pNode = pMan->vMapObjs->pArray[i];
        if ( Map_NodeIsBuf(pNode) )
        {
            assert( pNode->p2 == NULL );
            pNode->tArrival[0] = Map_Regular(pNode->p1)->tArrival[ Map_IsComplement(pNode->p1)];
            pNode->tArrival[1] = Map_Regular(pNode->p1)->tArrival[!Map_IsComplement(pNode->p1)];
            continue;
        }
        // skip primary inputs and secondary nodes if mapping with choices
        if ( !Map_NodeIsAnd( pNode ) || pNode->pRepr )
            continue;
        // make sure that at least one non-trival cut is present
        if ( pNode->pCuts->pNext == NULL )
        {
//            Extra_ProgressBarStop( pProgress );
            printf( "\nError: A node in the mapping graph does not have feasible cuts.\n" );
            return ;
        }
        Map_Cut_t * pCut;
        int sCount = 0, k = 0, j = 0;
        for ( pCut = pNode->pCuts->pNext; pCut; pCut = pCut->pNext ) {
            for (k = 0; k < 2; k ++) {
                Map_Match_t * pMatch = pCut->M + k;
                Map_Super_t * pSuper;
                Vec_Ptr_t * cutSuperNames = Vec_PtrAlloc(100);

                for ( pSuper = pMatch->pSupers, sCount = 0; pSuper; pSuper = pSuper->pNext, sCount++ ) {
                    if (sCount >= 30) break;
                    char* curSuperName = Mio_GateReadName(pSuper->pRoot);
                    if (isVecContainName(cutSuperNames, curSuperName) == 1 ) {
                        // set other supergate to
                        pSuper->tDelayPre.Rise = pSuper->tDelayPre.Fall = pSuper->tDelayPre.Worst
                                =  pSuper->tDelayPre.EstWorst = 1000; // Vec_FltEntry(preDelayVec, idx);
                        continue;
                    }
                    Vec_PtrPush (cutSuperNames, curSuperName);
                    if (pNode->Num != Vec_IntEntry(nodeIdVec, idx) || strcmp(curSuperName, (char*)Vec_PtrEntry(cellNameVec, idx)) || k != Vec_IntEntry(phaseVec, idx))
                        printf( "\nError: The index-to-value of delay for update prediction is incorrect: (%d,%s,%d,%d).\n",pNode->Num, curSuperName, k,  idx);
                    pSuper->tDelayPre.Rise = pSuper->tDelayPre.Fall = pSuper->tDelayPre.Worst
                            =  pSuper->tDelayPre.EstWorst =   Vec_FltEntry(preDelayVec, idx); //Vec_FltEntry(preDelayVec, idx)
                    Vec_PtrPush(pNode->pPreSupres, curSuperName);
                    Vec_IntPush(pNode->pPrePhases, k);
                    Vec_FltPush(pNode->pPreDelys, Vec_FltEntry(preDelayVec, idx));
                    idx += 1;
                }
                Vec_PtrClear(cutSuperNames);
            }
        }
    }
}


void Map_MappingWriteCutFeatures(Map_Man_t * pMan, Map_SuperLib_t * p, FILE * pCutFile) {
    Map_Node_t * pNode;
    int i = 0, j = 0, k = 0, nFi = 0;
    fprintf(pCutFile, "node_id,c1,c2,c3,c4,c5,"
                      "nleaves,nvolume,p1_nleaves,p2_nleaves,p1_nvolume,p2_nvolume,max_level,min_level,level_gap,max_fanout,min_fanout,fanout_gap,"
                      "cell_name,s_delay_r,s_delay_f,s_area,s_f1_delay,s_f2_delay,s_f3_delay,s_f4_delay,s_f5_delay,s_f6_delay,phase\n" );

    for ( i = 0; i < pMan->vMapObjs->nSize; i++ )
    {
        // for each node
        pNode = pMan->vMapObjs->pArray[i];
        if ( Map_NodeIsBuf(pNode) )
        {
            assert( pNode->p2 == NULL );
            pNode->tArrival[0] = Map_Regular(pNode->p1)->tArrival[ Map_IsComplement(pNode->p1)];
            pNode->tArrival[1] = Map_Regular(pNode->p1)->tArrival[!Map_IsComplement(pNode->p1)];
            continue;
        }

        // skip primary inputs and secondary nodes if mapping with choices
        if ( !Map_NodeIsAnd( pNode ) || pNode->pRepr )
            continue;

        // make sure that at least one non-trival cut is present
        if ( pNode->pCuts->pNext == NULL )
        {
//            Extra_ProgressBarStop( pProgress );
            printf( "\nError: A node in the mapping graph does not have feasible cuts.\n" );
            return ;
        }

        int sCount = 0;
        Map_Cut_t * pCut;
        float totalLevel = Map_MappingGetMaxLevel(pMan) *1.0;
        for ( pCut = pNode->pCuts->pNext; pCut; pCut = pCut->pNext ) {
            for (k = 0; k < 2; k ++) {
                Map_Match_t * pMatch = pCut->M + k;
                Map_Super_t * pSuper;
                Vec_Ptr_t * cutSuperNames = Vec_PtrAlloc(100);

                for ( pSuper = pMatch->pSupers, sCount = 0; pSuper; pSuper = pSuper->pNext, sCount++ ) {
                    if (sCount >= 30) break;
                    char* curSuperName = Mio_GateReadName(pSuper->pRoot);
                    if (isVecContainName(cutSuperNames, curSuperName) == 1 ) {
                        continue;
                    }
                    Vec_PtrPush (cutSuperNames, curSuperName);

                    // node id
                    // "node_id,c1,c2,c3,c4,c5,"
                    fprintf( pCutFile, "%d,", pNode->Num );
                    float  minLevel = 1000000, maxLevel = 0,  maxFanout = 0, minFanout = 10000;
                    for ( j = 0; j < pMan->nVarsMax; j++ ){
                        if ( pCut->ppLeaves[j] ){
                            fprintf( pCutFile,"%d,", pCut->ppLeaves[j]->Num );
                            if (pCut->ppLeaves[j]->Level < minLevel)
                                minLevel = pCut->ppLeaves[j]->Level;
                            if (pCut->ppLeaves[j]->Level > maxLevel)
                                maxLevel = pCut->ppLeaves[j]->Level;
                            if (Map_Regular(pCut->ppLeaves[j])->nFanouts < minFanout)
                                minFanout = Map_Regular(pCut->ppLeaves[j])->nFanouts;
                            if (Map_Regular(pCut->ppLeaves[j])->nFanouts > maxFanout)
                                maxFanout = Map_Regular(pCut->ppLeaves[j])->nFanouts;
                        }
                        else {
                            fprintf( pCutFile,"%d,", -1);
                        }
                    }
                    // cut features
                    // "nleaves,nvolume,p1_nleaves,p2_nleaves,p1_nvolume,p2_nvolume,max_level,min_level,level_gap,max_fanout,min_fanout,fanout_gap"
                    fprintf( pCutFile,"%d,%d,", pCut->nLeaves,pCut->nVolume );
                    fprintf( pCutFile,"%d,%d,", pCut->pOne->nLeaves, pCut->pTwo->nLeaves );
                    fprintf( pCutFile,"%d,%d,", pCut->pOne->nVolume,pCut->pTwo->nVolume);
                    fprintf( pCutFile,"%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,", maxLevel/totalLevel, minLevel/totalLevel,
                             (maxLevel-minLevel)/totalLevel, maxFanout, minFanout, maxFanout-minFanout);

                    // supergate features
                    // "s_name,s_delay_r,s_delay_f,s_area,s_f1_delay,s_f2_delay,s_f3_delay,s_f4_delay,s_f5_delay,s_f6_delay,phase"
                    fprintf( pCutFile,"%s,", Mio_GateReadName(pSuper->pRoot));
                    fprintf( pCutFile,"%.3f,", pSuper->tDelayMax.Rise );
                    fprintf( pCutFile,"%.3f,", pSuper->tDelayMax.Fall );
                    fprintf( pCutFile,"%.3f,", pSuper->Area );
                    for (nFi = 0; nFi < pSuper->nFanins; nFi++ )
                        fprintf( pCutFile,"%.3f,", (pSuper->tDelaysF[nFi].Fall + pSuper->tDelaysF[nFi].Rise + pSuper->tDelaysR[nFi].Fall +pSuper->tDelaysR[nFi].Rise ) / 2);
                    for (nFi; nFi < 6; nFi ++ )
                        fprintf( pCutFile,"%.3f,", 0.000);
                    fprintf( pCutFile,"%d\n", k );
                }
                Vec_PtrClear(cutSuperNames);
            }
        }
    }
}

void Map_MappingWriteNodeCut(Map_Man_t * pMan, Map_SuperLib_t * p, char* nodeFileStr, char*cutFileStr){
    if ( !strcmp(nodeFileStr, "") || !strcmp(cutFileStr, "") ) return;
    int i = 0;
    Map_Node_t * pNode;
    FILE * pNodeFile = fopen( nodeFileStr, "w" );
    FILE * pCutFile = fopen( cutFileStr, "w" );

    if ( pNodeFile == NULL || pCutFile == NULL )
        printf( "Cannot open text file \"%s, %s\" for writing node and cut features.\n", nodeFileStr, cutFileStr );
    else
    {
        // compute the fanout for each node.
        Map_MappingComputeFanouts(pMan);
        Map_MappingWriteNodeFeatures(pMan, pNodeFile);
        Map_MappingWriteCutFeatures(pMan, p, pCutFile);
        fclose( pNodeFile );
        fclose( pCutFile );
        printf( "Dumped node, cut features into text file \"%s, %s\".\n", nodeFileStr, cutFileStr );
    }
}

void Map_MappingLoadNodeCutDelay(Map_Man_t * pMan, Map_SuperLib_t * p, char* preDelayStr){
    if ( !strcmp(preDelayStr, "")  ) return;
    int i = 0;
    Map_Node_t * pNode;
    FILE * preDelayFile = fopen( preDelayStr, "r" );

    if ( preDelayFile == NULL   )
        printf( "Cannot open text file \"%s\" for load supergate delay.\n", preDelayStr );
    else
    {
        Map_MappingUpdateDelay( pMan, p, preDelayFile);
        fclose( preDelayFile );
        printf( "Loaded predicted supergate delay from the file \"%s \".\n", preDelayStr);
    }
}

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////


ABC_NAMESPACE_IMPL_END

