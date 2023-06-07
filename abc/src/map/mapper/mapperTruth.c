/**CFile****************************************************************

  FileName    [mapperTruth.c]

  PackageName [MVSIS 1.3: Multi-valued logic synthesis system.]

  Synopsis    [Generic technology mapping engine.]

  Author      [MVSIS Group]
  
  Affiliation [UC Berkeley]

  Date        [Ver. 2.0. Started - June 1, 2004.]

  Revision    [$Id: mapperTruth.c,v 1.8 2005/01/23 06:59:45 alanmi Exp $]

***********************************************************************/

#include "mapperInt.h"
#include "map/mio/mio.h"
#include "map/mio/mioInt.h"

ABC_NAMESPACE_IMPL_START


////////////////////////////////////////////////////////////////////////
///                        DECLARATIONS                              ///
////////////////////////////////////////////////////////////////////////

static void Map_TruthsCut( Map_Man_t * pMan, Map_Cut_t * pCut );
extern void Map_TruthsCutOne( Map_Man_t * p, Map_Cut_t * pCut, unsigned uTruth[] );
static void Map_CutsCollect_rec( Map_Cut_t * pCut, Map_NodeVec_t * vVisited );

Map_Cut_t * Map_CutArray2List( Map_Cut_t ** pArray, int nCuts )
{
    Map_Cut_t * pListNew, ** ppListNew;
    int i;
    pListNew  = NULL;
    ppListNew = &pListNew;
    for ( i = 0; i < nCuts; i++ )
    {
        // connect these lists
        *ppListNew = pArray[i];
        ppListNew  = &pArray[i]->pNext;
    }
//printf( "\n" );

    *ppListNew = NULL;
    return pListNew;
}

int Map_CutList2Array( Map_Cut_t ** pArray, Map_Cut_t * pList )
{
    int i;
    for ( i = 0; pList; pList = pList->pNext, i++ )
        pArray[i] = pList;
    return i;
}

/**Function*************************************************************
  Synopsis    [Randomly shuffles the array of cuts to check QoR impact.]
  Description [Generic random shuffle procedure]

  SideEffects []
  SeeAlso     []
***********************************************************************/
static void shuffle(void *array, size_t n, size_t size) {
    //struct timeval time = time(NULL);

    srand(time(NULL));
    // This if() is not needed functionally, but left per OP's style
    if (n > 1) {
        char *carray = array;
        void * aux;
        aux = malloc(size);
        size_t i;
        for (i = 1; i < n; ++i) {
            size_t j = rand() % (i + 1);
            j *= size;
            memcpy(aux, &carray[j], size);
            memcpy(&carray[j], &carray[i*size], size);
            memcpy(&carray[i*size], aux, size);
        }
        free(aux);
    }
}
////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////

/**Function*************************************************************

  Synopsis    [Derives truth tables for each cut.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Map_MappingTruths( Map_Man_t * pMan, Map_Train_t * pPara ) {
    ProgressBar *pProgress;
    Map_Node_t *pNode;
    Map_Cut_t *pCut;
    int nNodes, i;
    // compute the cuts for the POs
    nNodes = pMan->vMapObjs->nSize;
    pProgress = Extra_ProgressBarStart(stdout, nNodes);
    for (i = 0; i < nNodes; i++) {
        pNode = pMan->vMapObjs->pArray[i];
        if (!Map_NodeIsAnd(pNode))
            continue;
        assert(pNode->pCuts);
        assert(pNode->pCuts->nLeaves == 1);

        // match the simple cut
        pNode->pCuts->M[0].uPhase = 0;
        pNode->pCuts->M[0].pSupers = pMan->pSuperLib->pSuperInv;
        pNode->pCuts->M[0].uPhaseBest = 0;
        pNode->pCuts->M[0].pSuperBest = pMan->pSuperLib->pSuperInv;

        pNode->pCuts->M[1].uPhase = 0;
        pNode->pCuts->M[1].pSupers = pMan->pSuperLib->pSuperInv;
        pNode->pCuts->M[1].uPhaseBest = 1;
        pNode->pCuts->M[1].pSuperBest = pMan->pSuperLib->pSuperInv;

        // match the rest of the cuts
        for (pCut = pNode->pCuts->pNext; pCut; pCut = pCut->pNext)
            Map_TruthsCut(pMan, pCut);
        Extra_ProgressBarUpdate(pProgress, i, "Tables ...");
    }
    Extra_ProgressBarStop(pProgress);

    // update by junfeng. drop the cuts that can not be implemented by supergates.
//    Map_Cut_t * pCutCur, *pCutPrev;
//    for (i = 0; i < nNodes; i++) {
//        pNode = pMan->vMapObjs->pArray[i];
//        if (!Map_NodeIsAnd(pNode))
//            continue;
//        assert(pNode->pCuts);
//        if (pNode->pCuts->nLeaves != 1 ) {
//            printf("i:%d;%d\n",i, pNode->pCuts->nLeaves);
//        }
////        assert(pNode->pCuts->nLeaves == 1);
//        // if the first cut can not be implemented by supergates.
//        pCut = pNode->pCuts->pNext;
//        while(pCut && pCut->M[0].pSupers == NULL && pCut->M[1].pSupers== NULL ){
//            pCut = pCut->pNext;
//        }
//        // if the other cut can not be implemented by supergates.
//        pCutPrev = NULL;
//        pCutCur = pCut;
//        while(pCutCur) {
//            if(pCutCur->M[0].pSupers == NULL && pCutCur->M[1].pSupers== NULL ){
//                pCutCur = pCutCur->pNext;
//            } else{
//                pCutPrev = pCutCur;
//                pCutCur = pCutCur->pNext;
//            }
//        }
//
//        //train strategy 3: random cut filter
//        if (pPara != NULL && pPara->isTrain == 1  && pPara->randCutCount > 0) {
//            Map_Cut_t * pListNew;
//            Map_Cut_t ** pArrayCut = ABC_ALLOC( Map_Cut_t *, 1000);
//            int nCuts ;
//            // move the cuts from the list into the array
//            nCuts = Map_CutList2Array( pArrayCut, pNode->pCuts);
//            shuffle(  pArrayCut, nCuts, sizeof(Map_Cut_t *) );
//            if ( nCuts > pPara->randCutCount - 1 )
//            {
//                // free the remaining cuts
//                for ( i = pPara->randCutCount - 1; i < nCuts; i++ )
//                    Extra_MmFixedEntryRecycle( pMan->mmCuts, (char *)pArrayCut[i] );
//                // update the number of cuts
//                nCuts = pPara->randCutCount - 1;
//            }
//            pListNew = Map_CutArray2List(pArrayCut, nCuts );
//            pNode->pCuts = pListNew;
//        }
//
//    }

    // liujf: 输出每个节点对应的cut 和 supergate实现
//    Map_Super_t * pSuper;
//    int Counter = 0, j= 0;
//    float maxF=0.0, maxR = 0;
//    for ( i = 0; i < nNodes; i++ ) {
//        pNode = pMan->vMapObjs->pArray[i];
//        printf("node(%d,%d)\n",i, pNode->Num);
//        if (!Map_NodeIsAnd(pNode))
//            continue;
//        for (pCut = pNode->pCuts->pNext; pCut; pCut = pCut->pNext){
//            maxF=0.0, maxR = 0;
//            for (pSuper = pCut->M[0].pSupers, Counter = 0; pSuper; pSuper = pSuper->pNext, Counter++){
//                printf("\t\tcut(");
//                for (j =0; j < pMan->nVarsMax; j ++){
//                    if ( pCut->ppLeaves[j] ){
//                        if (pSuper->tDelaysF[j].Fall > maxF) maxF =pSuper->tDelaysF[j].Fall;
//                        if (pSuper->tDelaysF[j].Rise > maxR) maxR =pSuper->tDelaysR[j].Rise;
//                        printf("%d,", pCut->ppLeaves[j]->Num);
//                    }
//                }
//                if (Counter <20 ) {
//                    printf(") %s,(%.2f, %.2f, %.2f, %d, %d)\n", ((Mio_Gate_t *)pSuper->pRoot)->pName, pSuper->Area, maxF, maxR, pSuper->Num, pSuper->nSupers);
//                }
//            }
//            printf("\n");
////            for (pSuper = pCut->M[1].pSupers, Counter = 0; pSuper; pSuper = pSuper->pNext, Counter++){
////                for (j =0; j < pMan->nVarsMax; j ++){
////                    if ( pCut->ppLeaves[j] ){
////                        if (pSuper->tDelaysF[j].Fall > maxF) maxF =pSuper->tDelaysF[j].Fall;
////                        if (pSuper->tDelaysF[j].Rise > maxR) maxR =pSuper->tDelaysR[j].Rise;
////                    }
////                }
////                if (Counter <20 ) {
////                    printf("\t\tPhase(1) %s,(%.2f, %.2f, %.2f, %.2f)", ((Mio_Gate_t *)pSuper->pRoot)->pName, pSuper->Area, maxF, maxR, pSuper->tDelayMax.Worst);
////                }
////            }
////            printf("\n");
//
//        }
// }


}







/**Function*************************************************************

  Synopsis    [Derives the truth table for one cut.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Map_TruthsCut( Map_Man_t * p, Map_Cut_t * pCut )
{ 
//    unsigned uCanon1, uCanon2;
    unsigned uTruth[2], uCanon[2];
    unsigned char uPhases[16];
    unsigned * uCanon2;
    char * pPhases2;
    int fUseFast = 1;
    int fUseSlow = 0;
    int fUseRec = 0; // this does not work for Solaris

    extern int Map_CanonCompute( int nVarsMax, int nVarsReal, unsigned * pt, unsigned ** pptRes, char ** ppfRes );
 
    // generally speaking, 1-input cut can be matched into a wire!
    if ( pCut->nLeaves == 1 )
        return;
/*
    if ( p->nVarsMax == 5 )
    {
        uTruth[0] = pCut->uTruth;
        uTruth[1] = pCut->uTruth;
    }
    else
*/
    Map_TruthsCutOne( p, pCut, uTruth );


    // compute the canonical form for the positive phase
    if ( fUseFast )
        Map_CanonComputeFast( p, p->nVarsMax, pCut->nLeaves, uTruth, uPhases, uCanon );
    else if ( fUseSlow )
        Map_CanonComputeSlow( p->uTruths, p->nVarsMax, pCut->nLeaves, uTruth, uPhases, uCanon );
    else if ( fUseRec )
    {
//        Map_CanonComputeSlow( p->uTruths, p->nVarsMax, pCut->nLeaves, uTruth, uPhases, uCanon );
        Extra_TruthCanonFastN( p->nVarsMax, pCut->nLeaves, uTruth, &uCanon2, &pPhases2 );
/*
        if ( uCanon[0] != uCanon2[0] || uPhases[0] != pPhases2[0] )
        {
            int k = 0;
            Map_CanonCompute( p->nVarsMax, pCut->nLeaves, uTruth, &uCanon2, &pPhases2 );
        }
*/
        uCanon[0] = uCanon2[0];
        uCanon[1] = (p->nVarsMax == 6)? uCanon2[1] : uCanon2[0];
        uPhases[0] = pPhases2[0];
    }
    else
        Map_CanonComputeSlow( p->uTruths, p->nVarsMax, pCut->nLeaves, uTruth, uPhases, uCanon );
    pCut->M[1].pSupers = Map_SuperTableLookupC( p->pSuperLib, uCanon );
    pCut->M[1].uPhase  = uPhases[0];
    p->nCanons++;

//uCanon1 = uCanon[0] & 0xFFFF;

    // compute the canonical form for the negative phase
    uTruth[0] = ~uTruth[0];
    uTruth[1] = ~uTruth[1];
    if ( fUseFast )
        Map_CanonComputeFast( p, p->nVarsMax, pCut->nLeaves, uTruth, uPhases, uCanon );
    else if ( fUseSlow )
        Map_CanonComputeSlow( p->uTruths, p->nVarsMax, pCut->nLeaves, uTruth, uPhases, uCanon );
    else if ( fUseRec )
    {
//        Map_CanonComputeSlow( p->uTruths, p->nVarsMax, pCut->nLeaves, uTruth, uPhases, uCanon );
        Extra_TruthCanonFastN( p->nVarsMax, pCut->nLeaves, uTruth, &uCanon2, &pPhases2 );
/*
        if ( uCanon[0] != uCanon2[0] || uPhases[0] != pPhases2[0] )
        {
            int k = 0;
            Map_CanonCompute( p->nVarsMax, pCut->nLeaves, uTruth, &uCanon2, &pPhases2 );
        }
*/
        uCanon[0] = uCanon2[0];
        uCanon[1] = (p->nVarsMax == 6)? uCanon2[1] : uCanon2[0];
        uPhases[0] = pPhases2[0];
    }
    else
        Map_CanonComputeSlow( p->uTruths, p->nVarsMax, pCut->nLeaves, uTruth, uPhases, uCanon );
    pCut->M[0].pSupers = Map_SuperTableLookupC( p->pSuperLib, uCanon );
    pCut->M[0].uPhase  = uPhases[0];
    p->nCanons++;

//uCanon2 = uCanon[0] & 0xFFFF;
//assert( p->nVarsMax == 4 );
//Rwt_Man4ExploreCount( uCanon1 < uCanon2 ? uCanon1 : uCanon2 );

    // restore the truth table
    uTruth[0] = ~uTruth[0];
    uTruth[1] = ~uTruth[1];
}

/**Function*************************************************************

  Synopsis    [Computes the truth table of one cut.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Map_TruthsCutOne( Map_Man_t * p, Map_Cut_t * pCut, unsigned uTruth[] )
{
    unsigned uTruth1[2], uTruth2[2];
    Map_Cut_t * pTemp = NULL; // Suppress "might be used uninitialized"
    int i;
    // mark the cut leaves
    for ( i = 0; i < pCut->nLeaves; i++ )
    {
        pTemp = pCut->ppLeaves[i]->pCuts;
        pTemp->fMark = 1;
        pTemp->M[0].uPhaseBest = p->uTruths[i][0];
        pTemp->M[1].uPhaseBest = p->uTruths[i][1];
    }
    assert( pCut->fMark == 0 );

    // collect the cuts in the cut cone
    p->vVisited->nSize = 0;
    Map_CutsCollect_rec( pCut, p->vVisited );
    assert( p->vVisited->nSize > 0 );
    pCut->nVolume = p->vVisited->nSize;

    // compute the tables and unmark
    for ( i = 0; i < pCut->nLeaves; i++ )
    {
        pTemp = pCut->ppLeaves[i]->pCuts;
        pTemp->fMark = 0;
    }
    for ( i = 0; i < p->vVisited->nSize; i++ )
    {
        // get the cut
        pTemp = (Map_Cut_t *)p->vVisited->pArray[i];
        pTemp->fMark = 0;
        // get truth table of the first branch
        if ( Map_CutIsComplement(pTemp->pOne) )
        {
            uTruth1[0] = ~Map_CutRegular(pTemp->pOne)->M[0].uPhaseBest;
            uTruth1[1] = ~Map_CutRegular(pTemp->pOne)->M[1].uPhaseBest;
        }
        else
        {
            uTruth1[0] = Map_CutRegular(pTemp->pOne)->M[0].uPhaseBest;
            uTruth1[1] = Map_CutRegular(pTemp->pOne)->M[1].uPhaseBest;
        }
        // get truth table of the second branch
        if ( Map_CutIsComplement(pTemp->pTwo) )
        {
            uTruth2[0] = ~Map_CutRegular(pTemp->pTwo)->M[0].uPhaseBest;
            uTruth2[1] = ~Map_CutRegular(pTemp->pTwo)->M[1].uPhaseBest;
        }
        else
        {
            uTruth2[0] = Map_CutRegular(pTemp->pTwo)->M[0].uPhaseBest;
            uTruth2[1] = Map_CutRegular(pTemp->pTwo)->M[1].uPhaseBest;
        }
        // get the truth table of the output
        if ( !pTemp->Phase )
        {
            pTemp->M[0].uPhaseBest = uTruth1[0] & uTruth2[0];
            pTemp->M[1].uPhaseBest = uTruth1[1] & uTruth2[1];
        }
        else
        {
            pTemp->M[0].uPhaseBest = ~(uTruth1[0] & uTruth2[0]);
            pTemp->M[1].uPhaseBest = ~(uTruth1[1] & uTruth2[1]);
        }
    }
    uTruth[0] = pTemp->M[0].uPhaseBest;
    uTruth[1] = pTemp->M[1].uPhaseBest;
}

/**Function*************************************************************

  Synopsis    [Recursively collect the cuts.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Map_CutsCollect_rec( Map_Cut_t * pCut, Map_NodeVec_t * vVisited )
{
    if ( pCut->fMark )
        return;
    Map_CutsCollect_rec( Map_CutRegular(pCut->pOne), vVisited );
    Map_CutsCollect_rec( Map_CutRegular(pCut->pTwo), vVisited );
    assert( pCut->fMark == 0 );
    pCut->fMark = 1;
    Map_NodeVecPush( vVisited, (Map_Node_t *)pCut );
}

/*
    {
        unsigned * uCanon2;
        char * pPhases2;

        Map_CanonComputeSlow( p->uTruths, p->nVarsMax, pCut->nLeaves, uTruth, uPhases, uCanon );
        Map_CanonCompute( p->nVarsMax, pCut->nLeaves, uTruth, &uCanon2, &pPhases2 );
        if ( uCanon2[0] != uCanon[0] )
        {
            int v = 0;
            Map_CanonCompute( p->nVarsMax, pCut->nLeaves, uTruth, &uCanon2, &pPhases2 );
            Map_CanonComputeFast( p, p->nVarsMax, pCut->nLeaves, uTruth, uPhases, uCanon );
        }
//        else
//        {
//            printf( "Correct.\n" );
//        }
    }
*/

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////


ABC_NAMESPACE_IMPL_END

