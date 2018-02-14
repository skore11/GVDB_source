
#include "gvdb_volume_gvdb.h"
using namespace nvdb;

int		Node::getMaskBits()		{ int r = gVDB->getRes(mLev); return (uint64) r*r*r; }
int		Node::getMaskBytes()	{ int r = gVDB->getRes(mLev); return imax( ((uint64) r*r*r) >> 3, 1); }		// divide by bits per byte (2^3=8)
uint64	Node::getMaskWords()	{ int r = gVDB->getRes(mLev); return imax( ((uint64) r*r*r) >> 6, 1); }		// divide by bits per 64-bit word (2^6=64)
uint64*  Node::getMask()			{ return (uint64*) &mMask; }
int		Node::getNumChild()		{ return (mLev==0) ? 0 : countOn(); }
