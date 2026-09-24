# State/dataflow focused workload

This manifest is a diagnostic bucket for the next state-kernel candidate. It is
not an alternative acceptance workload. The parent full manifest has 62 cases
(11 correctness + 51 deployment rows); this file has 33 cases, including
inherited cases plus explicit boundary and random-state focus rows that are not
part of the parent denominator.

The selection has four purposes:

1. exercise all state semantics before looking at speed;
2. isolate low-slot long-prefill rows where the H kernel is grid-starved and the
   previous BV-gate mechanism actually fired;
3. cover chunk-boundary and near-boundary lengths without changing the full gate;
4. retain saturated B8/B32/B128 controls so a local improvement is not mistaken
   for a general improvement.

The seven primary rows are `uniform_b1_s4096`, `uniform_b1_s8192`,
`uniform_b1_s16384`, `uniform_b2_s4096`, `uniform_b2_s8192`,
`resumed_chunk_s16384` and `ragged_mixed_b2`. They are the rows where prior
NCU/paired evidence found the state-kernel opportunity.

The full workload remains the only promotion denominator. No live serving claim
can be made from this synthetic diagnostic bucket; a real 8K serving trace must
be measured separately after a candidate passes the isolated gates.
