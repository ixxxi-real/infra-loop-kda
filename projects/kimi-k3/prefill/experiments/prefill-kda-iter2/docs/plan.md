# Executed optimization plan

1. Freeze the 62-case KDA prefill workload and verify the exact base commit.
2. Run the full operator baseline and a warmed NSYS capture around five
   `kda_capture` iterations.
3. Exclude autotune/cache activity from hotspot accounting by correlating CUDA
   runtime launches with the NVTX capture intervals.
4. Measure the dominant H stage with NCU on `uniform_b1_s16384`.
5. Test one source change at a time, in this order: H `BV=64`, H `warps=8`,
   H `warps=2`, and H `BV=16`; run correctness and precision before each
   complete 51-case benchmark.
6. Reject any candidate below `1.03x` geomean or with stable regressions;
   restore the base source and retain evidence.

All four H candidates passed correctness and precision but were rejected on the
full-workload gate. The final code is the original `BV=32`, `num_warps=4`,
`num_stages=2` configuration. A future structural candidate must be based on
NCU metrics rather than another blind parameter sweep.
