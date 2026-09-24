# Reproducibility

An accepted result should let a second engineer answer five questions:

1. Which model configuration and checkpoint were used?
2. Which exact source commit and patch were used?
3. Which workload, batch/sequence shape and dtype were measured?
4. Which correctness and performance commands produced the evidence?
5. Where are the raw artifacts, and how are their checksums verified?

Put these answers in the task contract, run manifest and final report. Large files can live in object storage or an internal archive; commit a manifest with URI, size, SHA-256, collection timestamp and retention owner.
