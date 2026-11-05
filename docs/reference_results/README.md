# Reference results
These are actual Linux build-verification results, not results from your Windows machine. Reproduce the pipeline and export your aggregates to docs/my_results before using the numbers in your resume. Software versions, hardware and library differences can affect results and latency.

WDC matching: 644 held-out pairs; selected multimodal LightGBM AP 0.951389 vs lexical logistic regression AP 0.819382. Suggestion precision 0.980392 at coverage 0.395963 (255 pairs), using a validation-selected threshold. AP is not classification accuracy.

ABO: 147,702 metadata documents ingested in a separate MongoDB test. These are not matching training examples. Latency excludes embedding generation, network and database operations.
