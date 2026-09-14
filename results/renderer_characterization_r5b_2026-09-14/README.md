# RC-R5b — transfer characterization

Verdict: `TRANSFER_CHARACTERIZED`.

R5 historical run contained known ResourceWarning. R5b fixes the harness/resource handling and is a new measurement lineage. Historical source and receipts are unchanged.

Historical R5 host_copy included a shading call and an owned NumPy copy. It was not pure PCIe transfer time; the new stage boundaries must not be treated as identical.

A1 throughput is resident/scalar throughput, not full image export. A2 includes packing; A3 includes DMA completion, without compute/copy overlap. Full raw stage/headline timings and separate setup/validation costs are in summary.json and cells/. Device memory includes non-Torch allocations and other desktop processes. No per-library Warp zero is invented.

Causality versus D8b: `NOT_TESTED`.
