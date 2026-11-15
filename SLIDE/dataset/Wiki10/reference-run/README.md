# Provided Wiki10 WTA run

These files preserve one real author-run SLIDE WTA experiment using the documented 30-epoch Wiki10 workflow. They are supporting logs, not synthetic data or an average across runs.

- `training_wta.log`: epoch markers and evaluation measurements; all 30 epoch-end records (epochs 0–29) are present, ending at iteration 13,260.
- `console_wta.log`: captured standard output and error, ending with final evaluation and model-saving messages. The console log does not record a process exit code.
- `Config_wiki10_wta.csv`: a byte-for-byte copy of the configuration used for this run. Its relative paths describe the original working directory, `SLIDE/SLIDE/`; do not run the executable from this archive directory.

Each epoch-end evaluation covers 206 complete test batches, or 6,592 records, with batch size 32. The final incomplete test batch is excluded. Intermediate 20-batch evaluations are also retained in the raw log; the analyzer selects epoch-end measurements.

The validation host reported Ubuntu 24.04.3 LTS x86-64, GCC 13.3.0, and CMake 3.28.3. The per-run OpenMP thread count was not captured in these files. No model checkpoint or MP-WTA training log is included.

See the [provided-log analysis command](../../../../scripts/README.md#provided-wiki10-wta-logs) to compare this log with a real MP-WTA run. Fresh execution follows the [SLIDE guide](../../../README.md#5-run-wta-training) and writes outside this preserved directory.
