# Planned Zenodo upload manifest

The Zenodo release should contain:

1. a ZIP/tar archive of the final tagged GitHub release (`v1.0.0`);
2. the four raw P=256 long-stationarity chain NPZ files listed in
   `provenance/OMITTED_LARGE_DATA.csv`;
3. optionally, the four derived observable-cache NPZ files listed in the same
   manifest for convenience;
4. the repository `SHA256SUMS` and a separate checksum file covering the large
   Zenodo-only files;
5. the final manuscript/Supplement PDFs or source bundle if desired by the
   author, clearly labelled as the submitted version.

Do not use Git LFS for these files unless there is a separate reason to do so;
the immutable archive is the cleaner scientific record.
