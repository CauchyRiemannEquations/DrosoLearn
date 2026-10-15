# Official OSF permission metadata snapshot

These 14 JSON files are unchanged HTTP responses from the public OSF API, captured on 2026-10-15 UTC. They contain project, license, provider, folder and file-catalog metadata. They are **not participant response files or stimulus images**. Exact URLs, UTC access times, HTTP status, response byte sizes and independently calculated response SHA-256 hashes are in `source-manifest.json`.

Attribution: Marković, Reiter & Kiebel (2022), DOI 10.3389/fnbeh.2022.962494, OSF h526v / data component 75jrf; Cheng et al. (2023), DOI 10.1038/s41597-023-01933-6, data DOI 10.17605/OSF.IO/X2RAU. The nodes directly declare CC BY 4.0 through the saved license relationship and official license object. License URL: https://creativecommons.org/licenses/by/4.0/legalcode. The responses are unmodified; the manifest, this README and the verification report are new DrosoLearn documentation.

The node declaration does not establish identity or rights for separately hosted GitHub `pybefit` derived CSV files. Individual file-content exceptions were not reviewed. Catalog hashes for referenced participant or stimulus files are server declarations, not independent checks of those file bodies. Earlier access failures are retained as observations. Participant-level Pilot filenames were excluded from this repository snapshot.

From the repository root, run `python scripts/check_human_source_permissions.py`. The verifier reads saved metadata only and performs no network requests or behavioral analysis. It checks exact metadata hashes, public node IDs and license relationships, the official legalcode URL, complete catalog pages and file-metadata schema. An optional `--out` path must be new.
