# ACL/ARR review release checklist

This package is a draft review candidate, **not an automatically verified anonymous
public release**. Keep the original historical repository private until the
review-specific archive/link has been inspected.

1. **Anonymity:** Review README, notebooks, metadata, embedded ZIP artifacts,
   source comments, paths, URLs, Git remote configuration and full commit history
   for names, affiliations, personal handles, email addresses or identifying
   resource links. An archive without `.git/` is safer than exposing the original
   GitHub repository, but does not itself guarantee anonymity.
2. **How to supply code:** ARR currently allows anonymized supplementary software
   and data archives and properly anonymized repository links. Use the current
   ARR call and submission instructions to select the permitted route. Do not
   present a public personal GitHub link as anonymous review material.
3. **Integrity:** The article's nine raw JSONL files, configurations, engineering
   audits, notebooks and model-execution code have not been altered for this ACL
   update. `MANIFEST.sha256` records the current file hashes, including docs and
   analysis changes. Keep the original TACL-era ZIP as a separate immutable copy.
4. **CPU checks:** From the repository root, run

   ```bash
   pip install -r requirements-analysis.txt
   python analysis/reproduce_reported_results.py
   python analysis/acl_accuracy_reanalysis.py .
   python -m unittest discover -s tests -v
   ```

5. **Licensing and attribution:** Select a release license after checking any
   vendored-code terms. Add citation metadata when appropriate; do not insert
   identifying author metadata into an anonymous review package.
6. **Paper statement:** For the anonymous ACL/ARR submission, describe an
   anonymized software/data supplement or anonymized repository only once it
   is actually supplied. Do not promise a link that reviewers cannot access.
7. **Post-acceptance:** Replace review notices with the public repository URL,
   license, author list, publication citation and DOI.

Official policy: https://aclrollingreview.org/cfp
