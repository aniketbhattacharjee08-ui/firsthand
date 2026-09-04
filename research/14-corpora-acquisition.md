# Human-Prose Reference Corpora: Access, Licenses, Bulk-Download Status

Verified by direct fetch against canonical URLs, S3/GCS/Zenodo/HF/GitHub APIs, and Wayback for one unreachable host. Everything below is live-verified unless flagged unverified.

**Legend:** [FD] free + downloadable · [FR] free + registration · [$] paid · [BO] browse-only

---

## 1. MICUSP — Michigan Corpus of Upper-level Student Papers — [BO], scrapable

829 papers, 16 disciplines, 7 paper types, 4 levels (final-year undergrad through G3), native/non-native flag with first language, sex, 8 textual features. About 2.6M words. This is the closest thing to "A-grade university writing" as a labeled corpus.

Undocumented but verified live endpoints:
```bash
# Metadata CSV, all 829 rows, works anonymously
curl -L -o micusp_papers.csv 'http://micusp.elicorpora.info/browse?mode=download&'
# Full text per paper
http://micusp.elicorpora.info/view?pid=BIO.G0.15.1
# Original PDFs
http://micusp.elicorpora.info/static/search/pdf/<PAPER_ID>.pdf
```
A full 829-paper corpus is obtainable with 829 GETs joined on the CSV. No API, no bulk archive. `/search` currently returns HTTP 500; `/browse` and `/view` work.

License: University of Michigan Regents copyright. Freely available for study, research and teaching. **Commercial use needs advance permission and possibly a fee.** Contact micusp-help@umich.edu.

## 2. BAWE — British Academic Written English — [FD], direct zip

2,761 assessed assignments, 6.5M words, 4 disciplinary groups across 30 disciplines, undergraduate plus taught masters, 13 genre families, 500-5,000 words each.

```
https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/2539/2539.zip?sequence=3&isAllowed=y
```
107.89 MB, no login. TEI-style XML per assignment, foldered by discipline.

License: **CC BY-NC-SA 3.0** per the repository record, which is stronger than the "apply to researchers" framing on the Coventry page. Derivatives may be redistributed non-commercially with share-alike.

Warning: the Bodleian host was unreachable from this machine all session (TCP timeouts). The above is from a 2025-04-06 Wayback snapshot. Retry live before assuming it is dead.

## 3. PERSUADE 2.0 — [FD]

Over 25,000 argumentative essays, US grades 6-12, 15 prompts, two task types, holistic scores, discourse-element annotations with effectiveness ratings, plus writer demographics.

Repo: https://github.com/scrosseye/persuade_corpus_2.0 (CSVs on Google Drive). Test zip password: `persuade_test`.

License: **CC BY-NC-SA 4.0.** This is the human side of the Kaggle Feedback Prize and "LLM - Detect AI Generated Text" competitions.

## 4. ELLIPSE — [FD]

About 6,500 scored English-language-learner writing samples, grades 8-12, plus a raw file with ~9,000 essays and per-rater scores. Holistic proficiency plus six analytic dimensions: cohesion, syntax, vocabulary, phraseology, grammar, conventions.

Repo: https://github.com/scrosseye/ELLIPSE-Corpus. Passwords: `ellipse_test`, `ellipse_raw_data`. License: **CC BY-NC-SA 4.0.**

## 5. ICLE v3 and LOCNESS — time-sensitive

**ICLE v3 becomes free to all users on 2026-09-15**, twelve days from this research date. Over 5.5M words, 25 first-language backgrounds, advanced argumentative essays. Until then it is a paid license via i6doc. Do not pay; wait. Platform: https://corpora.uclouvain.be/cecl/icle/

**LOCNESS** [FR]: 324,304 words of native-speaker student essays (British A-level 60k, British university 96k, American university 168k). Request form at the Learner Corpus Association. License is restrictive: non-commercial, must credit, and **"No part of the corpus is to be distributed to a third party without specific authorization."** Usable to seed a private reference distribution, not for anything published.

## 6. COCA and corpusdata.org — [$], with a free sample

COCA is 950M words, 485,000 texts, 1990-2019, evenly split across spoken, fiction, magazine, newspaper, **academic (~120M words)**, blogs, web, and TV/movie subtitles.

| License | 1 corpus | 2 corpora | each additional |
|---|---|---|---|
| Academic (university email required) | $395 | $695 | $200 |
| Non-academic / commercial | $795 | $1,395 | $400 |

**Do not buy it for stylometry.** The full-text data is deliberately scrambled for copyright: roughly 10 words are removed out of every 200. Sentence-level and long-span features will be corrupted.

**Use the free sample instead:** 8.9M words of COCA, all three formats, no registration, at https://www.corpusdata.org/formats.asp

Word frequency lists: top 5,000 entries free with per-genre frequencies including an academic column. 20,000 words costs $135 academic / $225 commercial.

Worth quoting in positioning material: corpusdata.org now pitches itself as pre-AI human language, "nearly all of which were created right before generative AI became popular in the early 2020s."

## 7. Uppsala Student English (USE) — [FD], best free student-essay corpus

1,489 essays, 440 Swedish university students of English, 1,221,265 words, average 820 words, three levels, 14 essay-type folders.

```
https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/2457/USEcorpus.zip?sequence=5&isAllowed=y
```
3.18 MB, plain .txt files, no registration. License: **CC BY-NC-SA 3.0.** Same host-unreachable caveat as BAWE.

---

## 8. Free open proxies for human academic prose

### 8a. arXiv — [FD] via Google Cloud, requester-pays via S3

**The cleanest "definitely pre-ChatGPT" acquisition path in this report.** The `YYMM` in the filename is the submission year-month, so pre-2022 is trivially sliceable: grab `arXiv_src_0704_*` through `arXiv_src_2112_*` and stop.

- S3 requester-pays: `s3://arxiv`, us-east-1, ~9.2 TB total. `aws s3 cp s3://arxiv/pdf/arXiv_pdf_1001_001.tar . --request-payer requester`
- **Free anonymous alternative:** `gs://arxiv-dataset` on Google Cloud, no requester-pays. Metadata 4.5 GB, PDFs foldered by archive.

License caution: most submissions use the default arXiv license. Distribution permitted, copyright retained. If you build full-text tools you must link back to arXiv. Do not hammer arxiv.org directly.

Ready-made HuggingFace packagings exist but `armanc/scientific_papers` lists its license as "unknown", which is a real problem for a commercial product.

### 8b. PubMed Central OA Subset — [FD], **layout changed August 2026**

**This is the most important stale-documentation trap.** Every tutorial referencing `oa_comm/` and `oa_noncomm/` FTP paths is now wrong. Verbatim from NCBI: "All legacy PMC Article Dataset files on the FTP and Cloud Services were removed the week of August 24, 2026."

Current bucket: `s3://pmc-oa-opendata`, us-east-1, world-readable, free, no requester-pays, no login. About 8 million article versions, one prefix each:

```
s3://pmc-oa-opendata/PMC10009416.1/
  ├── PMC10009416.1.xml   (JATS)
  ├── PMC10009416.1.txt   ← plain text, pre-extracted
  ├── PMC10009416.1.json  (metadata incl. license_code)
  └── PMC10009416.1.pdf
metadata/    inventory-reports/    README.txt
```
```bash
aws s3 cp s3://pmc-oa-opendata/PMC10009402.1/PMC10009402.1.txt . --no-sign-request
```

License filtering replaces the old directory split: read each version's JSON `license_code`, driven from the daily S3 inventory CSV. Terms: no fees, must acknowledge NLM, must not imply NIH endorsement, and redistributors may only redistribute data itself licensed for redistribution.

### 8c. Wikipedia pre-2022 — [FD] via Internet Archive only

dumps.wikimedia.org retains only the last few months. **You cannot get a 2021 dump from Wikimedia.** Archive.org has 19 monthly 2021 items, e.g. `enwiki-20210120` (335 GB), through `enwiki-20211220` (357 GB).

```bash
ia download enwiki-20210120 --glob='*pages-articles-multistream*'
```
License: CC BY-SA 4.0 plus GFDL.

### 8d. Project Gutenberg — [FD]

Easiest modern option is `common-pile/project_gutenberg`: 71,810 documents, 26.2 GB UTF-8 JSON, headers and footers already stripped, **public domain**.

For a full mirror use rsync (`rsync.ibiblio.org::gutenberg`). The website itself is human-only; automated access requires the robot harvest endpoint with a mandatory 2-second delay.

### 8e. CORE.ac.uk — [FR]

Latest dump 2024-07-12, 749 GB compressed / ~2.7 TB extracted, full text plus metadata. Register for access, then wget with HTTP basic auth. Per-record licenses vary since CORE aggregates repositories. Commercial API use requires a paid membership tier.

### 8f. Theses aggregators — one is dead

- **DART-Europe: CLOSED.** Redirects to a UCL closure notice. Remove it from any plan.
- NDLTD, OATD, BASE, OpenDOAR: metadata and links only, no bulk text.
- EThOS: presumed still down after the 2023 British Library cyber-attack.
- **Better alternative: OpenAlex.** `s3://openalex` is anonymously listable and **CC0 1.0**. Metadata and inverted abstracts only, no full text, but free and unrestricted for building a target list of open-access PDFs.

### 8g. Reddit — [FD] via torrents, [BO] officially

**Pushshift is dead as a public resource.** It now requires you to be a registered Reddit moderator and forbids commercialization.

**arctic_shift is the live successor:** https://github.com/ArthurHeitmann/arctic_shift. Full dump 2005-06 through 2025-12 via Academic Torrents, plus monthly dumps and a "top 40k subreddits" slice that is month-sliceable to pre-2022.

Legal posture: these dumps are redistributed against Reddit's current API terms. Acceptable for an internal reference distribution. Do not redistribute, and expect no clean license.

Pre-packaged r/AskHistorians sits on HuggingFace under several `Pavithree/askHistorians*` datasets.

### 8h. peS2o, S2ORC, Semantic Scholar — [FD] and [FR]

- **peS2o** (`allenai/peS2o`) [FD]: V2 is 38.97M documents / 42.01B tokens, **knowledge cutoff 2023-01-03**, so it is almost but not strictly pre-ChatGPT. Filter by year yourself. License **ODC-BY**.
- **Semantic Scholar Datasets API**: 194 releases going back to **2022-05-10**, so you can pull a genuinely pre-LLM snapshot by release ID. `s2orc` is 10M full-text records in 30 × 4 GB files. Free sample without a key; full data needs a free API key. Rate limit 1 req/sec.
- **Dolma** (`allenai/dolma`) [FD]: 4.5 TB, 2.53B documents, 2.31T tokens, **ODC-BY**, not gated.

---

## 9. Paired human/AI corpora (the human side is the point)

| Corpus | Human side | Access | License |
|---|---|---|---|
| **HC3** | 24,300 Q/A pairs, 48,644 rows. reddit_eli5 17,100 · finance 3,930 · open_qa 1,190 · medicine 1,250 · wiki_csai 842 | HF `Hello-SimpleAI/HC3` | CC BY-SA 4.0 |
| **Herbold et al. 2023** | 90 human student essays, mean 339 words, 90 topics, paired with GPT-3 and GPT-4 versions | github.com/sherbold/chatgpt-student-essay-study | **Apache-2.0**. Too small for distributions, good for validation |
| **M4 / M4GT-Bench** | Human and machine in the **same row**: `prompt, human_text, machine_text, model, source`. Domains include **arxiv and peerread** | github.com/mbzuai-nlp/M4 | Check repo. **Best structural fit for academic-domain pairs** |
| **RAID** | Over 10M documents, 11 domains, 11 models, 4 decoding strategies, 11 adversarial attacks | `pip install raid-bench`; HF `liamdugan/raid` | **MIT** |
| **Ghostbuster** | IvyPanda essays, Reuters, WritingPrompts, **plus redistributable copies of BAWE (4,332 files), ETS, Lang8, PELIC, TOEFL91** | github.com/vivek3141/ghostbuster-data | **CC BY 3.0**, unusually permissive |
| **MAGE** | 436,606 rows, 554 MB | HF `yaful/MAGE` | Apache-2.0 |
| **OpenLLMText** | 60,000 human entries from OpenWebText/Reddit **pre-2019**, plus 60k each from ChatGPT, PaLM, LLaMA-7B, GPT2-XL | zenodo.org/records/8285326 | **CC BY 4.0** |

---

## 10. Pre-2022 "definitely human" baselines

| Source | Size / date | License |
|---|---|---|
| C4 `en` | 305 GB, 364.9M docs, Common Crawl April 2019 snapshot | ODC-BY |
| OpenWebText | 8,013,769 docs, 39.77 GB, GPT-2 era so pre-2020 | CC0 on packaging |
| enwiki 2021 dumps | 335-357 GB per monthly item, 19 items on archive.org | CC BY-SA 4.0 |
| Project Gutenberg / common-pile | 71,810 docs, 26.2 GB | Public domain |
| arXiv pre-2022 | filename-sliceable to `2112` | arXiv default |
| Semantic Scholar release 2022-05-10 | earliest of 194 releases | ODC-BY |
| Common Pile v0.1 | 8 TB public-domain / openly licensed, 40+ components | Open per component |

---

## Practical recommendation

Four sources to actually build on, for per-genre human reference distributions with clean licenses and real bulk text:

1. **Academic published prose:** PMC OA via `s3://pmc-oa-opendata` (free, no key, per-article license filtering) plus arXiv `src/` tars sliced to `≤2112` for a hard pre-LLM cut.
2. **Student academic writing:** BAWE (6.5M words, 30 disciplines) plus USE (1.22M words, plain text) plus PERSUADE 2.0 (25k essays) plus ELLIPSE. All free, giving discipline × level × first-language strata. Add MICUSP by scraping its 829 view pages.
3. **General prose:** OpenWebText plus enwiki-2021 from archive.org plus common-pile Gutenberg.
4. **Paired human/AI for calibration:** M4 (human and machine in one row, includes arXiv and PeerRead) and RAID (MIT).

**Two time-sensitive items.** ICLE v3 goes free 2026-09-15, so wait rather than paying. The PMC FTP paths every tutorial references were deleted the week of 2026-08-24, so any pipeline written before then silently breaks.

**Two things to avoid.** COCA full text at $395-$1,395 has 5% of words deliberately removed, which corrupts sentence-level stylometry; use the free 8.9M-word sample. LOCNESS forbids third-party distribution outright, so it can seed a private distribution but nothing published.

**Licensing note for a commercial product.** Most student-writing corpora (BAWE, USE, PERSUADE, ELLIPSE) are **NonCommercial**. They are fine for measuring reference distributions and for research, but training a shipped commercial model on them needs legal review. The clean commercial-safe set is: PMC OA (per-article), arXiv (default license, link back), Gutenberg (public domain), OpenWebText (CC0), Dolma and peS2o (ODC-BY), RAID (MIT), Ghostbuster (CC BY 3.0), Herbold (Apache-2.0), OpenLLMText (CC BY 4.0).

---

## Sources

MICUSP https://elicorpora.info/ · BAWE/OTA https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2539 · PERSUADE https://github.com/scrosseye/persuade_corpus_2.0 · ELLIPSE https://github.com/scrosseye/ELLIPSE-Corpus · ICLE https://corpora.uclouvain.be/cecl/icle/ · LOCNESS https://www.learnercorpusassociation.org/resources/tools/locness-corpus/ · corpusdata https://www.corpusdata.org/purchase.asp · USE/OTA https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457 · arXiv bulk https://info.arxiv.org/help/bulk_data_s3.html · PMC AWS https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/ · PMC README https://pmc-oa-opendata.s3.amazonaws.com/README.txt · Wikimedia dumps https://dumps.wikimedia.org/enwiki/ · Gutenberg mirroring https://www.gutenberg.org/help/mirroring.html · CORE https://core.ac.uk/documentation/dataset · DART-Europe closure https://www.ucl.ac.uk/library/dart-europe-e-theses-portal-has-closed-down · arctic_shift https://github.com/ArthurHeitmann/arctic_shift · peS2o https://huggingface.co/datasets/allenai/peS2o · S2 datasets API https://api.semanticscholar.org/api-docs/datasets · Dolma https://huggingface.co/datasets/allenai/dolma · C4 https://huggingface.co/datasets/allenai/c4 · OpenWebText https://huggingface.co/datasets/Skylion007/openwebtext · HC3 https://huggingface.co/datasets/Hello-SimpleAI/HC3 · RAID https://github.com/liamdugan/raid · M4 https://github.com/mbzuai-nlp/M4 · Ghostbuster https://github.com/vivek3141/ghostbuster-data · Herbold https://github.com/sherbold/chatgpt-student-essay-study · MAGE https://huggingface.co/datasets/yaful/MAGE · OpenLLMText https://zenodo.org/records/8285326
