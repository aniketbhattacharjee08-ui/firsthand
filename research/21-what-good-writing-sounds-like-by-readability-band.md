# What Good Writing Sounds Like at Each Readability Band

Date 2026-09-07. Report 21 in the humanizer research series. Working files, the scoring script and every collected passage are in `.readband/` (`score.py`, `extract_passages.py`, `extract_more.py`, `passages/`, `ai_passages/`, `provenance.json`, `tables.md`).

**Question.** When the humanizer targets a Flesch Reading Ease (FRE) band, what does *good* human prose actually do inside that band, so that the target is a texture and not a number?

**Method in one line.** 46 passages of 140-300 words by skilled human writers (public domain or short fair-use excerpts) plus 60 open-access PMC paragraphs and 4 MICUSP student papers were scored with `textstat` for FRE, Flesch-Kincaid grade, mean sentence length, sentence-length CV, syllables per word and a set of regex texture features; 12 AI passages were generated with the local Qwen2.5-7B-Instruct model at matched topics and, where the model would cooperate, matched FRE. Each band then got a close reading.

---

## Findings up front

1. **FRE is a sentence-length meter with a syllable correction. It does not measure difficulty, quality or humanness.** Darwin's "entangled bank" peroration scores FRE 18 (grade 24). Didion's opening of "Goodbye to All That" scores 30, in the same band as a PMC oncology introduction, because one sentence runs 98 words. The Gettysburg Address scores 64 with an 86-word final sentence. Meanwhile a Qwen paragraph asked for "plain English" scored 83 with a mean sentence of 10 words and read like a child's composition. Within every band the human passages and the AI passages sit within a few FRE points of each other; what separates them is never on the FRE axis.

2. **The human passages are bursty in a specific way: one or two very long sentences carry the argument, and short sentences are placed, not sprinkled.** Holmes (FRE 52): 11, 32, 46, **80**, 10, 10, 19. Woolf (FRE 66): 24, **5**, 27, 53, 10, **67**, 26, 10. Turing (FRE 67): 9, 14, **76**, **4**, 26, 20, 22, 11, 24, 33. The short sentence is almost always a verdict on the long one that precedes it ("But this is absurd." "That would be a mere shadow of freedom." "And the war came."). The AI passages at the same FRE have CVs of 0.19-0.35 and, when they do contain a short sentence, it is an announcement ("But the language can get better."), not a verdict.

3. **Sentence-length CV rises with FRE in good human prose.** Very dense band: CV 0.19-0.41 (Darwin 0.19, Hume 0.22, Federalist 10 0.26). Dense academic: 0.21-0.74. Upper-undergraduate: 0.27-0.81. Essay: 0.31-0.81. Plain: 0.36-0.62. This is the opposite of what a humanizer that "adds burstiness" does, which is to add it uniformly. The dense bands earn their FRE with *consistently* long, heavily nominal sentences; the essay bands earn theirs with alternation. A humanizer should read the target band before deciding how much variance to inject.

4. **Dense human prose is dense because of noun phrases and coordinated parallel clauses, not because of connectives.** Across the 46 skilled passages there are 12 sentence-initial formal connectives in roughly 9,500 words, about 1.3 per thousand, and eight of the twelve are in the four student papers. Hume, Darwin, Mill, Gibbon, Madison, Einstein: zero to one each. They connect by repeating the grammatical subject ("These laws... Thus, from the war of nature..."), by demonstrative reference ("Such phraseology...", "That at any rate is the theory..."), and by beginning the next sentence on the object of the last. The AI passages average 1.0 connective openers per 200 words and put them where the human passages put a demonstrative.

5. **Every skilled passage contains at least one *specific* that the writer alone could supply**: a date (29 July 1943; 15 minutes past eight, 6 August 1945), a number (a hundred pounds alive, five for the tusks; 126 homicides in the Seven-Five in 1993, 44 last year), a name (Bill Smith, William Hughes, a slave named Eli; Toshiko Sasaki; Nerva, Trajan, Hadrian), an object (an old .44 Winchester; a DC-7 at the old Idlewild temporary terminal; the *Osaka Asahi*). Measured as proper nouns per thousand words, the human passages run 0-176 with a median near 30; the AI passages run 0-29 except when hallucinating a court docket. This is the single largest gap at matched FRE and it is not fixable by rewriting expression.

6. **The good human passages permit imperfection but not error.** What appears: a one-word fragment used as a verdict ("Power."), a parenthetical aside in the middle of an argument ("(Somehow it always seems worse to kill a large animal.)"), a repeated word for effect ("desperate city... desperate country... desperate things"), a comma splice in a period sentence (Thoreau), a sentence that ends on "etc." (a philosophy graduate student), an archaic spelling left alone ("missletoe"). What never appears: a wrong word, a subject-verb disagreement, a misspelled name, an error inside a quotation or a number.

7. **First person is normal in every band and it does argumentative work.** Einstein: "we shall raise this conjecture to the status of a postulate." Watson and Crick: "We wish to suggest", "In our opinion", "we shall not comment on it." Holmes: "I strongly believe." Turing: "I propose", "I shall replace the question." The MICUSP philosophy paper: "In this paper, I will argue against this conclusion." First person per thousand words in the skilled passages runs from 0 (Gibbon, Madison) to 107 (Douglass) with a median of about 20. The AI passages either use none (all research-style outputs) or use it for feeling ("I found myself", "My heart pounded", "I craved").

8. **Openings are concrete or contrarian; closings are facts, images or single-clause verdicts. None of the 46 skilled passages closes on a summary.** Orwell closes a paragraph on a definition ("Orthodoxy, of whatever colour, seems to demand a lifeless, imitative style."), Baldwin on a sardonic clause ("until the real thing comes along"), Douglass on a precise measurement ("about one hundred yards from the treading-yard where we were fanning"). Eleven of the twelve AI passages close on a forward-looking abstraction ("remains within our grasp", "opens the door to a deeper understanding", "essential for personal growth and happiness", "I knew I had done my job").

9. **Two writers can sit at the same FRE with opposite mechanisms.** Watson & Crick (FRE 53) is sixteen sentences of 11-29 words, CV 0.32, no sentence over 30: short-sentence density, because it is a report. Holmes in *Lochner* (FRE 53) is nine sentences from 8 to 64 words, CV 0.75: long-sentence argument punctured by aphorisms. A humanizer that infers a single "human shape" from FRE will be wrong for one of them. The band sets a ceiling on mean sentence length and syllable load; genre sets the shape inside it.

10. **The instruction-tuned model cannot hold a band.** Asked for "a serious literary essayist", Qwen produced FRE 11 (grade 18), denser than Hume. Asked for the same argument in "plain, mostly one- and two-syllable words", it produced FRE 83, mean sentence 10.4 words, 15 sentences, and the prose of a ten-year-old. Neither is Orwell's 50. The model has two registers, ornate and infantile, and readability instructions toggle between them rather than sliding along the axis. This is why a pipeline needs a deterministic band gate and not a prompt.

---

## 1. Method

**Bands.** Five, as briefed: very dense (FRE below 30), dense academic (30-45), upper-undergraduate and research-article (45-55), quality essay and long-form journalism (55-70), plain (above 70). Passages were assigned to bands by their *measured* FRE, not by where one would expect the author to sit. That produced several surprises which are themselves findings (Didion in the 30s, Darwin's peroration below 20, Turing in the high 60s).

**Passages.** Slices of 140-300 words, starting at a known phrase and running to the sentence boundary nearest the target length, cut from full texts downloaded with `curl` from Project Gutenberg, Gutenberg Australia, Wikisource, law.cornell.edu, avalon.law.yale.edu, orwellfoundation.com, abrahamlincolnonline.org and newyorker.com (`.readband/extract_passages.py`, `.readband/extract_more.py`). PMC paragraphs are the first body paragraph of 160+ words from each of the 60 local files in `data/raw/pmc` (see report 14). MICUSP papers were fetched from `micusp.elicorpora.info/view?pid=...` in the first pass of this work (four papers: PHI.G1.01.1, BIO.G0.15.1, HIS.G1.01.1, ENG.G0.22.1). Two passages could not be fetched and were transcribed from memory; they are flagged in the Sources section and their text should be checked before any external use. PERSUADE 2.0 score-6 essays were not downloaded (the corpus is a 25,000-essay CSV and report 10 already characterises score-6 essays statistically); the MICUSP papers stand in for high-graded student writing.

**Metrics.** `textstat` 0.7.13 for FRE, Flesch-Kincaid grade and syllable counts. Sentence splitting is a regex with an abbreviation guard (`v.`, `Mr.`, `No.`, `p.`, initials), checked by hand against every non-PMC passage. Sentence-length CV is population SD over mean. Texture features (share of sentences opening on The/This/It/In/There; sentence-initial connectives; sentence-initial And/But/So; subordinate-clause openers; first- and second-person tokens; questions; parentheses; numerals; mid-sentence capitalised words as a proxy for proper nouns; sentences of four words or fewer) are regex heuristics in `score.py::texture`. All numbers are in `.readband/tables.md`.

**AI contrasts.** Twelve paragraphs from `mlx-community/Qwen2.5-7B-Instruct-4bit` (temperature 0.7, top-p 0.9), each prompted on the topic of a specific human exemplar and, in the second batch, toward a target readability (`.readband/gen_ai_contrast.py`, `gen_ai_contrast2.py`). The model is the same family the pipeline uses, so its habits are the habits the pipeline has to undo.

**Caveats.** Passage-level FRE on 200 words is noisy: one 90-word sentence moves it by 15 points. The passages were chosen as famous or representative, so they over-sample peroration and opening paragraphs relative to the middle of an argument; the PMC paragraphs correct for that in the research-article bands. Regex texture features miscount in places (a passage that quotes Austen inherits her capitals). Nothing here is a population estimate; report 10 has those. This report is about what the good passages *do*.

---

## 2. Where the exemplars landed

| Band | FRE | Skilled human exemplars (measured FRE) | PMC / student comparators |
|---|---|---|---|
| A. Very dense | < 30 | Darwin, entangled bank (18.4); Holt essay, MICUSP history grad (25.1); Einstein 1905 intro (25.2); Federalist 10 opening (25.8); Darwin, natural selection scrutinising (26.8) | PMC8719469 intro (20.1); PMC8809424 (-3.8); PMC8752981 (5.7); PMC8783091 (7.7); PMC8928841 (20.4) |
| B. Dense academic | 30-45 | Didion, Goodbye to All That (30.5); Mill, On Liberty (32.1); Hersey, Hiroshima (34.7); Scalia, Morrison dissent (34.7); Gibbon ch. 1 (34.9); Jackson, Barnette (40.0); Orwell, defence of the indefensible (41.4); Hume, Relations of Ideas (41.5); Federalist 51 (44.4); Thoreau, live deliberately (44.8); Twain, the water as a book (44.9) | MICUSP biology (32.2); PMC8786337 (32.1); PMC8719469 body (36.4); PMC9244602 methods (44.4) |
| C. Upper-undergraduate / research article | 45-55 | MICUSP English on Austen (49.2); Orwell, opening of Politics and the English Language (49.6); Hume, Custom (52.0); Holmes, Abrams dissent (52.2); Holmes, Lochner dissent (53.0); Watson & Crick 1953 (53.0); Darwin, Struggle for Existence (54.6) | PMC8826890 (47.1) |
| D. Quality essay / long-form | 55-70 | Gladwell, Tipping Point opening (56.2); James, Pragmatism (56.2); Woolf, Death of the Moth (58.4); MICUSP philosophy grad (59.4); Douglass, learning to read (60.8); Gettysburg Address (64.1); Woolf, A Room of One's Own (66.3); Baldwin, Notes of a Native Son (66.6); Emerson, Self-Reliance (66.8); Turing 1950 (66.9); Lincoln, Second Inaugural (67.0); Douglass, opening (67.5); Orwell, tiny incident (67.6); Thoreau, quiet desperation (67.6); Orwell, scrupulous writer (68.9) | PMC8807128 (57.2); PMC8948479 (66.9) |
| E. Plain | > 70 | Douglass, made a man (80.0); Orwell, did not want to shoot (80.8); Orwell, pulled the trigger (80.9); Hemingway, in our time vignettes (80.9, 83.4) | PMC9072890 results (72.1) |

Two observations before the close readings. First, the whole of English literary and scientific prose lives between FRE 18 and 81 on these samples, and the celebrated passages cluster in 45-70. Second, the PMC corpus (60 paragraphs, median FRE 25, interquartile range 19-35) sits almost entirely in bands A and B, which means that for the product's core use case, academic prose, the relevant texture specs are A, B and C, and D and E matter for essays and humanities writing.

---

## 3. Band A: very dense (FRE below 30)

### 3.1 Exemplar metrics

| Passage | Words | Sents | FRE | FK grade | Mean SL | SL CV | Min/max | Syl/word | % 3+ syl | Commas /1k | ; | 1st person /1k | Proper nouns /1k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Darwin, entangled bank (1859) | 219 | 4 | 18.4 | 24.3 | 54.8 | 0.28 | 29/67 | 1.57 | 16.4 | 114 | 4 | 9 | 59 |
| PMC8719469, introduction (2022) | 124 | 6 | 20.1 | 14.8 | 20.7 | 0.40 | 7/31 | 1.97 | 25.0 | 32 | 0 | 16 | 56 |
| MICUSP HIS.G1.01.1, Holt essay | 172 | 9 | 25.1 | 14.5 | 19.1 | 0.41 | 7/34 | 1.92 | 26.7 | 105 | 0 | 0 | 76 |
| Einstein, 1905 introduction (tr. 1923) | 242 | 7 | 25.2 | 18.2 | 34.6 | 0.37 | 13/52 | 1.72 | 22.3 | 54 | 0 | 4 | 4 |
| Madison, Federalist 10 opening | 175 | 5 | 25.8 | 18.4 | 35.0 | 0.26 | 26/47 | 1.72 | 22.9 | 86 | 2 | 0 | 11 |
| Darwin, natural selection scrutinising | 190 | 4 | 26.8 | 21.3 | 47.5 | 0.19 | 34/59 | 1.56 | 15.8 | 84 | 3 | 32 | 0 |
| PMC8809424 (DFT functionals) | 167 | 7 | -3.8 | 19.7 | 23.9 | 0.18 | 19/33 | 2.20 | 37.1 | 54 | 0 | 12 | 60 |
| PMC8752981 (mutation signatures) | 285 | 9 | 5.7 | 16.7 | 31.7 | 0.36 | 12/52 | 2.04 | 34.4 | 42 | 5 | 0 | 42 |
| PMC8783091 (Ewing sarcoma) | 307 | 10 | 7.7 | 19.7 | 30.7 | 0.35 | 11/47 | 1.97 | 27.0 | 62 | 0 | 3 | 55 |

Sentence-length sequences: Darwin bank **66, 67, 29, 57**; Einstein **31, 13, 42, 41, 21, 52, 42**; Federalist 10 **27, 26, 29, 45, 47**; Darwin scrutinising **59, 50, 34, 47**; Holt **24, 19, 25, 17, 7, 34, 12, 11, 23**; PMC8719469 intro **14, 21, 7, 29, 31, 22**.

### 3.2 Close reading

Two routes produce an FRE below 30 and the exemplars split cleanly between them. The nineteenth-century route is *length*: Darwin and Madison average 35-55 words a sentence with ordinary vocabulary (1.56-1.72 syllables per word, 16-23 percent polysyllables). The modern route is *lexis*: the PMC introductions and the graduate history essay average 19-31 words but carry 1.9-2.2 syllables per word and 25-37 percent polysyllabic tokens ("etiopathogenesis", "asymptotically-corrected", "historicization"). Both are legitimate, and the humanizer should know which one the source text is on, because the repairs differ. A long-sentence text tolerates being cut; a heavy-lexis text does not tolerate being simplified (report 00 §4: simplifying vocabulary has high grade cost).

**Sentence shape.** Darwin's long sentences are not subordinated; they are *coordinated and listed*. The entangled bank sentence is a frame ("It is interesting to contemplate an entangled bank... and to reflect that these elaborately constructed forms... have all been produced by laws acting around us") with four parallel prepositional phrases hung inside it ("clothed with many plants of many kinds, with birds singing on the bushes, with various insects flitting about, and with worms crawling through the damp earth"). The next sentence is a verbless list: "These laws, taken in the largest sense, being Growth with Reproduction; Inheritance which is almost implied by reproduction; Variability from...; a Ratio of Increase so high as to lead to a Struggle for Life, and as a consequence to Natural Selection, entailing Divergence of Character and the Extinction of less-improved forms." Sixty-seven words, no finite main verb, four semicolons. This is exactly the phrasal density Biber describes (report 15): the complexity is in the noun phrases, and the clause structure is flat. Madison does the same with abstract nouns: "The instability, injustice, and confusion introduced into the public councils, have, in truth, been the mortal diseases under which popular governments have everywhere perished."

**Where the short sentence falls.** Darwin's 29-word "Thus, from the war of nature, from famine and death, the most exalted object which we are capable of conceiving, namely, the production of the higher animals, directly follows" is the short sentence of the paragraph, and it is the *conclusion*, marked by "Thus" and by a cascade of fronted prepositional phrases that delay the subject to word 14 and the verb to the last two words. Einstein's 13-word sentence is a command: "Take, for example, the reciprocal electrodynamic action of a magnet and a conductor." The Holt essay's 7-word sentence is a signpost: "Here, the author introduces his historicization project." The PMC introduction's 7-word sentence is the pivot: "However, treatment options for ET remain limited." In all four the short sentence changes the paragraph's direction; none is decorative.

**Concrete vs abstract.** Even at FRE 18, Darwin keeps concrete nouns in the frame ("birds singing on the bushes", "worms crawling through the damp earth", "leaf-eating insects green, and bark-feeders mottled-grey; the alpine ptarmigan white in winter, the red-grouse the colour of heather") and reserves abstraction for the Capitalised Laws. Einstein has a magnet and a conductor before he has "the phenomena of electrodynamics". The PMC introduction has "primidone" and "propranolol" in sentence four. Madison is the exception, and it is the least readable passage in the set: nothing concrete for 175 words.

**Verbs.** The dense human passages have strong finite verbs at the ends of long noun phrases: "have all been *produced*", "have everywhere *perished*", "directly *follows*", "*leads* to asymmetries", "*suggest* that the phenomena of electrodynamics as well as of mechanics possess no properties corresponding to the idea of absolute rest". The PMC paragraphs lean on "is", "are associated with", "has been demonstrated", but even they have "thrust RNA-based therapeutics into the mainstream" (PMC8786337).

**Transitions.** Darwin: "Thus", once, as a conclusion marker, and demonstrative pick-up ("These laws"). Einstein: "For if the magnet is in motion... But if the magnet is stationary... In the conductor, however, we find..." A conditional pair and one mid-sentence "however". Madison: "He will not fail, therefore," mid-sentence. Nobody opens a sentence with "Furthermore" or "Moreover". The PMC introduction opens one sentence with "However" and one with "Approximately one-third of the ET patients", which is how a research article does it: a hedge or a number, not a connective. The Holt essay, the one student text in the band, has two sentence-initial connectives in nine sentences ("First, marking entails...", "Secondly, and perhaps more significantly, marking occasions..."), and that enumerative pair is the one place where it reads like a student.

**Imperfections permitted.** Darwin's second sentence has no main verb. Einstein has a dangling participial aside set off by dashes in the middle of a clause: "which gives rise—assuming equality of relative motion in the two cases discussed—to electric currents of the same path and intensity". The Holt essay has the word "marking" seven times in 172 words; repetition of the key term is a feature, not a fault, in a paper about a concept. Madison's "have, in truth, been" puts an aside between auxiliary and participle. None of these would survive a copy-editor bot; all of them are in the canon.

**Openings and closings.** Darwin opens with a picture and closes with a fact stated as a paradox: from "the war of nature, from famine and death" follows "the production of the higher animals". Einstein opens with what is "known" and closes with a conjecture stated as an observation. Madison opens on the thesis noun ("advantages") and closes on a concession ("it would be an unwarrantable partiality, to contend that they have as effectually obviated the danger... as was wished and expected"). The PMC introduction opens with a definition and closes with a statistic and a hedged cause ("Approximately one-third... probably due to..."). Nothing summarises.

**Stance and first person.** Einstein: "we find", "the phenomena... suggest". Darwin: "we are apt to consider", "we must believe". First person plural for observation and inference, never for feeling. The PMC intro: "our understanding". Holt essay: none, and it is poorer for it; the writer hides behind "the author" and "Holt" as subjects for nine sentences.

### 3.3 What band A is not

It is not the band where humanizers should inject short sentences. The human CV here is 0.18-0.41 and the passages with the *lowest* CV (Darwin scrutinising 0.19, PMC8809424 0.18) are competent professional prose. The band's texture is long parallel structures, heavy nominal groups, a single pivot sentence, and concrete nouns kept alive inside the abstraction.

---

## 4. Band B: dense academic (FRE 30-45)

### 4.1 Exemplar metrics

| Passage | Words | Sents | FRE | FK grade | Mean SL | SL CV | Min/max | Syl/word | % 3+ syl | Commas /1k | ; : dash | 1st person /1k | Proper nouns /1k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Didion, Goodbye to All That (1967) | 179 | 3 | 30.5 | 23.8 | 59.7 | 0.57 | 15/98 | 1.37 | 8.4 | 61 | 0 0 0 | 67 | 56 |
| Mill, On Liberty ch. 1 (1859) | 222 | 7 | 32.1 | 16.7 | 31.7 | 0.44 | 12/53 | 1.68 | 16.7 | 77 | 2 1 0 | 5 | 63 |
| MICUSP BIO.G0.15.1, invasive species | 228 | 13 | 32.2 | 12.8 | 17.5 | 0.35 | 9/27 | 1.87 | 25.9 | 39 | 0 0 0 | 0 | 79 |
| Hersey, Hiroshima opening (1946) | 165 | 2 | 34.7 | 18.7 | 82.5 | 0.21 | 65/100 | 1.54 | 13.9 | 97 | 3 0 0 | 0 | 176 |
| Scalia, Morrison v. Olson dissent (1988) | 195 | 7 | 34.7 | 13.7 | 27.9 | 0.74 | 1/52 | 1.75 | 23.1 | 67 | 0 1 3 | 0 | 149 |
| Gibbon, Decline and Fall ch. 1 (1776) | 230 | 8 | 34.9 | 15.5 | 28.8 | 0.51 | 12/56 | 1.69 | 20.0 | 70 | 3 1 0 | 0 | 43 |
| Jackson, Barnette (1943) | 220 | 9 | 39.4 | 13.8 | 24.4 | 0.41 | 8/45 | 1.69 | 18.2 | 36 | 0 0 0 | 32 | 9 |
| Orwell, defence of the indefensible (1946) | 218 | 9 | 41.4 | 13.5 | 24.2 | 0.50 | 10/51 | 1.67 | 14.7 | 55 | 0 3 0 | 9 | 32 |
| Hume, Relations of Ideas (1748) | 230 | 9 | 41.5 | 13.8 | 25.6 | 0.22 | 17/36 | 1.65 | 19.6 | 78 | 3 0 0 | 4 | 35 |
| Madison, Federalist 51 | 202 | 10 | 44.4 | 12.1 | 20.2 | 0.51 | 7/39 | 1.68 | 20.8 | 50 | 2 1 0 | 0 | 0 |
| Thoreau, live deliberately (1854) | 213 | 4 | 44.8 | 20.2 | 53.2 | 0.51 | 26/99 | 1.28 | 7.0 | 99 | 2 0 0 | 47 | 14 |
| PMC8719469, primidone pharmacology | 307 | 14 | 36.4 | 12.8 | 21.9 | 0.39 | 8/44 | 1.77 | 20.2 | 46 | 1 0 0 | 0 | 45 |

Sequences: Scalia **7, 1, 48, 47, 7, 52, 33**; Orwell **14, 51, 14, 29, 31, 32, 17, 10, 20**; Jackson **22, 27, 34, 26, 23, 14, 8, 21, 45**; Federalist 51 **38, 20, 7, 15, 20, 14, 9, 16, 39, 24**; Hume **24, 24, 25, 17, 22, 23, 34, 36, 25**; Gibbon **26, 14, 17, 12, 31, 27, 56, 47**; Didion **15, 66, 98**; Hersey **65, 100**.

### 4.2 Close reading

This band contains the widest spread of *kinds* of writing in the study: a Supreme Court majority, a dissent, two eighteenth-century political philosophers, a historian, a novelist's memoir, a New Yorker report, Thoreau, Orwell and a final-year biology student. That is itself the lesson. FRE 30-45 is where most published argumentative prose lives when it is not being deliberately plain, and the writers reach it by entirely different means.

**The two ways to get here.** Didion and Hersey reach FRE 30-35 with the *simplest* vocabulary in the whole study (1.37 and 1.54 syllables per word; 8 and 14 percent polysyllables) because their sentences are 60-100 words long. Hersey's opening is two sentences, 65 and 100 words, and the 100-word one is a semicolon chain of five people at the same instant: "At that same moment, Dr. Masakazu Fujii was settling down cross-legged to read the *Osaka Asahi* on the porch of his private hospital...; Mrs. Hatsuyo Nakamura, a tailor's widow, stood by the window of her kitchen...; Father Wilhelm Kleinsorge, a German priest of the Society of Jesus, reclined in his underwear on a cot..." Everything is a name, a place, a posture and an object. The student biology paper reaches the same FRE from the other side, with 17-word sentences and 26 percent polysyllables ("extirpation", "ramifications", "bioinvasions"). Between them, Mill, Gibbon, Jackson, Hume and Orwell sit at 24-32 words and 1.65-1.75 syllables per word, which is the register the pipeline's academic users mostly write in.

**Sentence shape and rhythm.** The characteristic move of this band is the *period sentence followed by the aphorism*. Scalia: 48 and 47 words of constitutional allocation and Federalist citation, then "But this wolf comes as a wolf." (7). Jackson: a 34-word sentence on patriotism and free minds, then five sentences later "That would be a mere shadow of freedom." (8), then a 45-word closing sentence that everyone quotes. Madison: 38 words on constitutional means and personal motives, then "Ambition must be made to counteract ambition." (7) and "If men were angels, no government would be necessary." (9). Orwell: 51 words of atrocities, then the colon-and-label pattern repeated three times, "this is called *pacification*... this is called *transfer of population*... this is called *elimination of unreliable elements*", then "Such phraseology is needed if one wants to name things without calling up mental pictures of them." The short sentence is not a break from the argument; it *is* the argument, compressed. And it is never first. Every aphorism in this band follows the long sentence it judges.

Hume is the counter-case: nine sentences from 17 to 36 words, CV 0.22, the flattest rhythm in the band. He is defining terms, and definitional prose is even. This is why a CV floor should be genre-conditional: a philosophy passage that is setting out a distinction will legitimately sit at 0.2.

**Noun phrases vs clauses.** Mill's opening is one 45-word sentence that is grammatically a single noun phrase with appositions: "The subject of this Essay is not the so-called Liberty of the Will... but Civil, or Social Liberty: the nature and limits of the power which can be legitimately exercised by society over the individual." Then a 40-word *fragment*, a noun phrase with a relative clause and no main verb: "A question seldom stated, and hardly ever discussed, in general terms, but which profoundly influences the practical controversies of the age by its latent presence, and is likely soon to make itself recognised as the vital question of the future." Gibbon's paragraph is eight sentences with paired abstract nouns and paired adjectives in nearly every one ("ancient renown and disciplined valor", "enjoyed and abused the advantages of wealth and luxury", "the virtue and abilities of Nerva, Trajan, Hadrian, and the two Antonines"). The density is nominal. The biology student, by contrast, writes finite clause after finite clause ("This study was carried out in isolated mountain pools. With the exception of the introduced trout, these ecosystems have had minimal disturbance and human contact. This is quite different from...") and it reads as younger, though the FRE is the same as Mill's.

**Concrete vs abstract.** The best passages anchor every abstraction to a case within one sentence. Orwell's "euphemism, question-begging and sheer cloudy vagueness" is followed at once by "Defenceless villages are bombarded from the air, the inhabitants driven out into the countryside, the cattle machine-gunned, the huts set on fire with incendiary bullets". Madison's "auxiliary precautions" comes after the angels. Jackson's "occasional eccentricity and abnormal attitudes" refers to the actual children in the actual case ("those we deal with here"). Scalia has EPA documents, a Superfund hearing, a memorandum of 30 November 1982. Hume is the abstract exception, and he compensates with worked examples in italics: "*That the square of the hypothenuse is equal to the square of the two sides*". The student paper is good on this: French Polynesia, *Euglandina rosea*, *Achatina fulica*, Hawaii, the Partulids.

**Verb choice.** Concrete, often violent, in the essayists: "bombarded", "driven out", "machine-gunned", "robbed", "trudging", "shot in the back of the neck". Latinate and stative in the philosophers: "consists in giving", "must be made commensurate", "renders our experience useful". Jackson mixes them: "transcends constitutional limitations", "invades the sphere of intellect and spirit". A humanizer should not de-Latinize Madison; it should notice whether the source is Orwell-type or Madison-type and keep the verb register consistent.

**Transitions.** Twelve passages, roughly 2,600 words, four sentence-initial connectives total, two of them in the student paper ("However, studies like this..."; "Insufficient understanding..." is not one). Jackson's "Nevertheless, we apply the limitations of the Constitution" is the one formal connective in a Supreme Court passage, and it opens sentence two. The dominant transitional devices are: demonstrative subject ("Such phraseology", "These slaves", "This bread", "That would be"), repetition of the previous sentence's key noun ("Ambition must be made to counteract ambition"), sentence-initial "But" (Scalia, Jackson, Madison, Mill, Hume all use it), and a question ("But what is government itself, but the greatest of all reflections on human nature?").

**Imperfections.** Mill's forty-word fragment. Scalia's one-word sentence "Power." Thoreau's "I did not wish to live what was not life, living is so dear" is a comma splice, and it is in one of the most quoted paragraphs in American literature. Didion's "can never cut through the ambiguities and second starts and broken resolves" uses polysyndeton where a style guide wants commas. Hersey's "reclined in his underwear on a cot on the top floor of his order's three-story mission house" stacks five prepositional phrases. Orwell repeats "this is called" three times. The biology student writes "one that's already threatened by an exotic", a contraction in a paper, and "in peoples' backyards", an apostrophe error; those are the imperfections of a real student and the pipeline should *not* introduce them (report 00 §4).

**Openings and closings.** Openings state the paradox or the picture: "In our time, political speech and writing are largely the defence of the indefensible." "The case is made difficult not because the principles of its decision are obscure but because the flag involved is our own." "It is easy to see the beginnings of things, and harder to see the ends." Closings are aphorisms or cases, never restatements: "The 14th Amendment does not enact Mr. Herbert Spencer's Social Statics" is in the next band, but this band's closings are of the same kind: "He cannot say outright, 'I believe in killing off your opponents when you can get good results by doing so'." "The prayers of both could not be answered." "Hawaii and French Polynesia are both pacific archipelagos, but their differing ecosystems led to the tragedy of the Partulids." The student closes on a case; that is a good student.

**Stance.** Jackson: "we apply", "We can have", "we deal with here", "We think". Institutional first-person plural doing judicial work. Thoreau: 47 first-person tokens per thousand, but every one attached to an action or a decision ("I went", "I wished", "I wanted to live deep"). Orwell: "one wants", "Consider for instance", second-person imperatives. Mill and Gibbon: no first person; authority by syntax. Didion: dense first person with body-level specifics ("the nerves in the back of my neck constrict").

### 4.3 The band's signature

Long-then-short. A period sentence of 35-55 words carrying parallel noun phrases or a list, then a sentence of 7-10 words that passes judgment on it. Concrete cases inside one sentence of every abstraction. Demonstratives and repeated nouns rather than connectives. First person, when present, for institutional or argumentative acts.

---

## 5. Band C: upper-undergraduate and research article (FRE 45-55)

### 5.1 Exemplar metrics

| Passage | Words | Sents | FRE | FK grade | Mean SL | SL CV | Min/max | Syl/word | % 3+ syl | Commas /1k | ; : ( | 1st person /1k | Proper nouns /1k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MICUSP ENG.G0.22.1, Austen's Persuasion | 221 | 7 | 49.2 | 14.2 | 31.6 | 0.64 | 12/75 | 1.48 | 11.8 | 90 | 0 2 3 | 0 | 54 |
| Orwell, opening of Politics and the English Language | 227 | 10 | 49.6 | 12.0 | 22.7 | 0.34 | 8/35 | 1.59 | 13.7 | 31 | 0 1 0 | 35 | 9 |
| Hume, Custom (1748) | 168 | 6 | 52.0 | 13.0 | 28.0 | 0.58 | 9/61 | 1.49 | 10.1 | 71 | 0 0 0 | 65 | 0 |
| Holmes, Abrams dissent (1919) | 208 | 7 | 52.2 | 13.3 | 29.7 | 0.81 | 10/80 | 1.48 | 13.0 | 29 | 0 0 0 | 19 | 5 |
| Holmes, Lochner dissent (1905) | 227 | 9 | 53.0 | 11.5 | 25.2 | 0.75 | 8/64 | 1.55 | 16.3 | 53 | 0 0 0 | 40 | 35 |
| Watson & Crick, Nature 1953 | 282 | 16 | 53.0 | 10.2 | 17.6 | 0.32 | 11/29 | 1.60 | 14.5 | 39 | 0 1 5 | 25 | 32 |
| Darwin, Struggle for Existence | 199 | 6 | 54.6 | 13.9 | 33.2 | 0.27 | 21/43 | 1.40 | 10.1 | 60 | 0 0 1 | 10 | 10 |
| PMC8826890 (COVID public-health measures) | 228 | 15 | 47.1 | 10.5 | 15.2 | 0.41 | 8/32 | 1.71 | 21.5 | 75 | 0 0 0 | 0 | 40 |

Sequences: Holmes Abrams **11, 32, 46, 80, 10, 10, 19**; Holmes Lochner **18, 25, 35, 48, 8, 9, 64, 11, 9**; Watson & Crick **16, 11, 13, 12, 20, 28, 29, 13, 12, 20, 17, 16, 12, 23, 17, 23**; Orwell **35, 18, 25, 23, 32, 24, 24, 13, 25, 8**; Darwin **42, 23, 31, 39, 43, 21**; Austen essay **40, 12, 37, 18, 24, 15, 75**.

### 5.2 Close reading

This is the band the product should treat as home for humanities and social-science coursework, and it contains the clearest demonstration in the study that FRE does not fix shape. Watson & Crick and Holmes in *Lochner* both score 53.0. One is sixteen sentences of 11-29 words with no sentence over 30 (CV 0.32); the other is nine sentences from 8 to 64 (CV 0.75). Both are among the best-written documents of their century.

**Two shapes at one score.** Watson & Crick is *report* shape: every sentence does one thing and the sentence order is the order of the reasoning. "We wish to suggest a structure for the salt of deoxyribose nucleic acid (D.N.A.). This structure has novel features which are of considerable biological interest. A structure for nucleic acid has already been proposed by Pauling and Corey. They kindly made their manuscript available to us in advance of publication." Subject-verb-object, subject repeated from the previous object, no subordination deeper than one relative clause. Its evenness is a virtue: the reader is never asked to hold two things. Holmes is *argument* shape: the 80-word sentence in *Abrams* is one idea unfolding ("But when men have realized that time has upset many fighting faiths, they may come to believe even more than they believe the very foundations of their own conduct that the ultimate good desired is better reached by free trade in ideas—that the best test of truth is the power of the thought to get itself accepted in the competition of the market, and that truth is the only ground upon which their wishes safely can be carried out."), and the three sentences that follow are 10, 10 and 19 words: "That at any rate is the theory of our Constitution. It is an experiment, as all life is an experiment. Every year if not every day we have to wager our salvation upon some prophecy based upon imperfect knowledge." The long sentence earns the short ones. Orwell's opening paragraph is between the two: ten sentences, mostly 18-35 words, the shortest last ("The point is that the process is reversible.", 8 words).

**Where the short sentences are.** In every argumentative passage in this band the shortest sentence is either the paragraph's last (Orwell, 8 words) or immediately follows the longest (Holmes twice, Hume: "Custom, then, is the great guide of human life." opens, then a 32-word sentence). In *Lochner* the pattern repeats twice inside one paragraph: 48 then 8 and 9 ("Sunday laws and usury laws are ancient examples. A more modern one is the prohibition of lotteries."), 64 then 11 and 9 ("The 14th Amendment does not enact Mr. Herbert Spencer's Social Statics. The other day we sustained the Massachusetts vaccination law."). The short sentences are *examples and verdicts*, the long ones are *principles*. A humanizer that shortens randomly will put the verdict before the principle.

**Noun-phrase vs clause density.** Lower than bands A and B. Holmes and Orwell run at 1.48-1.59 syllables per word and 13-16 percent polysyllables, with finite clauses doing the work ("If you have no doubt of your premises or your power and want a certain result with all your heart you naturally express your wishes in law and sweep away all opposition"). Watson & Crick is 1.60 with the technical nouns ("phosphate diester groups joining beta-D-deoxyribofuranose residues with 3',5' linkages") concentrated in two sentences and the rest plain. Darwin's *Struggle for Existence* paragraph is 1.40 syllables per word and 10 percent polysyllables, in a passage that is defining a technical term; he does it with canines, a plant on the edge of a desert, mistletoe and apple trees.

**Concrete vs abstract.** This band is where good writers put one concrete instance per abstraction *as a rule*, not as decoration. Orwell: language decline is abstract, so "A man may take to drink because he feels himself to be a failure, and then fail all the more completely because he drinks." Hume: custom is abstract, so "we should be entirely ignorant of every matter of fact beyond what is immediately present to the memory and senses." Holmes: liberty of contract is abstract, so "Sunday laws and usury laws", "the prohibition of lotteries", "the Postoffice", "the Massachusetts vaccination law". The Austen essay quotes the novel's "the last smiles of the year upon the tawny leaves and withered hedges" to ground its claim about autumn. PMC8826890 is the weak text in this band precisely because it stays general ("simple but effective strategies became popular"; "People should avoid eye, nose, and mouth contact") and has fifteen sentences with no case, no number and no name except WHO and CDC.

**Verbs.** Holmes: "sweep away", "upset", "wager". Orwell: "take to drink", "fail all the more completely". Watson & Crick: "wish to suggest", "kindly made", "repel", "coiled round". Darwin: "struggle", "languish and die". The verbs are physical even when the nouns are not. The PMC comparator: "became popular", "were made mandatory", "is recommended" five times.

**Transitions.** Orwell in ten sentences: "Now, it is clear that..." (a spoken-register pivot), "But an effect can become a cause...", "It is rather the same thing that is happening...", "The point is that...". Holmes: "But when men have realized...", "That at any rate is the theory...". Watson & Crick: "In our opinion, this structure is unsatisfactory for two reasons: (1)... (2)...", "Another three-chain structure has also been suggested by Fraser", "We wish to put forward a radically different structure". Ordinal enumeration is used in a numbered list, not as a sentence-initial adverb. The student Austen essay is the one text with sentence-initial "However" twice in seven sentences, and it is the one text that reads as student work.

**Imperfections.** Orwell's "so the argument runs" is a parenthetical aside between dashes. Holmes writes "whole heartedly" as two words and "Postoffice" as one. Watson & Crick's numbered reasons are inside a running sentence, "(1) We believe that the material which gives the X-ray diagrams is the salt, not the free acid... (2) Some of the van der Waals distances appear to be too small." Darwin spells "missletoe". The Austen essay ends a paragraph on a block quotation with no comment after it. Hume writes "at once" where a modern writer would write "immediately" and nobody minds.

**Openings and closings.** Openings state a claim the writer will complicate: "Most people who bother with the matter at all would admit that the English language is in a bad way, but it is generally assumed that we cannot by conscious action do anything about it." "Persecution for the expression of opinions seems to me perfectly logical." "This case is decided upon an economic theory which a large part of the country does not entertain." Each is a plain declarative that a reader can disagree with, which is what makes the paragraph an argument. Closings: a fact (the vaccination law), a maxim (the process is reversible), a definition (opposite directions of the two chains). No paragraph closes by restating its opening.

**Stance and first person.** Very high in this band and always for intellectual acts. Holmes: "seems to me", "I should desire to study it further", "I do not conceive that to be my duty", "I strongly believe". Watson & Crick: "We wish to suggest", "In our opinion", "We believe", "we shall not comment on it", "We have made the usual chemical assumptions". Hume: "we should be entirely ignorant". Orwell: "our civilization", "we cannot by conscious action". The MICUSP English essay uses none, and it hedges instead with "Austen may have seen herself" and "presumably".

### 5.3 The band's signature

Either an even report of 12-29-word sentences (science) or a principle-then-example argument where the shortest sentences are examples and verdicts (humanities, law). One concrete case per abstraction. Physical verbs. First person for claims. No sentence-initial connectives beyond an occasional "But" or "Now".

---

## 6. Band D: quality essay and long-form journalism (FRE 55-70)

### 6.1 Exemplar metrics

| Passage | Words | Sents | FRE | FK grade | Mean SL | SL CV | Min/max | Syl/word | % 3+ syl | Commas /1k | ; dash | 1st person /1k | Proper nouns /1k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gladwell, Tipping Point opening (1996) | 140 | 4 | 56.2 | 12.3 | 35.0 | 0.25 | 23/48 | 1.44 | 8.6 | 79 | 0 1 | 0 | 136 |
| James, Pragmatism Lecture II (1907) | 186 | 9 | 56.2 | 8.9 | 20.7 | 0.31 | 12/34 | 1.59 | 17.2 | 27 | 1 3 | 11 | 22 |
| Woolf, Death of the Moth (1942) | 253 | 8 | 58.4 | 13.0 | 31.6 | 0.47 | 14/61 | 1.38 | 7.9 | 59 | 3 0 | 4 | 0 |
| MICUSP PHI.G1.01.1, statue and clay | 228 | 10 | 59.4 | 10.6 | 22.8 | 0.31 | 9/34 | 1.47 | 14.0 | 75 | 0 0 | 22 | 66 |
| Douglass, learning to read (1845) | 215 | 7 | 60.8 | 12.4 | 30.7 | 0.46 | 11/57 | 1.35 | 6.5 | 70 | 4 1 | 107 | 5 |
| Douglass, Covey breaks him | 211 | 11 | 63.9 | 9.1 | 19.2 | 0.58 | 6/39 | 1.46 | 10.4 | 90 | 2 0 | 90 | 19 |
| Lincoln, Gettysburg (1863) | 278 | 10 | 64.1 | 11.0 | 27.8 | 0.73 | 10/86 | 1.33 | 7.2 | 79 | 0 7 | 54 | 7 |
| Orwell, the dead coolie | 223 | 10 | 65.6 | 9.1 | 22.3 | 0.29 | 10/31 | 1.43 | 9.4 | 63 | 2 0 | 27 | 18 |
| Woolf, A Room of One's Own opening (1929) | 222 | 8 | 66.3 | 10.9 | 27.8 | 0.74 | 5/67 | 1.32 | 6.3 | 41 | 6 1 | 36 | 59 |
| Baldwin, Notes of a Native Son (1955)* | 223 | 10 | 66.6 | 9.5 | 22.3 | 0.40 | 10/39 | 1.39 | 8.1 | 94 | 1 0 | 72 | 22 |
| Emerson, Self-Reliance (1841) | 228 | 9 | 66.8 | 10.2 | 25.3 | 0.68 | 8/69 | 1.36 | 11.4 | 57 | 4 0 | 13 | 4 |
| Turing, 1950 opening | 239 | 10 | 66.9 | 9.3 | 23.9 | 0.81 | 4/76 | 1.39 | 9.2 | 38 | 0 0 | 13 | 71 |
| Lincoln, Second Inaugural (1865) | 229 | 13 | 67.0 | 8.3 | 17.6 | 0.57 | 6/38 | 1.44 | 10.9 | 44 | 1 0 | 9 | 39 |
| Douglass, opening of the Narrative | 224 | 14 | 67.5 | 7.8 | 16.0 | 0.42 | 6/36 | 1.46 | 10.7 | 71 | 0 0 | 76 | 45 |
| Orwell, a tiny incident (1936) | 228 | 9 | 67.6 | 10.1 | 25.3 | 0.41 | 9/45 | 1.34 | 8.3 | 57 | 0 0 | 57 | 9 |
| Thoreau, quiet desperation (1854) | 169 | 11 | 67.6 | 7.6 | 15.4 | 0.70 | 7/47 | 1.46 | 9.5 | 47 | 0 0 | 12 | 0 |
| Orwell, a scrupulous writer (1946) | 199 | 13 | 68.9 | 7.5 | 15.3 | 0.53 | 5/34 | 1.45 | 9.5 | 50 | 0 0 | 20 | 10 |
| Twain, the boys' ambition (1883) | 369 | 9 | 53.4 | 15.0 | 41.0 | 1.48 | 6/208 | 1.37 | 5.7 | 76 | 13 1 | 33 | 27 |

\* Transcribed from memory; see Sources. Twain is listed here although his FRE is 53 because the passage is 369 words with a single 208-word sentence, and it is discussed below as the limiting case of what a human sentence can be.

Sequences: Turing **9, 14, 76, 4, 26, 20, 22, 11, 24, 33**; Woolf Room **24, 5, 27, 53, 10, 67, 26, 10**; Lincoln Second Inaugural **22, 8, 12, 38, 16, 19, 13, 19, 36, 8, 7, 6, 25**; Gettysburg **30, 24, 10, 27, 11, 22, 21, 21, 26, 86**; Orwell scrupulous **23, 5, 8, 9, 14, 8, 11, 18, 34, 18, 13, 27, 11**; Thoreau **9, 7, 22, 18, 11, 12, 47, 9, 11, 10, 13**; Emerson **69, 30, 17, 11, 17, 17, 27, 32, 8**; Baldwin **10, 13, 30, 23, 22, 12, 19, 22, 33, 39**; Twain **19, 10, 6, 12, 55, 17, 17, 19, 10, 208**.

### 6.2 Close reading

Eighteen passages, the largest band, and the one where the canon of English prose actually sits: Lincoln, Emerson, Thoreau, Douglass, Woolf, Orwell, Baldwin, Turing. The vocabulary is the simplest in the study (1.32-1.47 syllables per word; 6-11 percent polysyllables, against 20-37 percent in band A) and the sentence-length CV is the highest (median 0.53; six passages above 0.68). This is where "burstiness" is real, and the passages show what it is made of.

**The long sentence is a list or a build; the short sentence is a turn.** Turing's 76-word sentence is a single conditional with a Gallup poll at the end; the next sentence is "But this is absurd." (4). Woolf's 53-word sentence is a list of six things the lecture might have been ("a few remarks about Fanny Burney; a few more about Jane Austen; a tribute to the Brontës and a sketch of Haworth Parsonage under snow; some witticisms if possible about Miss Mitford; a respectful allusion to George Eliot; a reference to Mrs Gaskell and one would have done"); the next is "But at second sight the words seemed not so simple." (10). Lincoln's 38-word sentence sets both parties' aims; "And the war came." (4, in the full text) is the turn. Emerson opens on a 69-word sentence of four parallel "that" clauses and closes the paragraph on "It is a deliverance which does not deliver." (8). Gettysburg is the inverse: nine sentences of 10-30 words and then an 86-word closing period with four dashes, the long sentence being the *climax* rather than the setup. Twain's 208-word sentence is a semicolon inventory of a sleeping river town, twelve items long, and every item is a thing you can see ("a sow and a litter of pigs loafing along the sidewalk, doing a good business in watermelon rinds and seeds"). None of the long sentences in this band is long because of nested subordination. They are long because they enumerate or accumulate, and the reader can always tell where they are in the list.

**Two rhythms, both human.** Orwell's "scrupulous writer" paragraph and Thoreau's "quiet desperation" paragraph are the *staccato* variety: 13 and 11 sentences averaging 15 words, CV 0.53-0.70, with questions and aphorisms ("What am I trying to say? What words will express it?... Could I put it more shortly?"; "What is called resignation is confirmed desperation."). Woolf's moth and Gladwell's opening are the *flowing* variety: 4-8 sentences averaging 31-35 words, CV 0.25-0.47, with a controlled long build ("As you drive east on Atlantic Avenue, through the part of New York City that the Police Department refers to as Brooklyn North, the neighborhoods slowly start to empty out: the genteel brownstones of the western part of Brooklyn give way to sprawling housing projects and vacant lots."). Orwell's coolie paragraph, CV 0.29, is ten sentences between 10 and 31 words and it is plainly good prose. So even within the essay band a CV of 0.29 is human when the sentences are doing sequential narrative work. The pipeline's CV floor should not be applied to narrative paragraphs.

**Noun phrases vs clauses.** This band has the highest clause density and the lightest noun phrases. Douglass: "I used also to carry bread with me, enough of which was always in the house, and to which I was always welcome; for I was much better off in this regard than many of the poor white children in our neighborhood." Finite clause, relative clause, relative clause, "for" clause, a comparison. Baldwin: "On the 29th of July, in 1943, my father died. On the same day, a few hours later, his last child was born." Two clauses, four prepositional phrases, nine of the eleven words in the first sentence monosyllabic. Where the noun phrases are heavy they are *names*: "Father Wilhelm Kleinsorge, a German priest of the Society of Jesus"; "the Seventy-fifth Precinct, a 5.6-square-mile tract".

**Concrete vs abstract.** The concrete-to-abstract ratio is the highest of any band. Orwell's tiny incident has a phone call, a bazaar, a pony, "an old .44 Winchester and much too small to kill an elephant", a mahout "twelve hours' journey away". Baldwin has 29 July, 3 August, Detroit, Harlem, "a wilderness of smashed plate glass", his nineteenth birthday. Gladwell has 126 homicides and 44. Douglass has Bill Smith, William Hughes, Eli, three o'clock, one hundred yards. When these writers do go abstract they do it in the last sentence, once: "the real nature of imperialism", "the spoils of injustice, anarchy, discontent, and hatred", "Orthodoxy, of whatever colour, seems to demand a lifeless, imitative style." The abstraction is *earned by the preceding specifics* and is the paragraph's payoff. The MICUSP philosophy paper, the one student text here, inverts this: it is abstract throughout (Lump, Peter, modal properties) but it uses a worked example with times of day ("re-molded Lump into one big, shapeless lump at 5pm before the fire") which is what a good philosophy paper does.

**Verbs.** "rend", "wringing", "shooing", "clicking their tongues", "slobbered", "sagged", "suck out all the marrow", "put to rout", "drive life into a corner". Strong, physical, often in the past tense of narrative. Lincoln is the exception with stative verbs and it compensates with biblical rhythm.

**Transitions.** Almost none that a connective detector would catch. Baldwin's paragraph is bound by *dates* ("On the 29th of July... On the same day... Over a month before this... A few hours after... On the morning of the 3rd of August"). Woolf's by *"But"* three times in eight sentences. Lincoln's by *grammatical parallelism* ("Neither party expected... Neither anticipated... Each looked for... Both read the same Bible"). Orwell's scrupulous-writer paragraph by *pronoun shift* ("A scrupulous writer... he will probably ask himself... But you are not obliged... You can shirk it"). Thoreau's by *repeating a word* ("desperation" four times, "desperate" three). Gettysburg by the demonstrative "that" ("that nation", "that war", "that field", "that cause"). This is what report 00 §4 means by stripping connectives with no grade cost: the good writers never needed them.

**Imperfections.** Frequent and characteristic. Turing: "this attitude is dangerous, If the meaning of the words..." (a comma where a full stop belongs, in the published text). Orwell: a whole sentence in parentheses, "(Somehow it always seems worse to kill a large animal.)", and a sentence beginning "Besides,". Woolf: a rhetorical question addressed to the audience ("what, has that got to do with a room of one's own?"), "I will try to explain." (5 words), an em-dash aside. Baldwin: "very well, life seemed to be saying, here is something that will certainly pass for an apocalypse until the real thing comes along", speech rhythm inside an essay. Douglass: an exclamation ("and behold a man transformed into a brute!"). Emerson: "Not for nothing one face, one character, one fact, makes much impression on him, and another none." (inverted syntax, a subject list treated as singular). Thoreau: "From the desperate city you go into the desperate country, and have to console yourself with the bravery of minks and muskrats." The imperfections are all *rhetorical*: fragments, asides, exclamations, addressed questions, repetitions, comma-spliced series. None is a spelling, agreement or word-choice error.

**Openings and closings.** Openings in this band are frequently *tiny*: "Custom, then, is the great guide of human life." "The mass of men lead lives of quiet desperation." "Mr. Covey succeeded in breaking me." "I propose to consider the question, 'Can machines think?'" "On the 29th of July, in 1943, my father died." Ten words or fewer, a hard fact or a thesis, and then the paragraph earns it. Or the opening is a direct address that pretends the argument has already started: "But, you may say, we asked you to speak about women and fiction". Closings are images or sardonic clauses: "shrouded ghosts, to terrify and torment me with thoughts of my wretched condition"; "here is something that will certainly pass for an apocalypse until the real thing comes along"; "evidently there was something that the children ought not to have seen"; "No way of thinking or doing, however ancient, can be trusted without proof." The MICUSP philosopher closes on the thesis ("it is not the case that Peter and Lump have different modal properties"), which is right for a paper's introduction.

**Stance.** Highest first-person density in the study (Douglass 107 and 90 per thousand, Baldwin 72, Orwell 57-66, Gettysburg 54 as "we"). But it is first person as *witness and agent*: "I got on to a pony and started out", "I converted into teachers", "I had declined to believe in that apocalypse", "I propose", "I shall replace the question". Zero instances of "I think", "I feel", "in my opinion" across 18 passages. Second person appears as address to the reader in Woolf ("you may say", "you asked me") and Orwell ("You can shirk it") and Holmes; it is a move the essayists make once per paragraph, not a habit.

### 6.3 The band's signature

Simple words, wide sentence range, list-built long sentences and verdict short ones, a tiny opening, a concrete texture of names, dates and objects with one earned abstraction at the end, transitions by date, parallelism, repetition and "But", first person as agent, rhetorical imperfections and zero mechanical errors.

---

## 7. Band E: plain (FRE above 70)

| Passage | Words | Sents | FRE | FK grade | Mean SL | SL CV | Min/max | Syl/word | % 3+ syl | 1st person /1k |
|---|---|---|---|---|---|---|---|---|---|---|
| Douglass, a slave made a man (1845) | 221 | 11 | 80.0 | 7.1 | 20.1 | 0.36 | 10/33 | 1.26 | 4.5 | 63 |
| Orwell, I did not want to shoot (1936) | 227 | 14 | 80.8 | 6.0 | 16.2 | 0.41 | 7/28 | 1.30 | 7.5 | 66 |
| Orwell, I pulled the trigger (1936) | 211 | 13 | 80.9 | 6.0 | 16.2 | 0.62 | 3/37 | 1.29 | 5.2 | 28 |
| Hemingway, in our time ch. 1 (1924) | 112 | 11 | 80.9 | 4.9 | 10.2 | 0.74 | 3/26 | 1.37 | 8.0 | 39 |
| Hemingway, in our time ch. 5 (1924) | 129 | 11 | 83.4 | 4.6 | 11.7 | 0.42 | 3/19 | 1.32 | 6.2 | 0 |

Sequences: Orwell trigger **37, 28, 13, 22, 23, 3, 9, 9, 7, 26, 5, 8, 21**; Hemingway ministers **18, 8, 11, 3, 9, 8, 10, 18, 9, 16, 19**.

**The warning first.** Skilled expository prose almost never scores above 70 for a whole paragraph. Hemingway's own opening of *The Sun Also Rises* scores 68.8; Orwell's "Why I Write" opening 54.3; Douglass's opening 67.5; Lincoln 64-67. Everything in this study above 70 is *narrative action*: a shooting, a fight in a wheat-field, a firing squad. A humanizer asked to put academic content into band E is being asked for something no good writer does, and the pipeline should refuse it or cap at about 70.

**What the plain passages do.** The short sentences are physical facts at the moment of consequence: "His mouth slobbered." "It rained hard." "I fired a third time." "He sat down in a puddle of water." Hemingway's firing-squad vignette is eleven sentences of 3-19 words, the longest a single coordinated chain ("They tried to hold him up against the wall but he sat down in a puddle of water"), and its 3-word sentence is "It rained hard."; Orwell's shooting paragraph spends 90 words on five seconds and its five shortest sentences all fall inside them. Douglass reaches FRE 80 with sentences averaging 20 words by coordination and semicolons ("I broke down; my strength failed me; I was seized with a violent aching of the head, attended with extreme dizziness; I trembled in every limb") and by names and quantities (Bill Smith, William Hughes, Eli, three o'clock, one hundred yards). Vocabulary is monosyllabic (1.26-1.37 syllables per word) but the *sentences are not uniformly short*: the CV is 0.36-0.74 and the maximum is 19-37. Feeling is reported once or never; Orwell's only emotional word is the crowd's "glee". Endings are mid-event. The AI paragraph at the same FRE (§8.4) has none of the compression, announces a feeling every 40 words, and ends on cheering.

---

## 8. AI at the same readability: what actually differs

Twelve Qwen2.5-7B-Instruct paragraphs were generated on the topics of specific exemplars (`.readband/ai_passages/`, prompts in `gen_ai_contrast.py` and `gen_ai_contrast2.py`; all twelve are labelled as machine-generated and none was edited). Three pairs land within a few FRE points of their human counterpart and are read closely here. The full AI metric and texture tables are in `.readband/tables.md`.

| Pair | Human | FRE | Mean SL | CV | Syl/w | Proper nouns /1k | AI | FRE | Mean SL | CV | Syl/w | Proper nouns /1k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Physics intro | Einstein 1905 | 25.2 | 34.6 | 0.37 | 1.72 | 4 | ai_dense_physics_asymmetry | 23.6 | 26.5 | 0.29 | 1.85 | 10 |
| Tremor intro | PMC8719469 intro | 20.1 | 20.7 | 0.40 | 1.97 | 56 | ai_research_intro_tremor | 15.4 | 21.4 | 0.34 | 2.01 | 29 |
| Free-speech dissent | Holmes, Abrams | 52.2 | 29.7 | 0.81 | 1.48 | 5 | ai_essay_free_speech | 56.2 | 23.1 | 0.22 | 1.50 | 16 |
| Shooting the elephant | Orwell, trigger | 80.9 | 16.2 | 0.62 | 1.29 | 0 | ai_plain_elephant | 80.5 | 14.6 | 0.34 | 1.32 | 0 |
| Cabin by a pond | Thoreau, desperation | 67.6 | 15.4 | 0.70 | 1.46 | 0 | ai_plain_walden | 61.2 | 22.5 | 0.19 | 1.45 | 0 |

### 8.1 Band A pair: Einstein (25.2) vs Qwen physics (23.6)

Same FRE, same topic, same first-person plural. Sentence sequences: Einstein **31, 13, 42, 41, 21, 52, 42**; Qwen **30, 36, 15, 18, 42, 19, 24, 22, 30, 26, 29**. Ranked by how much weight a supervised detector would probably put on each difference (largest first), with a fixability tag: **E** = fixable by rewriting expression, **C** = requires content the model does not have.

1. **Empty specificity (C).** Einstein has one apparatus, a magnet and a conductor, and states what is observed in each of two cases, with the quantity named ("an electric field with a certain definite energy"; "an electromotive force, to which in itself there is no corresponding energy"). Qwen reports "a series of experiments" it did not do, "empirical observations" it does not have, and "a puzzling discrepancy" it never names. Every noun that should be a measurement is a category.
2. **Announced stance vs performed stance (E, partly).** Qwen: "We have delved into the intricacies", "This reevaluation opens the door to a deeper understanding", "invites us to re-examine the very foundations". Einstein: "Take, for example,"; "we find"; "suggest that". The AI paragraph *tells you it is being profound*; the human paragraph *does the inference*. The lexical layer ("delved", "intricacies", "opens the door", "very foundations") is scrubbable; the habit of ending each sentence on an evaluation of its own importance is structural and survives lexical scrubbing.
3. **Where the short sentence sits (E).** Einstein's 13-word sentence is an imperative example in position two. Qwen's 15- and 18-word sentences are "However, when we apply these same principles to moving bodies, we encounter a puzzling discrepancy." and a restatement of it. The AI's short sentences are transitions; the human's are instances.
4. **Connective openers (E).** Qwen: "However," "For instance," "However," three in eleven sentences. Einstein: zero, with "For if", "But if" and a mid-sentence "however" instead.
5. **The two-paragraph summary shape (E).** Qwen's second paragraph re-proposes what the first said and ends on the widest abstraction available ("the nature of space and time"). Einstein's ends on the specific conjecture the paper will test.
6. **Uniform sentence length (E).** CV 0.29 vs 0.37 is a small gap here; the larger tell is that Qwen has *no* sentence over 42 words and Einstein has a 52-word one with a dash aside inside it.

Difference 1 dominates and is unfixable by paraphrase. A pipeline given the Qwen paragraph can fix 2-6 and will still produce a paragraph in which nothing happens to any specific object.

### 8.2 Band A pair: PMC8719469 introduction (20.1) vs Qwen tremor introduction (15.4)

Same topic, same register, near-identical mean sentence length (20.7 vs 21.4) and syllable load (1.97 vs 2.01). Sequences: human **14, 21, 7, 29, 31, 22**; Qwen **26, 13, 22, 19, 15, 16, 23, 37**.

1. **Numbers and named things (C).** Human: two drug names in sentence four, "only two frontline medications", a mechanism (GABA-ergic vs beta blocker), "(i)... (ii)...", "Approximately one-third of the ET patients discontinue". Qwen: "a significant subset", "range from mild to severe", "Genetic polymorphisms, pharmacokinetic differences, and drug interactions have been proposed" (by no one, for nothing in particular). The human paragraph would let a reader check a claim; the AI paragraph would not. Proper nouns per thousand: 56 vs 29, and the AI's are all "ET" and "Primidone".
2. **The aims sentence (E).** Qwen closes with "This study aims to explore the prevalence and correlates of poor tolerability... with the goal of identifying biomarkers or clinical predictors that could guide more personalized therapeutic approaches." The human introduction's first paragraph closes on a statistic and a hedged cause ("probably due to perceived suboptimal medication efficacy and/or the bothersome ADR") and leaves the aims for a later paragraph. The AI's paragraph is a complete miniature IMRaD; the human's is one step of an argument.
3. **Evaluative adjectives on the problem (E).** "crucial for optimizing treatment strategies and improving patient outcomes"; "not fully elucidated"; "significant subset". The human text has "bothersome" (twice), which is a clinical term of art, and "Significant progress", once. The AI applies an importance adjective to every noun that names the research problem.
4. **The 7-word pivot (E).** The human paragraph has "However, treatment options for ET remain limited." at 7 words; that is its only connective opener and its shortest sentence, and it is the paragraph's hinge. Qwen's "However, a significant subset of patients..." is 13 words and is one of three "however/these/understanding" openers.
5. **Hedging distribution (E).** The human hedges once, at the end, about a cause. Qwen hedges the mechanism ("not fully elucidated", "have been proposed", "may play a role") and asserts the importance. The human asserts the facts and hedges the explanation; the model does the reverse.

### 8.3 Band C/D pair: Holmes in *Abrams* (52.2) vs Qwen dissent (56.2)

Sequences: Holmes **11, 32, 46, 80, 10, 10, 19**; Qwen **20, 25, 31, 17, 31, 19, 20, 22**. CV 0.81 vs 0.22 at the same FRE. This is the pair where the shape difference is starkest.

1. **The argument is not made, it is described (C/E).** Holmes's paragraph is a chain of reasoning: persecution is logical *if* you are certain; permitting speech implies you think it impotent or you doubt yourself; *but* men have seen time upset fighting faiths; *therefore* the market test. Qwen's paragraph asserts the conclusion eight times in different words ("truth is best tested in the open competition of ideas"; "the adversarial nature of free speech allows for the emergence of truth"; "in the crucible of debate... we find the strength to discern the true from the false"). There is no *if* and no *but* that changes anything. A rewrite can vary the sentences but cannot supply the missing inferential steps without an author.
2. **Sentence-length range (E).** Nothing under 17 or over 31 in the AI; Holmes goes 80 then 10, 10. Uniformity of this degree (CV 0.22 in an argumentative paragraph) is the single most visible statistical difference and is fully fixable.
3. **Aphorism vs slogan (E, hard).** Holmes: "It is an experiment, as all life is an experiment." Qwen: "Let us not stifle the voices of dissent, however much we may loathe them, but rather encourage them to flourish." The human short sentence is a *concession that widens the claim*; the AI's is an *exhortation*. Instruct models produce exhortations when asked for aphorisms.
4. **Second person and the concrete case (C).** Holmes addresses "you" ("If you have no doubt of your premises") and has a man who "says that he has squared the circle". Qwen has no addressee and no case. Fixable only if the source has one.
5. **Register lexis (E).** "compelled to voice a dissent", "in its wisdom", "the very rights that define our democracy", "the crucible of debate". Scrubbable, but the replacement words have to come from somewhere; without a case to describe, they will be replaced by other abstractions.

### 8.4 Band E pair: Orwell shooting (80.9) vs Qwen elephant (80.5)

FRE identical to the decimal, mean sentence length 16 vs 15, syllables per word 1.29 vs 1.32. Sequences: Orwell **37, 28, 13, 22, 23, 3, 9, 9, 7, 26, 5, 8, 21**; Qwen **12, 18, 17, 28, 15, 12, 7, 18, 8, 13, 13, 16, 16, 11**.

1. **Nothing is observed (C).** Orwell: "He neither stirred nor fell, but every line of his body had altered." "His mouth slobbered." "he sagged flabbily to his knees." "climbed with desperate slowness to his feet and stood weakly upright, with legs sagging and head drooping." Qwen: "a tusks that gleamed in the early light", "its trunk swaying as it snorted and trumpeted", "I could see the panic in their eyes", "The sun rose higher, and the heat intensified." Every AI detail is the stock detail for its category (elephant: tusks and trunk; villagers: panic in eyes; tension: sweat down back). Orwell's details are the ones you would only have if you were there. This is the difference a detector trained on 28 million documents learns first and it is the one paraphrase cannot touch.
2. **Time is compressed in the human, evenly spread in the AI (E, partly).** Orwell spends 90 words on the five seconds after the shot ("In that instant, in too short a time, one would have thought, even for the bullet to get there...") and 20 on firing twice more. Qwen spends equal words on arriving, seeing, signalling, closing in, coaxing, cheering. The human paragraph has a *centre*; the AI paragraph has a *sequence*.
3. **The short sentences (E).** Orwell's 3-, 5-, 7-, 8- and 9-word sentences are all observations or actions: "His mouth slobbered." "I fired again into the same spot." "I fired a third time." "That was the shot that did for him." Qwen's 7- and 8-word sentences are "I knew I had to act fast." and "I kept my composure." — internal states announced.
4. **Self-interpretation (E).** "I could feel sweat trickling down my back", "My heart pounded", "I breathed a sigh of relief", "I knew I had done my job". Orwell reports one feeling in 211 words and it is the crowd's ("the devilish roar of glee"). The AI narrator narrates its emotions at a rate of one per 40 words. This is the plain-band form of the "announced stance" problem in 8.1.
5. **The resolved ending (E).** Qwen: "The villagers cheered, and I knew I had done my job." Orwell's paragraph ends mid-agony and the essay's famous ending is "I often wondered whether any of the others grasped that I had done it solely to avoid looking a fool." Human narrative paragraphs end on the next fact; AI paragraphs end on closure.
6. **Grammar error (E, and a warning).** "a tusks that gleamed". The AI paragraph has an agreement error and the human has none. Report 00 §11 says zero errors is a tell, but this is the wrong kind of error: a wrong-number determiner is what a rubric grader penalises. The permitted imperfections are rhetorical (see §6.2), not grammatical.

### 8.5 The prompt-following failure

Asked for Orwell's argument "in the style of a serious literary essayist", Qwen produced FRE 11.0 (2.0 syllables per word, "the rich tapestry of traditional language", "a foregone conclusion", "vessels of thought and culture"). Asked for the same argument in "plain, mostly one- and two-syllable words... readable by a 16-year-old", it produced FRE 82.9 with a mean sentence of 10 words and sentences like "English is getting worse." and "This will help make the language better." Orwell's actual paragraph on this topic is FRE 49.6. The model does not slide along the readability axis; it has an ornate mode and a simple mode, and instructions select between them. Neither mode has a 51-word sentence followed by a 14-word one, which is what Orwell does (14, 51, 14, 29, 31, 32, 17, 10, 20). The consequence for the pipeline is in §10.

### 8.6 What separates them, in order

Across the five matched pairs, the differences a detector would weight, largest first:

1. Specific, checkable, first-hand content (names, dates, numbers, apparatus, observed detail) — **C**, unfixable without the author.
2. Inferential structure: an argument that turns (if/but/therefore) vs a claim restated — **C/E**; rewriting can restore a turn only if the source contains one.
3. Announced importance and narrated feeling ("crucial", "opens the door", "I knew I had done my job") — **E**, but the *habit* is structural and needs a closing-sentence rule, not a word list.
4. Sentence-length uniformity and the absence of a verdict sentence after a long one — **E**.
5. Closing on summary, exhortation or resolution — **E**.
6. Connective openers where a human uses a demonstrative or a repeated noun — **E**.
7. Register lexis ("delve", "tapestry", "crucible", "in its wisdom") — **E**, and already handled by the scrub.

Items 4-7 are what current humanizers fix. Items 1-3 are why they get caught (report 12).

---

## 9. Texture specs per band

Each spec lists properties a paragraph of 150-300 words should have to sound like the good human writing in that band, and five things it must not do. Numbers are drawn from the exemplar tables above; where a property is measurable the threshold is given as a check that passes on at least three quarters of the skilled exemplars in the band and fails on the AI passages at the same FRE. Where the exemplars disagree, the spec says so rather than averaging them. These are paragraph-level; document-level distributions are in report 10.

### 9.1 Band A, very dense (FRE below 30)

Should:
1. Mean sentence length 20-55 words, **or** syllables per word above 1.9; not both low. If mean sentence length is under 25, polysyllable share should be above 25 percent (the modern research route); if it is over 35, syllables per word may fall to 1.55 (the Darwin route).
2. Sentence-length CV 0.18-0.45. Do not push higher; Darwin's scrutinising paragraph is 0.19.
3. Exactly one pivot sentence of 7-15 words per paragraph, placed *after* the longest sentence or as the final sentence, and it must be a conclusion, an example or a reversal ("Thus...", "Take, for example,...", "However, treatment options for ET remain limited.").
4. At least one sentence over 40 words built from coordinated noun phrases or a list with semicolons, not from nested subordinate clauses. Semicolons or colons: at least 1 per 200 words.
5. At least two concrete nouns (an organism, an instrument, a drug, a place, a number) in the first three sentences, even when the paragraph's subject is abstract.
6. Sentence-initial connectives: at most 1 per paragraph, and only *However*, *Thus*, *Therefore* or an ordinal inside an explicit enumeration.
7. Demonstrative or repeated-noun cohesion ("These laws...", "This structure...") in at least two consecutive sentence pairs.
8. First-person plural for inference permitted at 5-30 per thousand words ("we find", "we must believe", "our understanding"); none for feeling.
9. Comma density 40-115 per thousand words; the exemplars are all above 30.
10. Final sentence states a fact, a figure or a specific conjecture. It may be the longest sentence in the paragraph (Darwin) or a hedged number (PMC).

Must not:
1. Contain a sentence under 6 words. Fragments belong to bands B and D.
2. Open more than one sentence with an evaluative frame ("It is crucial to", "Understanding X is essential").
3. Close on "aims", "goals", "future work", "opens the door" or any forward-looking abstraction.
4. Repair density by splitting a 50-word list sentence into three 17-word sentences; that moves the text to band C shape while keeping band A lexis, which no exemplar does.
5. Simplify vocabulary. A band-A text with 1.9 syllables per word is that way because the terms are technical; replacing "etiopathogenesis" with "causes" changes the discipline, not the humanness.

### 9.2 Band B, dense academic (FRE 30-45)

Should:
1. Mean sentence length 20-32 words (24-29 is the centre: Mill, Gibbon, Jackson, Orwell, Hume, Scalia), syllables per word 1.60-1.80.
2. CV 0.35-0.75, with the flat exception (0.22) permitted for definitional prose; do not force it up.
3. At least one sentence of 40+ words and at least one of 10 words or fewer in every paragraph over 180 words, and the short one must come *after* a sentence of 30+ words at least once (Scalia 48/47 then 7; Madison 38 then 7; Jackson 34 then... 8).
4. The short sentence is an aphorism, an example or a label ("But this wolf comes as a wolf." "Ambition must be made to counteract ambition." "this is called *pacification*."), never a transition.
5. One concrete case within one sentence of every abstraction: a statute, a place, a species, a date, a quoted phrase.
6. Sentence-initial *But* permitted and characteristic: 1-2 per paragraph (Scalia, Jackson, Madison, Mill, Hume all do it). Sentence-initial formal connectives at most 1 per paragraph.
7. Cohesion by demonstrative subject ("Such phraseology", "That would be", "These slaves") at least twice per paragraph.
8. Paired or listed abstract nouns ("instability, injustice, and confusion"; "euphemism, question-begging and sheer cloudy vagueness") permitted; but the *next* sentence must cash one of them out concretely.
9. First person: institutional or argumentative ("We think", "we apply", "I strongly believe", "one wants") at 0-50 per thousand words.
10. Colons used for labelling and definition at least once per 250 words (Orwell, Mill, Scalia, Gibbon).
11. A question is permitted once per paragraph as a transition ("But what is government itself, but the greatest of all reflections on human nature?").

Must not:
1. Open two consecutive sentences on the same word, or more than 60 percent of sentences on *The/This/It/In/There* (Gibbon and Federalist 10 reach 75-80 percent and are the least modern-sounding passages; the modern exemplars are at 11-43 percent).
2. Follow the aphorism with a restatement of it; the exemplars move on to the next case.
3. End on a summary of the paragraph's own claim.
4. Use a contraction or an apostrophe error (the one student paper does both and it is the reason it reads as a student).
5. Let the polysyllable share exceed 27 percent; above that the text is band A and its rhythm rules apply.

### 9.3 Band C, upper-undergraduate and research article (FRE 45-55)

Should:
1. Mean sentence length 17-32 words, syllables per word 1.40-1.60, polysyllables 10-17 percent.
2. Two admissible shapes, and the paragraph must be recognisably one of them: **report** (12-29 words per sentence, CV 0.25-0.35, no sentence over 30, each sentence one proposition; Watson & Crick, Darwin's definition) or **argument** (CV 0.55-0.85, at least one sentence over 45 and two under 12; Holmes twice). Do not blend them into a CV of 0.45 with a 35-word maximum, which none of the exemplars has.
3. In argument shape, the shortest sentences are examples or verdicts and follow the longest sentence ("Sunday laws and usury laws are ancient examples."). In report shape, the shortest sentence is a stated assumption or an acknowledgment ("They kindly made their manuscript available to us in advance of publication.").
4. One concrete instance per abstraction, with a preference for the homely over the technical when explaining (a man who takes to drink; canines in a time of dearth; the Postoffice).
5. Physical verbs at the main-clause level in at least half of the sentences ("sweep away", "wager", "repel", "languish and die", "take to drink").
6. Sentence-initial connectives: at most one per 250 words, and *But* or *Now* preferred to *However*. Enumeration inside a sentence with (1), (2) is the research-article way to sequence reasons.
7. First person for intellectual acts at 15-65 per thousand words: "We wish to suggest", "In our opinion", "I do not conceive that to be my duty", "I shall replace the question".
8. The paragraph's first sentence is a plain declarative a reader could disagree with ("Persecution for the expression of opinions seems to me perfectly logical.").
9. The last sentence is a fact, a definition or a maxim of 8-15 words ("The point is that the process is reversible.").
10. Parentheses for citations, numbering and asides: 0-5 per paragraph, all short.

Must not:
1. Use "However" to open two sentences in one paragraph (the marker of the student Austen essay).
2. Hedge the facts and assert the importance; do the reverse (assert the facts, hedge the explanation).
3. Include a sentence whose only content is that the topic is important, complex or multifaceted.
4. End with a sentence that begins "This shows" / "This demonstrates" / "Overall".
5. Put the verdict before the principle; the exemplars state the rule in the long sentence and the instance in the short one.

### 9.4 Band D, quality essay and long-form (FRE 55-70)

Should:
1. Syllables per word 1.30-1.47; polysyllables 6-12 percent. This is the plainest vocabulary of the study and the writing is not less intelligent for it.
2. CV 0.40-0.85 for argumentative or reflective paragraphs; 0.25-0.35 permitted for narrative sequence (Orwell's coolie paragraph, Gladwell's drive down Atlantic Avenue).
3. At least one sentence under 8 words per 150 words, and at least one over 45 per 200 words, in argumentative paragraphs. The long sentence is a list or an accumulation, not a nest: the reader must be able to count the items.
4. The short sentence is a turn ("But this is absurd.", "And the war came.", "I will try to explain.") or a verdict ("It is a deliverance which does not deliver."), and at least one short sentence directly follows a sentence of 45+ words.
5. Openers: at least one paragraph in three begins with a sentence of 10 words or fewer that states a fact or a thesis ("The mass of men lead lives of quiet desperation.", "On the 29th of July, in 1943, my father died.").
6. Proper nouns, dates or numerals: at least 3 per 200 words in narrative and journalistic paragraphs (Baldwin 22 per thousand, Gladwell 136, Douglass 36-45); in reflective paragraphs (Emerson, Thoreau, Woolf's moth) they may be near zero, but a physical object must appear in each sentence pair instead ("minks and muskrats", "the plough scoring the field", "shingle-shavings").
7. One abstraction per paragraph, placed last and earned by the specifics before it ("the real nature of imperialism"; "the spoils of injustice, anarchy, discontent, and hatred").
8. Cohesion by date, parallelism, repetition of a key word (three to seven times), pronoun shift, or sentence-initial *But* (1-3 per paragraph); sentence-initial formal connectives 0.
9. First person as agent or witness, 10-110 per thousand words, attached to actions and decisions; direct address to the reader ("you may say") at most once per paragraph.
10. Rhetorical imperfections permitted at about one per paragraph: a parenthetical full sentence, an exclamation, a rhetorical question, a fragment, an inverted sentence, a comma-spliced series.
11. Questions permitted in runs when they are the argument (Orwell's six questions in a row).
12. Closing on an image, a sardonic clause or the next fact; never on the thesis.

Must not:
1. Narrate feelings ("I felt", "my heart pounded", "I craved") at more than one per 200 words; the exemplars report actions and let the feeling follow.
2. Use "I think", "I feel", "in my opinion", or "personally" anywhere.
3. Use stock detail that any writer could have supplied for the category (gleaming tusks, panic in their eyes, a riot of red and gold leaves). If the source has no first-hand detail, do not invent any; leave the sentence plain.
4. Resolve: no "I knew I had done my job", no "it was time to find a new chapter", no lesson.
5. Introduce a grammatical error to look human; the permitted imperfections are rhetorical.

### 9.5 Band E, plain (FRE above 70)

Should (see §7 for the small exemplar set and its warning):
1. Mean sentence length 12-20 words; syllables per word 1.25-1.35; polysyllables under 8 percent.
2. CV 0.35-0.65, produced by two or three sentences of 3-9 words placed at the moment of highest consequence, not evenly spread.
3. At least one sentence over 30 words per paragraph, built by coordination with "and", "but" or semicolons (Douglass: seven semicolons in 221 words).
4. Every short sentence is an action or an observation ("His mouth slobbered." "I fired a third time." "Eli was turning, Smith was feeding, and I was carrying wheat to the fan."); none is an internal state announced.
5. Time compressed unevenly: the paragraph spends most of its words on its central seconds and hurries the rest.
6. Names and numbers where the writer has them ("Bill Smith, William Hughes, a slave named Eli"; "about one hundred yards"; "five seconds, I dare say"), and where the writer has none, objects instead ("his bunch of grass", "an old .44 Winchester").
7. Sentence-initial *But*, *Besides*, *At last*, *Alive,* permitted; sentence-initial formal connectives 0.
8. Parenthetical asides as complete sentences permitted once per paragraph.
9. Ending on the next physical fact, mid-event.

Must not:
1. Drop the mean sentence length below 12 or the maximum below 25; that is the AI "simple mode" (FRE 83, mean 10.4) and it reads as a child's composition, not as plain prose.
2. Narrate emotion more than once per paragraph.
3. Use any stock category detail (see 9.4 must-not 3).
4. Close with relief, cheering, a sigh, a lesson, or "I knew".
5. Contain an agreement, number or spelling error.

---

## 10. How the pipeline uses this

The pipeline's current shape (`src/humanizer/humanize/pipeline.py`, `llm.py`): candidates are hard-gated on length ratio, content overlap and invariants (`_gate_candidate`), ranked by proxy AI-probability plus a CV-distance and drift penalty (`_rank_key`, `_cv_distance`), quality-flagged when CV leaves `HUMAN_SENTENCE_CV` or FRE moves more than 20 points (`_quality`), and generated under `STYLE_CONTRACT` rules 1-11 with twelve persona/constraint specs. Three consequences of this report: (a) the CV band and the "one sentence under eight, one over thirty" rule should be conditioned on the source's FRE band, because they are right for band D and wrong for band A; (b) `readability_shifted` at 20 points is too loose, since the model's two modes sit 70 points apart and a 15-point move crosses a band; (c) the checks below are cheap enough to run on every candidate.

### 10.1 Checks per band (spaCy `en_core_web_sm` is installed; `textstat` is in the venv; most checks need only the regex splitter)

Band is chosen from the *source* paragraph's FRE. "Gate" = hard reject; "Rank" = add to `_rank_key` penalty.

| # | Check | A (<30) | B (30-45) | C (45-55) | D (55-70) | E (>70) | Use | Tool |
|---|---|---|---|---|---|---|---|---|
| 1 | FRE of candidate within band of source | ±8 | ±8 | ±8 | ±8 | ±8, cap 72 for expository | Gate | textstat |
| 2 | Sentence-length CV | 0.18-0.45 | 0.30-0.78 | report 0.25-0.38 or argument 0.55-0.90 | 0.40-0.90 (0.25+ if narrative) | 0.35-0.90 | Gate outside, Rank inside | splitter |
| 3 | Shortest sentence (words) | 6-15 | 5-10 | ≤12 (argument) | ≤8 | ≤6 | Rank | splitter |
| 4 | Longest sentence (words) | ≥40 | ≥40 | ≥45 (argument) / ≤30 (report) | ≥45 | ≥26 | Rank | splitter |
| 5 | Short-after-long: a sentence ≤12 words immediately follows the longest sentence, or is final | required in B, C-arg, D | required | required (arg) | required | preferred | Rank | splitter |
| 6 | Sentence-initial formal connectives (regex list in `score.py::CONNECTIVES`) | ≤1 | ≤1 | ≤1 per 250 w | 0 | 0 | Gate at >2, Rank | regex |
| 7 | Sentence-initial *And/But/So* | ≤1 | ≤2 | ≤2 | ≤3 | ≤3 | Gate above | regex |
| 8 | Share of sentences opening on The/This/It/In/There | ≤60% | ≤60% | ≤55% | ≤45% | ≤50% | Rank | regex |
| 9 | Consecutive sentences with same first token | 0 (unless deliberate parallelism in D) | 0 | 0 | ≤1 | ≤1 | Rank | regex |
| 10 | Syllables per word | ≥1.85 if MSL<25, else ≥1.55 | 1.60-1.80 | 1.40-1.60 | 1.30-1.47 | 1.25-1.35 | Rank | textstat |
| 11 | Polysyllable share | 15-37% | 14-27% | 10-17% | 6-12% | <8% | Rank | textstat |
| 12 | Concrete anchors per 200 words (PROPN + NUM + dates, spaCy) | ≥2 | ≥3 | ≥3 | ≥3 narrative, ≥0 reflective | ≥3 | Rank; never inject | spaCy NER/POS |
| 13 | Source anchors preserved (numbers, names, dates) | 100% | 100% | 100% | 100% | 100% | Gate (already `_invariant_failures`) | existing |
| 14 | Importance adjectives on the topic (*crucial, vital, essential, pivotal, significant, key, important*) | ≤1 | ≤1 | 0 | 0 | 0 | Gate at >1 | regex (extend `BANNED_VOCABULARY`) |
| 15 | Closing-sentence class: no *aims/goal/future/opens the door/overall/in conclusion/I knew/lesson*; ends on NOUN/NUM/quoted phrase | required | required | required | required | required | Gate | regex + spaCy POS of last token |
| 16 | Feeling verbs in first person (*I felt, I knew, my heart, I craved, I found myself*) per 200 words | 0 | 0 | 0 | ≤1 | ≤1 | Gate at >1 | regex |
| 17 | "I think / I feel / in my opinion / personally" | 0 | 0 | 0 | 0 | 0 | Gate | regex |
| 18 | Contractions in body | 0 | 0 | 0 | 0 | ≤1 | Gate | regex |
| 19 | Semicolon or colon per 200 words | ≥1 | ≥1 | ≥0 | ≥0 | ≥0 | Rank | count |
| 20 | Comma density /1k | 40-115 | 30-100 | 30-90 | 40-95 | 35-90 | Rank | count |
| 21 | Grammatical error injected (agreement, number, spelling) | 0 | 0 | 0 | 0 | 0 | Gate | spaCy dep + wordlist, or diff vs source |
| 22 | Rhetorical imperfection present (fragment ≤4 words, parenthetical sentence, question, exclamation) | 0 | ≤1 | ≤1 | 1 per paragraph | 1 | Rank (D, E only) | regex |

### 10.2 Prompt instructions per band, and where prompting will fail

| Band | Paste-able instructions (5-8) | Moves the instruct model is bad at (needs a deterministic transform) |
|---|---|---|
| A | Keep every technical term. Build one sentence of 40+ words from a list joined by semicolons, not from nested clauses. Put one sentence of 7-15 words after it that states the consequence or the pivot. Open the paragraph on a named organism, instrument, drug, equation or place. Use "we" only for "we find / we suggest / we assume". No sentence may say the topic is important. Close on a number, a hedged cause or a specific conjecture. | Holding lexis while lengthening sentences (model shortens and simplifies together, §8.5). Producing a verbless list sentence (Darwin, Mill) at all. Not appending an aims sentence. Keeping CV low; the model either flattens to 0.2 or splits everything. |
| B | Write one period sentence of 35-55 words listing cases, then a sentence of 7-10 words that judges it. Begin one sentence with "But". Name the statute, place, species or date within one sentence of each abstract noun. Use a colon to attach a label or definition once. Link sentences with "Such", "That", "These", not with "However" or "Furthermore". Do not restate the aphorism; move to the next case. End on a case or an aphorism, not on the claim. | Placing the short sentence *after* the long one (model puts it first as a topic sentence). Writing an aphorism rather than an exhortation (§8.3). Demonstrative cohesion (model reverts to connectives under length pressure). Not summarising at the end (rule 5 of the contract is the most-violated). |
| C | Decide whether this is a report or an argument and keep to one shape. Report: sentences of 12-29 words, one proposition each, number reasons inside a sentence as (1), (2). Argument: one sentence over 45 words stating the principle, then two under 12 giving examples. Use a physical verb in the main clause of most sentences. "We suggest / I believe / in our opinion" for claims; never "I think". Open on a plain declarative someone could deny. Close on a fact or a maxim of 8-15 words. | Keeping report shape (model adds variance where none belongs). Producing the example *after* the principle. Physical verbs for abstract subjects. Removing "However" at sentence start (the model treats it as neutral). Numbered reasons inside running prose. |
| D | Open with a sentence of ten words or fewer stating a fact or a thesis. Build the longest sentence as a countable list or an accumulation, 45+ words, then follow it with a sentence under 8 words that turns or judges. Use the source's names, dates and numbers; if there are none, use its objects; invent nothing. Bind sentences by repeating a key word three to five times, by dates, or by "But"; use no formal connectives. Report actions, not feelings. Permit one parenthetical sentence, question or exclamation. End on an image or the next fact. | Not narrating feelings (§8.4). Not inventing stock detail when the source has none. Earning one abstraction at the end instead of one per sentence. Repetition as cohesion (model treats repetition as an error and varies synonyms). Ending mid-event. |
| E | Only for narrative source paragraphs; refuse for expository content above FRE 72. Sentences average 12-20 words with two or three of 3-9 words at the moment of consequence and one over 30 built with "and", "but" or semicolons. Every short sentence is an action or an observation. Spend most words on the central seconds. One parenthetical aside allowed. No emotion words except one. End mid-event. | Uneven time (model spreads words evenly across events). Emotion restraint. A 30+-word coordinated sentence in plain register (model either writes 10-word sentences or reverts to ornate). Not resolving. |

Across all bands the model reliably follows: banned-word avoidance (partially), first-person allowance, length ratio, "no preamble". It reliably fails: closing-sentence discipline, short-after-long placement, not announcing importance, not narrating feeling, holding a readability band, and every instruction that asks it to *withhold* something (an abstraction, a resolution, a synonym). Those six should be deterministic gates or transforms, not prompt lines.

### 10.3 The matched-FRE differences, ordered for a detector, with fixability

From §8.6, restated for implementation. **E** = expression, fixable by rewrite; **C** = content, needs the author.

| Rank | Difference | Fixable | Pipeline action |
|---|---|---|---|
| 1 | Absence of first-hand specifics (names, dates, numbers, apparatus, observed detail); stock category detail in their place | C | Preserve every source anchor (gate 13); never invent; report anchor density to the user as a quality signal, not as a thing the tool can supply |
| 2 | No inferential turn; the claim is restated instead of argued | C/E | Detect restatement (high lexical overlap between sentences within a paragraph, spaCy lemma sets); rank down; cannot add a turn the source lacks |
| 3 | Announced importance and narrated feeling | E (habit is structural) | Gates 14, 16; closing-sentence gate 15 |
| 4 | Uniform sentence length; no verdict sentence after the long one | E | Checks 2-5; transform that merges the two shortest adjacent sentences and splits the longest at its final clause, then re-checks |
| 5 | Closing on summary, exhortation or resolution | E | Gate 15; transform that deletes a final sentence matching the class if the paragraph keeps ≥85% length |
| 6 | Connective openers where a human uses a demonstrative or repeated noun | E | Gate 6; existing scrub already strips; add a demonstrative-substitution transform ("However, X" → "That X" only when X's subject is the previous sentence's object) |
| 7 | Register lexis | E | Existing `BANNED_VOCABULARY` and `replace_ai_vocabulary` |

---

## 11. Sources and provenance

All passage texts are in `.readband/passages/<id>.txt`; provenance for every id is in `.readband/provenance.json`; metrics in `.readband/tables.md`; scoring in `.readband/score.py`; extraction in `.readband/extract_passages.py` and `.readband/extract_more.py`; AI generation in `.readband/gen_ai_contrast.py` and `.readband/gen_ai_contrast2.py` with outputs in `.readband/ai_passages/` and `.readband/ai_contrast.json`.

Public domain, fetched by `curl`: Hume, *Enquiry* (Gutenberg #9662); Madison, Federalist 10 and 51 (Gutenberg #1404); Darwin, *Origin* (Gutenberg #1228); Emerson, *Essays* (Gutenberg #16643); Thoreau, *Walden* (Gutenberg #205); Mill, *On Liberty* (Gutenberg #34901); Gibbon, *Decline and Fall* vol. 1 (Gutenberg #731); James, *Pragmatism* (Gutenberg #5116); Douglass, *Narrative* (Gutenberg #23); Twain, *Life on the Mississippi* (Gutenberg #245); Hemingway, *in our time* 1924 (Gutenberg #61085) and *The Sun Also Rises* (Gutenberg #67138), both US public domain; Woolf, *A Room of One's Own* (Gutenberg Australia #0200791, US public domain since 2025) and "The Death of the Moth" (Gutenberg Australia #1203811h; public domain in Australia, quoted for research); Lincoln, Second Inaugural (avalon.law.yale.edu) and Gettysburg Address, Bliss text (abrahamlincolnonline.org); Holmes, *Lochner* and *Abrams* dissents, Jackson, *Barnette*, Scalia, *Morrison v. Olson* dissent (law.cornell.edu; US court opinions are public domain); Einstein 1905, Perrett and Jeffery translation (fourmilab.ch).

Copyrighted, quoted in short excerpts for research: Orwell, "Politics and the English Language", "Shooting an Elephant", "Why I Write" (orwellfoundation.com; public domain in UK/EU/Canada/Australia, not in the US); Turing 1950, section 1 (PDF at csee.umbc.edu); Didion, "Goodbye to All That", opening 179 words (PDF at erhsnyc.org); Hersey, *Hiroshima*, opening 165 words (newyorker.com); Gladwell, "The Tipping Point", opening 140 words (newyorker.com, 3 June 1996).

MICUSP (University of Michigan; research use permitted, commercial use needs permission): PHI.G1.01.1, BIO.G0.15.1, HIS.G1.01.1, ENG.G0.22.1, fetched from `micusp.elicorpora.info/view?pid=<ID>` in the first pass of this work (`.readband/raw/micusp_*.txt`). PMC: the 60 local files in `data/raw/pmc` (open access; report 14).

**Transcribed from memory, not verified against a fetched copy, and to be checked before any external use:** Watson & Crick 1953 (the first two paragraphs, about 120 words, were verified against the nature.com abstract HTML in `.readband/raw/watsoncrick.txt`; the remainder is from memory); Baldwin, "Notes of a Native Son", opening two paragraphs (archive.org returned "Item not available").

**Written by the research agent:** all AI-contrast prompts; the "stereotypical" characterisations in §8 are of the Qwen outputs, not of hand-written parodies. No human passage was edited beyond removing Gutenberg paragraph numbers, footnote markers, italics underscores and a hyphenation break in the Cornell *Barnette* text.

**Not done.** PERSUADE 2.0 score-6 essays were not downloaded. The Watson & Crick and Baldwin texts are unverified as noted. Texture features are regex heuristics and were checked by hand only on the non-PMC passages. The band-E exemplar set is small because skilled expository prose does not reach it, which is itself the finding of §7.
