# Project structure log

## Step 1 — created the requested structure

What we did: created `paper`, `Report`, `code`, `checkpoint`, and `data`.

Why: these are the only project folders requested.

What we found: there were no existing project files to move into them.

## Step 2 — added the starting file

What we did: added `main.py` in the project root.

Why: it is the single starting point requested.

What we found: running it prints a clear terminal confirmation.

## Step 3 — built the factual dataset

What we did: saved a local Wikidata source snapshot and generated 100 capital-city concepts with five fixed realization families each.

Why: this holds C constant while changing S in a complete factorial layout.

What we found: the dataset has 500 rows. Rejected source records were saved with their reasons instead of being silently discarded.

## Step 4 — validated and audited the dataset

What we did: ran the validator and inspected a fixed sample of 50 concept groups.

Why: this checks factor consistency, split separation, full S coverage, and meaning across realizations.

What we found: validation passed with zero leaks, duplicates, missing values, or inconsistent C fields. The semantic audit passed 50 out of 50 groups.

## Step 5 — designed reusable semantic structure

What we did: defined a future fact as domain, relation, subject, and value.

Why: repeated relations and overlapping subjects give later analysis reusable structure instead of isolated capital facts.

What we found: the design uses four domains and repeated relations, but does not create any new dataset rows.

## Step 6 — audited relation feasibility

What we did: checked every proposed relation for source property, value cardinality, reuse target, overlap, and ambiguity risk.

Why: ambiguous facts would make C poorly defined before the larger dataset is built.

What we found: clean geography and chemical-element relations can proceed after source filtering; multi-valued people and object relations are deferred.

## Step 7 — ran source-backed relation coverage

What we did: queried Wikidata and saved factual tables, a source snapshot, rejection reasons, and value-reuse statistics.

Why: each domain needs at least 50 subjects shared across all three relations before text construction.

What we found: geography has 174 shared subjects. Replacing the unusable group relation with Royal Society of Chemistry `period_of` gives science 118 shared subjects; both coverage checks pass.

## Step 8 — built the factual C-matrix

What we did: created one row per domain, relation, subject, and value proposition from the approved common-subject sets.

Why: this fixes semantic facts and subject-level splits before adding any surface text.

What we found: the matrix has 876 rows. Every retained subject has exactly three facts, and validation found no subject leakage or text/S fields.

## Step 9 — tested surface templates

What we did: rendered three factual examples per relation across five surface families and checked the candidate texts.

Why: templates must preserve each proposition before full text generation is allowed.

What we found: 216 candidates contain their fixed subject and value. Each sampled fact has 3 declarative, 3 question, 3 paraphrase, 2 formal, and 1 structured variant. No full surface dataset was generated.

## Step 10 — generated strict V1 surface rows

What we did: generated all approved variants from the strict source-validated factual matrix.

Why: V1 rejects unresolved labels instead of introducing a second label source.

What we found: 148 geography subjects and 118 science subjects produce 798 facts and 9,576 rows. All validations and the 50-fact audit pass.

## Step 11 — final stratified semantic spot audit

What we did: checked a fixed sample of 60 facts, with 10 facts from every relation and all 12 variants for each fact.

Why: structural checks alone cannot confirm that every surface form preserves its proposition.

What we found: all 60 fact groups passed proposition preservation, surface variation, and no-factual-drift checks.

## Step 12 — audited frozen representations

What we did: extracted frozen Gemma 2 2B activations at pre-registered early, middle, and late layers, then measured pair similarities and diagnostic linear decodability.

Why: C and S both need to be measurable before any separation method is considered.

What we found: C domain/relation and S family are strongly decodable on held-out subjects at all three layers. Same-fact rows are similar across S but are not identical. No InfoNCE or SAE training was run.

## Step 13 — added layer 8 to the representation audit

What we did: extracted the same frozen, masked-mean activations at layer 8 and reran the fixed diagnostics.

Why: layer choice remains a diagnostic sweep decision rather than a single arbitrary layer.

What we found: layer 8 also has strong held-out C and S decodability, with same-fact/different-S mean cosine similarity of 0.969.

## Step 14 — tested C/S linear subspace overlap

What we did: projected out learned S-family directions and remeasured C, then projected out learned C directions and remeasured S.

Why: decodability alone does not show whether desired and nuisance information share directions.

What we found: C and S remain almost perfectly decodable after removing each other’s learned linear subspaces. The frozen representation is linearly separable on this diagnostic.

## Step 15 — tested iterative linear erasure

What we did: repeatedly learned an S-family or relation probe on train subjects, removed its learned row space, then retrained on the remaining representation. We tested layers 5, 8, 13, and 21 and stopped using the predeclared no-material-decrease rule.

Why: a single probe direction may miss redundant copies of the same information.

What we found: after removing ranks 4 and 8 for S family, S accuracy remained 0.997–1.000 while C domain and relation stayed 1.000. Relation also remained 1.000 after ranks 5 and 10 were removed. These frozen representations contain redundant linear paths for both S and relation; this diagnostic did not find a small removable S-only subspace.

## Step 16 — trained and tested the contrastive encoder

What we did: trained a standard InfoNCE encoder on frozen layer-8 activations from C-train subjects and declarative, question, and paraphrase rows only. Formal and structured were excluded from every training pair. We then froze the encoder and tested subject-disjoint C-test rows.

Why: positive pairs with the same fact and different allowed forms should reduce surface-form information while keeping factual structure.

What we found: C domain and relation stayed perfectly decodable. Overall S-family accuracy fell to 0.681, but the unseen formal and structured families were still strongly decodable at 0.943. The result is partial S suppression, not full generalization to unseen surface families. No SAE was trained.

## Step 17 — audited frozen cross-surface retrieval

What we did: kept the contrastive encoder frozen and compared same-fact C-test cosine similarities across trained and held-out surface-family groups. We also measured five-family probe confusion and retrieved facts across trained and held-out formats.

Why: reduced average S-family accuracy does not by itself show whether an unseen format can still retrieve the same fact.

What we found: train-format pairs are most similar (0.950), while train-to-held-out pairs are lower (0.771). Even so, formal/structured queries retrieve their same fact from a train-format-only bank with R@1 of 0.801 and MRR of 0.834. Surface-family labels remain especially clear for formal and structured rows. No new encoder or SAE training occurred.

## Step 18 — compared raw and contrastive retrieval

What we did: ran the same C-test cross-surface fact-retrieval task on raw Gemma layer-8 activations and the frozen contrastive output. We separately measured formal and structured queries.

Why: absolute contrastive scores cannot show whether contrastive learning improved the original representation.

What we found: raw Gemma was better in every requested direction. All held-out to train-S R@1 was 0.915 for raw Gemma and 0.801 for contrastive. Structured to train-S fell most sharply, from 0.863 to 0.598. This contrastive setup did not improve held-out surface-form retrieval.

## Step 19 — compared factual factor retrieval

What we did: used each C-test surface row as a query and excluded its same-fact and same-family rows from the bank. We retrieved other facts with the same relation but a different subject, and facts with the same subject but a different relation.

Why: the factual representation should be checked separately for relation-level and subject-level structure.

What we found: contrastive improved relation-sharing retrieval from R@1 0.883 to 0.961, but reduced subject-sharing/different-relation retrieval from 0.117 to 0.032. The current in-batch-negative objective pushes same-subject/different-relation facts apart.

## Step 20 — tested partial-C contrastive supervision

What we did: classified controlled fact pairs into exact-fact, relation-sharing, subject-sharing, domain-only, and no-shared-factor groups. We trained one new encoder with exact facts as strong positives, relation and subject sharing as weak positives, domain-only pairs neutral, and no-shared-factor pairs negative. Formal and structured stayed held out.

Why: the first contrastive objective treated partial factual sharing as a negative and damaged subject structure.

What we found: this version failed the gate. It improved relation retrieval to R@1 0.989, but reduced subject retrieval to 0.004 and held-out-to-train-S fact retrieval to 0.444. Its loss reached zero after the first epoch by separating the two domains, while S-family accuracy remained 0.998. We stop contrastive expansion here and do not train an SAE.

## Step 21 — checked partial-C collapse and geometry

What we did: measured effective rank, normalized-output norms, unrelated cosine similarity, and five controlled pair geometries on C-test. We then put raw Gemma, plain InfoNCE, and partial-C results in one matched table.

Why: a near-zero contrastive loss can be satisfied by an impoverished representation rather than by the intended factual geometry.

What we found: the 1,024-dimensional output has participation ratio 1.03 and entropy effective rank 1.11. It is effectively a one-dimensional domain split: all within-domain pair types have cosine 0.986–0.994 and cross-domain pairs have cosine -0.721. The new scheme fails the gate; do not train an SAE.

## Step 22 — trained a controlled 256-D fact bottleneck

What we did: trained one 256-dimensional bottleneck using only same-fact/different-family positives and different-fact/same-exact-template negatives. Training used C-train and S-train only; formal and structured were held out.

Why: this directly tells the model to ignore wording and distinguish facts without treating subject-sharing facts as a special negative class.

What we found: the bottleneck did not collapse (effective rank 32.76) and preserved unseen-S fact retrieval at R@1 0.926. It separates matched positives from matched negatives 97% of the time. However, unseen formal/structured S-family accuracy is still 1.000, so it fails the required generalization of S muting. Do not train an SAE yet.

## Step 23 — expanded S-family coverage with formal

What we did: kept the same 256-D bottleneck and matched-pair loss, added formal to the training families, and kept structured entirely unseen.

Why: the previous bottleneck worked for trained wording families but did not generalize S suppression to unobserved forms.

What we found: overall S-family accuracy improved from 0.575 to 0.395 and trained-family accuracy from 0.433 to 0.340. Structured remained perfectly decodable at 1.000. Geometry stayed healthy (effective rank 30.93), but coverage of four families still does not generalize to structured. No SAE was trained.

## Step 24 — tested one new indirect held-out family

What we did: created one separate natural-language indirect/reordered family, trained the unchanged bottleneck on all five existing families, and held indirect out completely.

Why: this is the clean generalization test of whether wider surface coverage produces universal S removal.

What we found: indirect facts retrieved extremely well (R@1 0.991) and the representation stayed healthy (effective rank 31.53). But indirect remained 0.991 decodable by the six-way S-family probe. The block is invariant relative to observed nuisance families, not universally S-free. No SAE was trained.

## Step 25 — ran raw Gemma baseline on indirect

What we did: evaluated frozen raw Gemma layer-8 activations on the same indirect-held-out retrieval and six-way S-family probe used for the bottleneck.

Why: the bottleneck needs to be compared with its input representation, not only judged by its own absolute scores.

What we found: raw Gemma has seen-family and indirect decodability of 1.000. The bottleneck reduces seen-family decodability to 0.326, keeps indirect decodability at 0.991, and improves indirect-to-seen fact R@1 from 0.983 to 0.991. This supports relative C specialization, not universal S erasure. No SAE was trained.

## Step 26 — audited candidate domains for diversity expansion

What we did: ran Wikidata feasibility checks for Films, Books, and Biological taxa using three relations per domain, exact-one-value filtering, source snapshots, retained tables, value-reuse statistics, and rejection logs where the source completed the cardinality query.

Why: more-domain training must use clean shared-subject structure rather than force the previously deferred People or Objects domains.

What we found: Books is eligible with 1,114 shared subjects across author, country of origin, and original language. Films (1,981) and Taxa (1,970) have large exact-one intersections, but their explicit multi-value rejection audit failed due to Wikidata service errors, so they are provisional and not approved for construction. No text or model training occurred.

## Step 27 — constructed the approved three-domain dataset

What we did: froze the approved Geography and Science rows, then added only the 1,114 source-validated Books subjects with author, country of origin, and original language facts. We generated the existing six surface families plus the held-out indirect family and checked every row.

Why: this creates a clean third-domain condition without using the unapproved Films or Taxa candidates.

What we found: the dataset has 4,140 facts and 53,820 surface rows. Every fact has the required rows, canonical labels and provenance are present, and validation found zero fact-count, surface-count, C-consistency, or unresolved-label failures. No model was trained during construction.

## Step 28 — ran balanced three-domain contrastive training

What we did: trained the same frozen-Gemma layer-8 to 256-D bottleneck on GPU. We used 83 C-train subjects from each of Geography, Science, and Books (747 facts total), trained on the five existing surface families, and held indirect out. The positive and negative rules, loss, optimizer, and training budget were unchanged.

Why: Books would otherwise supply 81% of facts, so an unbalanced run would test both added-domain diversity and added book volume.

What we found: training completed with loss 0.00078 and no collapse (effective rank 45.56; participation ratio 18.87). Indirect remained strongly decodable at 0.996, essentially unchanged from the two-domain value of 0.991. Indirect-to-seen fact retrieval remained high at R@1 0.986. Thus, balanced three-domain coverage preserves factual retrieval but does not improve unseen-family S suppression. No SAE was trained.

## Step 29 — tested natural-language rewrite coverage

What we did: replaced only the three paraphrase texts per fact with independently authored conversational rewrites. We kept the same five training families and retained indirect as the completely unseen natural test family. During validation, 64 Book subjects with pre-existing cross-fact exact-text collisions were rejected and logged; no labels, values, or provenance were replaced. We then repeated the same balanced 256-D GPU training and frozen audit.

Why: this tests whether more natural wording in training improves generalization to an unseen natural transformation without changing the model or contrastive loss.

What we found: the strict dataset retained 3,948 facts and 51,324 rows, with zero missing labels, row-count failures, or cross-fact exact duplicates. The training used 83 subjects per domain (747 facts) and finished at loss 0.00080 without collapse (effective rank 44.39). Indirect decodability fell from 0.996 to 0.963, while indirect-to-seen fact retrieval improved from R@1 0.986 to 0.997. This is a small but favorable improvement, not evidence of universal S removal. No SAE was trained.

## Step 30 — ran the matched fixed-template control

What we did: used exactly the same retained 3,948 facts, 64-subject rejection log, train/test splits, 83-subject-per-domain balanced training subset, seed, 256-D bottleneck, pairs, negatives, optimizer, epochs, and held-out indirect family. The only change was restoring the original fixed paraphrase text.

Why: the earlier natural-rewrite comparison used a larger population, so it could not isolate paraphrase style from subject selection.

What we found: fixed templates gave seen-S accuracy 0.396, held-out indirect accuracy 0.994, indirect-to-seen fact R@1 0.995, and effective rank 45.80. Natural rewrites gave 0.389, 0.963, 0.997, and 44.39 respectively. On matched facts, natural rewrites lower unseen-S leakage by 0.030 while retaining slightly better factual retrieval. No SAE was trained.

## Step 31 — trained matched raw and C-block SAEs

What we did: trained two ReLU-L1 sparse autoencoders on the same 8,964 natural-rewrite C-train rows from the same 747 balanced facts. One used frozen raw Gemma layer-8 activations (2,304 inputs, 9,216 features); the other used the frozen 256-D C bottleneck (1,024 features). Both used 4x expansion, per-dimension training-set standardization, L1 coefficient 0.001, unit-norm decoder columns, seed 20260825, and 30 GPU epochs. We evaluated only held-out C-test subjects, including indirect.

Why: this is the direct test of whether the C-dominant representation yields features that respond more to factual relation than surface family.

What we found: the C-block SAE has higher mean top-50 C-relation purity (0.594 versus 0.521), lower S-family purity (0.287 versus 0.526), 210 relation-selective features (20.5% of 1,024), and zero S-family-selective features. Raw Gemma has 281 relation-selective features (3.1% of 9,216) and 234 S-family-selective features (2.5%). Held-out reconstruction was successful for both after standardization (MSE 0.054 C-block; 0.154 raw). However, both codes are dense: mean L0 is 598/1,024 and 4,908/9,216. This is encouraging C/S feature separation under matched L1 training, but it is not yet a strong sparse-feature result. No contrastive encoder was retrained.

## Step 32 — swept Top-k SAE sparsity

What we did: froze the contrastive encoder and trained matched Top-k ReLU SAEs for k=16, 32, 64, and 128. Every raw/C pair used the same natural-rewrite C-train rows, balanced facts, expansion factor, optimizer, seed, and 30 epochs. Top-k fixes mean L0 exactly, so no arbitrary L1 penalty was used.

Why: this gives a direct reconstruction–sparsity–feature-quality curve and makes dictionary-normalized comparisons fair despite 9,216 raw features versus 1,024 C-block features.

What we found: the C-block had much lower S-family purity at every k (0.282–0.287 versus 0.483–0.626 raw). Its relation-selective feature fraction was 12.6–17.8% of the dictionary, compared with 0.6–2.9% for raw Gemma. It had zero S-selective features at k=16, 64, and 128, and one at k=32; raw Gemma had 0.3–1.2%. Reconstruction improved as k increased: at k=128, C-block MSE was 0.152 and raw MSE was 0.243. No contrastive encoder was retrained.

## Step 33 — tested C-block dictionary width

What we did: kept the natural-rewrite C bottleneck frozen, fixed Top-k at 64 active features, and compared 4x, 8x, and 16x dictionaries (1,024, 2,048, and 4,096 features). The 4x result was reused from the identical earlier run; only the 8x and 16x SAEs were newly trained. Training rows, facts, seed, optimizer, and epochs stayed fixed.

Why: the 4x dictionary was almost fully used, so more overcompleteness might allow the surviving factual information to split into more concept-pure sparse features.

What we found: relation purity increased modestly with width (0.548, 0.558, 0.564), while S-family purity remained low but rose slightly (0.283, 0.293, 0.301). Relation-selective dictionary fractions were 14.6%, 15.4%, and 13.0%; S-selective fractions were 0%, 0.05%, and 0.10%. Held-out reconstruction did not improve with the fixed 30-epoch budget (MSE 0.217, 0.238, 0.266). Wider dictionaries therefore give no strong evidence of improved factorization under this matched budget. No contrastive encoder or data was changed.

## Step 34 — trained and audited a two-branch C/S partition

What we did: trained a frozen-layer-8 encoder with two 128-D blocks, z_C and z_S, using only the requested controlled contrastive objectives. z_C matched same fact across different surface families against different facts with the same exact template. z_S matched different facts with the same exact template against the same fact under a different surface family. Training used the same balanced C-train subjects and five observed families; indirect remained held out. There was no decoder, reconstruction, covariance term, or SAE.

Why: the goal was to test whether separate pair supervision can produce a C-dominant block and an S-dominant block before applying an SAE only to z_C.

What we found: z_C is strongly C-decodable (domain 0.975, relation 0.996) and has excellent held-out same-fact retrieval (R@1 0.998), but S-family is still decodable at 0.656. z_S strongly encodes S (0.990; same-template/different-fact R@1 0.948), but it also retains domain 0.942 and relation 0.837. Its effective rank is only 4.40, so it is too collapsed for a clean S block. The directional pattern exists, but the required block separation does not. Do not train an SAE on this partition.

## Step 35 — compared z_C with the previous C bottleneck

What we did: evaluated the frozen 256-D natural-rewrite C bottleneck and the frozen 128-D two-route z_C block on the identical data, C-test subjects, six-family probe, and indirect-to-seen fact retrieval. No model was trained or changed.

Why: giving S its own route only helps if it reduces S leakage in z_C without damaging factual structure.

What we found: C-relation accuracy is effectively tied (0.997 single bottleneck; 0.996 z_C) and z_C has slightly higher fact retrieval (R@1 0.998 versus 0.997). But seen-family S accuracy rises sharply from 0.382 to 0.635, and all-six-family S accuracy rises from 0.426 to 0.656. z_C also has lower entropy effective rank (26.26 versus 44.39). Therefore the explicit S route makes the C block less pure under this objective. Do not add losses or train an SAE on z_C.

## Step 36 — tested gradient-reversal anti-leakage probes

What we did: kept the 128+128 two-route partition, controlled contrastive objectives, data, seed, optimizer, and epoch budget fixed. We added only two gradient-reversal classifiers at weight 0.1: S-family from z_C and C-relation from z_S. There was no reconstruction, swap, covariance, or SAE term.

Why: if the explicit partition is workable, adversarial probes should make each opposite factor harder to decode while preserving each block's intended factor.

What we found: this one modest-weight run fails. z_C keeps C relation at 0.997 and fact retrieval R@1 at 0.998, but its S-family accuracy rises from 0.656 to 0.954. z_S keeps S at 0.990 but C relation rises from 0.837 to 0.996. z_S remains low-rank (effective rank 6.72). The adversarial training constraints did not produce the intended partition, so do not freeze it or train an SAE on z_C.

## Step 37 — checked gradient-reversal mechanics and adversary signal

What we did: without retraining the partition, computed encoder gradients from ordinary nuisance classification and the gradient-reversal version on one identical batch. We also froze z_C and z_S, trained fresh nuisance probes alone, evaluated the final jointly trained adversary heads, and verified the target labels.

Why: a reversed sign, wrong labels, or a weak adversary could make the failed adversarial result an implementation problem rather than an optimization result.

What we found: the GRL sign is correct. The normal-versus-GRL encoder-gradient dot products are negative (-0.000553 for S from z_C; -0.026574 for C relation from z_S), with cosine -1.0 in both cases. Labels are correct: z_C targets the five observed S families and z_S targets the nine canonical C relations. Frozen fresh probes are very strong (S from z_C: 0.955 train / 0.956 C-test-seen; C relation from z_S: 0.996 train / 0.993 C-test). But the jointly trained adversary heads are at chance (about 0.21 for five-way S and 0.11 for nine-way relation). Thus the mechanics are correct, but the joint adversaries were too weak to give the encoder a useful anti-leakage signal. No lambda sweep or partition retraining was run.

## Step 38 — trained matched SAEs on raw Gemma, z_C, and z_S

What we did: kept the non-adversarial partition frozen and used Top-k SAEs at k=64, 4x expansion, 30 epochs, and the same balanced natural-rewrite training rows. The matching raw Gemma SAE result was reused from the completed identical run; new SAEs were trained only for z_C and z_S. All feature statistics use held-out C-test subjects and the same top-50 purity rule.

Why: this checks whether the C/S pair objectives create SAE features with different factual versus surface selectivity, even though the representation partition itself was not clean enough to freeze as a final method.

What we found: raw Gemma has reconstruction MSE 0.245, effective active dictionary size 3,671/9,216, relation purity 0.579, domain purity 0.853, S purity 0.517, 2.1% C-selective features, and 0.7% S-selective features. z_C has MSE 0.094, 512/512 active features, relation purity 0.601, domain purity 0.883, S purity 0.305, 20.1% C-selective features, and 0% S-selective features. z_S has MSE 0.014, 469/512 active features, relation purity 0.519, domain purity 0.879, S purity 0.732, 0.2% C-selective features, and 19.3% S-selective features. This is a directional SAE-feature result, but the upstream partition still has substantial cross-factor decodability. The partition was not retrained and no loss was added.

## Step 39 — checked availability for SAE activation-consistency audit

What we did: checked local checkpoints and saved artifacts before attempting the requested same-C/different-S versus same-S/different-C feature-activation audit.

Why: the audit requires each Top-k SAE's encoder weights or full per-example feature activations. Reconstructing these by rerunning an SAE would violate the instruction not to retrain anything.

What we found: the Top-k runs saved their aggregate report and feature-selectivity CSVs, but not their model checkpoints or per-example activation matrices. Therefore the requested audit cannot be computed from the frozen existing artifacts. No model, partition, or SAE was retrained.

## Step 40 — deterministically reran Top-k SAEs to persist audit artifacts

What we did: under explicit authorization, reran only the three exact existing Top-k jobs: raw Gemma layer-8, non-adversarial z_C, and non-adversarial z_S. All used k=64, 4x expansion, the same data, selected rows, seed, preprocessing, optimizer, 30 epochs, and train/test split. The sole added behavior was saving checkpoints, sparse C-test activations, and C-test example IDs.

Why: the earlier Top-k jobs did not save weights or per-example activations, which blocked the requested frozen feature-consistency audit.

What we found: every checked aggregate metric reproduced exactly, including reconstruction MSE, mean L0, active dictionary size, relation/domain/S purity, and selective-feature counts. The rerun is therefore a valid persistence copy of the prior jobs, not a new setting. The requested consistency analysis has not yet been run.

## Step 41 — audited frozen SAE feature-activation consistency

What we did: used only the saved sparse C-test activations and aligned example IDs for the frozen raw Gemma, z_C, and z_S Top-k SAEs. For 10,000 same-fact/different-S pairs and 10,000 same-template/different-fact pairs, computed per-feature consistency as one minus total absolute activation change divided by total activation mass. Features with zero activation in either pair set were excluded from cross-condition comparisons.

Why: the direct feature-level definition is whether a feature remains stable when C is preserved but S changes, versus when S is preserved but C changes.

What we found: z_C has mean C-fixed consistency 0.682 and S-fixed consistency 0.100. All 512 evaluated features have positive C-minus-S consistency; none are S-oriented. z_S shows the opposite majority pattern: C-fixed 0.092, S-fixed 0.370, with 52.4% S-oriented features versus 36.0% C-oriented. Raw Gemma is less clean: C-fixed 0.136, S-fixed 0.021, 51.9% C-oriented, 7.2% S-oriented, and 40.8% ties. This directly supports directional C/S feature specialization for the two frozen SAE branches, while the earlier representation probes still show the partition is not fully disentangled. No model or SAE was trained.

## Step 42 — audited frozen SAE features for multiple known factors

What we did: for every sufficiently active C-test SAE feature, inspected its top 50 activating examples and scored domain, relation, subject, reusable value, and S-family purity. A subject required support in at least three distinct C-test facts; a value required support in at least ten distinct C-test facts, plus at least three distinct top facts for a subject or value assignment. Primary labels use the strongest purity above that label's C-test baseline frequency.

Why: relation purity alone cannot show whether z_C decomposes into other reusable semantic factors such as subjects and values.

What we found: raw Gemma's primary assignments are mixed: relation 30.1%, subject 26.6%, reusable value 8.7%, domain 11.2%, and S family 23.4%. z_C shifts strongly toward semantic factors: relation 38.3%, subject 30.5%, reusable value 17.8%, domain 8.4%, and S family 5.1%. z_S is mostly surface-family assigned (74.0%), with relation 10.7%, subject 11.9%, reusable value 2.3%, and domain 1.1%. z_C's mean S-family purity is 0.305, compared with 0.516 raw and 0.732 z_S. This provides a feature-level decomposition result consistent with z_C being C-dominant, while not claiming the upstream representation partition is fully disentangled. No model or SAE was trained.

## Step 43 — tested seen-selected concept features on unseen indirect S

What we did: for raw Gemma SAE and z_C SAE independently, selected one feature per target concept using only C-test rows in the five seen surface families. We then evaluated that exact frozen feature only on C-test indirect rows using AUC, positive-minus-negative activation margin, and same-fact seen-to-indirect activation stability. No indirect row influenced feature selection.

Why: a C feature is more convincing if it retains the same concept selectivity under a surface family never used to choose it.

What we found: Europe has zero C-test support in both seen and indirect rows, so it is explicitly unavailable for this split. Across the four supported concepts, z_C mean indirect AUC is 0.994 versus 0.788 for raw, and mean normalized same-fact activation stability is 0.658 versus 0.601. z_C matches raw on capital_of (both AUC 1.000), improves currency_of (0.994 versus 0.652), nearly matches atomic_number_of (1.000 versus 1.000), and strongly improves period_4 (0.982 versus 0.500). This supports held-out-S concept selectivity for the selected z_C features, with the stated support limitation. No model or SAE was trained.

## Step 44 — scaled frozen seen-to-indirect audit to all supported concepts

What we did: repeated the exact seen-family feature-selection and indirect-only evaluation for every supported C-test concept: 9 relations, 10 reusable values, and 219 subjects. Relations required at least five indirect facts; values required at least ten all-C-test facts and five indirect facts; subjects required at least three indirect facts. We bootstrapped paired raw-versus-z_C differences over concepts with 5,000 fixed-seed resamples.

Why: a handful of hand-picked concepts cannot establish whether the held-out-S feature advantage generalizes across the controlled semantic structure.

What we found: across 238 concepts, raw mean indirect AUC is 0.692 and z_C mean AUC is 0.899, for a paired difference of +0.208 with 95% bootstrap CI [+0.179, +0.237]. Mean activation stability is 0.257 raw and 0.646 z_C, a paired difference of +0.388 with CI [+0.347, +0.430]. The strongest and most precise gain is for subjects (AUC +0.220, CI [+0.190, +0.250]); relations also improve (+0.129, CI [+0.027, +0.250]). Reusable values show no clear difference (+0.003, CI [-0.087, +0.091]) with only 10 concepts. No model, SAE, loss, or data changed.

## Step 45 — tested two additional held-out surface families

What we did: created one conversational family and one syntactically reordered family for the same 657 C-test facts. We extracted activations through frozen Gemma, the frozen non-adversarial partition, and frozen raw/z_C Top-k SAEs. Feature selection remained restricted to the original five seen families; each new family was evaluated separately across the same 238 supported concepts.

Why: the indirect-family result could otherwise be specific to one held-out wording style rather than evidence of broader surface invariance.

What we found: conversational rows give raw mean AUC 0.711 and z_C 0.897, a paired advantage of +0.186 (95% CI [+0.160, +0.213]); stability improves by +0.368 (CI [+0.330, +0.406]). Reordered rows give raw AUC 0.699 and z_C 0.911, advantage +0.212 (CI [+0.185, +0.240]); stability improves by +0.421 (CI [+0.385, +0.460]). Subject-level gains are strong in both families; reusable-value intervals remain inconclusive with ten values. No model, loss, SAE, or data used for training changed.

## Step 46 — audited RAVEL pairability

What we did: read only the official RAVEL entity–attribute JSON and prompt-template JSON files. We defined C as the published entity type, entity label, attribute, and value; S as the exact template scoped to entity type and attribute. We counted same-C/different-S positive pairs, different-C/same-S negative pairs, and the source’s published test templates as whole held-out S families.

Why: RAVEL is usable for the planned controlled contrastive setup only if both pair directions are available at scale and whole templates can remain unseen during training.

What we found: after rejecting 10 missing values, 34,535 explicit-value facts and 1,182 scoped templates remain. They form 50,871,345 same-C/different-S positive pairs and 2,077,820,997 different-C/same-S negative pairs. The published split has 282 fully held-out test templates, while every retained fact also has both train and test template coverage. This passes the structural pairability gate. RAVEL does not provide canonical entity IDs or fact-level provenance beyond its archive files, and 5,545 retained values are delimited/compound strings; both must be filtered or explicitly scoped before any strict C-matrix or training stage. No prompts were generated and no model was trained.

## Step 47 — built the strict RAVEL fact/template subset

What we did: constructed facts from RAVEL entity attributes and stored source templates separately. We retained only scalar, non-compound values; required one renderable entity placeholder per template; removed fact-template assignments where the rendered prompt already contained its target value; used published train and val templates only as training S; and reserved published test templates as held-out S. Entity train/val/test assignments remain exactly the published RAVEL split. The dynamic sampler chooses pairs from these fact/template lists and never stores a pair list.

Why: the external test requires same-fact/different-template positives and different-fact/same-template negatives without answer leakage, held-out template leakage, or entity leakage.

What we found: 28,204 facts and 975 templates pass the strict construction. Every retained fact has at least two clean train/val templates and one clean held-out test template. The retained entity counts are 14,122 train, 6,770 val, and 7,312 test facts. Dynamic pairability is substantial in every entity split; train alone has 8,591,110 positives and 236,580,897 negatives. Validation found zero entity split leakage and zero held-out-template IDs in training lists. Rejections were logged: 5,545 compound values, 10 missing values, 141,790 answer-leaking fact-template assignments, 562 facts with fewer than two clean training templates, and 224 without a clean held-out template. City Timezone, physical-object Texture, and verb Pronunciation have zero retained facts under this strict compound-value rule. No pair list, text dataset, frozen activation, or model training was created.

## Step 48 — trained and audited the frozen RAVEL C/S partition

What we did: used frozen Gemma-2-2B layer-8 masked-mean activations and trained the existing non-adversarial 2304-to-128+128 partition for the same 30 epochs, batch 256, AdamW settings, and contrastive objectives. To keep the previous training budget, we deterministically selected three published train/val templates per C-train fact; 11 sampled rows that could not form a same-template negative were removed. Training used 42,355 rows from 14,120 facts and 718 templates. Evaluation used only 21,541 rows from published C-test entities and published test templates. No test template entered training. No decoder, reconstruction, adversary, swap loss, or SAE was used.

Why: this is the external validity gate: z_C should retain attribute/fact information while reducing template information, and z_S should show the reciprocal pattern, under simultaneously unseen entities and templates.

What we found: z_S shows the expected surface specialization: held-out template identity is 1.000 in the frozen diagnostic probe and same-template/different-fact retrieval is R@1 0.996. But z_C is not cleanly template-muted: held-out template identity remains 0.926 despite an attribute accuracy of 0.674 (chance 0.053). z_C same-fact retrieval across held-out templates is R@1 0.268, R@5 0.646, and MRR 0.432. z_S still retains attribute accuracy 0.487. Both blocks have non-collapsed ranks (z_C entropy effective rank 29.84; z_S 27.76), but the required clean directional partition is not reproduced externally. Therefore no RAVEL SAE was trained. The known RAVEL limitations remain: label/file-level provenance and structural rather than fully pragmatic template verification.

## Step 49 — preflighted and reran RAVEL with prompt+published-answer views

What we did: before the full rerun, audited 200 fixed facts across all retained relations and both train/test entity splits. Each audit fact used two source templates and appended its published value verbatim after the rendered template. The 400 views passed: every value was present after append, no value occurred in the template before append, same-fact templates preserved the same source fact fields, training negatives existed, and published test templates stayed held out. We then reran the same frozen Gemma layer-8, 128+128 non-adversarial partition with the identical training rows, optimizer, epoch count, dimensions, and pair logic. Only the view text changed from template-only to rendered template plus published value.

Why: RAVEL template-only prompts omit the answer value, whereas the controlled C definition includes entity, attribute, and value. Appending the published value makes each positive view explicitly carry its full C fact.

What we found: z_C held-out same-fact/test-template retrieval improves from R@1 0.268 to 0.561 (MRR 0.432 to 0.674), which shows the added value makes factual matching easier. But the partition still fails the required specialization gate: z_C held-out template identity is 0.890 (chance 0.004) and z_S retains attribute accuracy 0.492 (chance 0.053). z_C attribute accuracy is 0.485; z_S same-template/different-fact retrieval remains very high at R@1 0.997. Both blocks remain non-collapsed (z_C entropy effective rank 33.65; z_S 27.82). No SAE was trained because the external C/S partition is still not clean. The RAVEL label/file-level provenance and structural-template limitations remain explicit.

## Step 50 — audited FLORES language coverage with frozen Gemma

What we did: used the official FLORES-200 dev split, a fixed first-200-sentence sample, frozen Gemma-2-2B layer-8 masked-mean activations, and a pre-registered ten-language pool: English, French, Spanish, German, Russian, Modern Standard Arabic, Hindi, Simplified Chinese, Swahili, and Turkish. We checked nonempty text coverage, tokenizer unknown-token fraction, and retrieval of the aligned same sentence from each language into an English bank. No partition, contrastive loss, SAE, or language holdout split was trained or selected.

Why: an external C/S experiment needs source-aligned translations and evidence that the frozen encoder represents the selected languages well enough before treating language as surface variation.

What we found: all ten languages have 200/200 nonempty translations and zero tokenizer unknown tokens. Same-sentence English retrieval is strong for every language: R@1 ranges from 0.930 for Swahili to 1.000 for French and Spanish; Arabic and Hindi are both 0.960. This passes the frozen-coverage gate. It supports proceeding to a predeclared train/held-out language split, but does not itself establish partitioning or ordinary within-English paraphrase invariance. No model training occurred.

## Step 51 — ran the fixed FLORES language partition

What we did: fixed eight training languages (English, French, Spanish, German, Russian, Hindi, Swahili, Turkish) and held out Arabic and Simplified Chinese completely. We also fixed the 200 aligned sentence identities into 140 train, 30 val, and 30 test sentences. Using frozen Gemma layer-8 activations, we trained the unchanged 128+128 non-adversarial partition for 30 epochs: same sentence/different training language is the z_C positive and different sentence/same training language is the negative; z_S uses the reverse. No reconstruction, adversary, swap loss, or SAE was used.

Why: this tests the existing partition method on natural aligned translations while evaluating simultaneously unseen sentences and languages.

What we found: z_C retrieves Arabic-to-Chinese held-out sentence identities strongly (R@1 0.967, MRR 0.983), and z_S is perfectly language-decodable while its sentence retrieval is low (R@1 0.100). However z_C still has high held-out language decodability (0.867 versus raw 1.000), and z_S is low rank (entropy effective rank 3.82). z_C is healthier (effective rank 15.91), but the full partition gate does not pass. No SAE was trained. This validates a directional language/content split, not complete language muting or general paraphrase invariance.

## Step 52 — trained matched FLORES Top-k SAEs and audited feature stability

What we did: after deterministically rerunning the exact FLORES partition only to save its raw, z_C, and z_S arrays, trained matched 4x Top-k SAEs with k=64 for 30 epochs on the same 1,120 C-train/training-language rows. We then compared each SAE feature's activation consistency for 30 held-out C-test sentence pairs: Arabic and Chinese translations of the same sentence.

Why: the external feature claim would require z_C SAE features to be more stable across held-out translations than raw, while z_S features should be less meaning-stable and more language-specific.

What we found: the requested pattern does not hold. Raw SAE has mean cross-language feature consistency 0.811 and 81.3% of features above 0.5 consistency. z_C is lower at 0.526 and 60.9%; z_S is 0.553 and 54.9%. Reconstruction MSE is 0.008 raw, 0.047 z_C, and 0.028 z_S. Thus this small FLORES run does not provide the hoped-for external SAE validation. It is reported as a negative result; no further model changes were made.

## Step 53 — checked FLORES SAE artifact availability

What we did: inspected the saved FLORES artifacts before attempting the requested sentence-selectivity versus language-selectivity audit.

Why: this analysis requires the frozen SAE encoder weights or per-example sparse feature activations; aggregate reconstruction and consistency statistics cannot recover individual feature responses.

What we found: raw Gemma, z_C, z_S, sentence IDs, language IDs, and split arrays were saved, but the FLORES Top-k run did not persist SAE checkpoints or sparse activations. Therefore the requested frozen-feature audit cannot be computed without an explicitly authorized exact persistence rerun of the three SAE jobs. No SAE, partition, or other model was retrained.

## Step 54 — persisted and reproduced the exact FLORES SAEs

What we did: under explicit authorization, reran the three exact raw, z_C, and z_S Top-k SAE jobs with k=64, 4x expansion, 30 epochs, the same seed, preprocessing, optimizer, and train rows. We saved checkpoints and sparse Top-k indices/values for every FLORES row, with aligned sentence, language, and split arrays.

Why: the requested feature-selectivity audit needs frozen per-example activations. The rerun had to reproduce the original aggregate results before those activations could be trusted.

What we found: reproduction passed exactly for reconstruction MSE, mean L0, mean cross-language consistency, and the fraction above 0.5 consistency for all three SAEs. No new setting or tuning was introduced.

## Step 55 — audited frozen FLORES SAE sentence and language selectivity

What we did: on 30 C-test sentences across all ten languages, compared each feature's between-sentence-to-within-sentence variance ratio against its between-language-to-within-language ratio. A feature is sentence-oriented only when its sentence ratio is larger than its language ratio and above one; language-oriented uses the reciprocal rule. Arabic-to-Chinese consistency was then computed only for sentence-oriented features.

Why: raw cross-language stability can be high for a feature that fires uniformly, so the stronger test requires both selectivity and invariance.

What we found: raw SAE is mostly mixed or unselective (1.3% sentence-oriented; 0.1% language-oriented). SAE(z_C) is strongly sentence-oriented (71.9%, 368/512 features), with no language-oriented features; its sentence-oriented features have mean Arabic-to-Chinese consistency 0.580. SAE(z_S) is strongly language-oriented (59.4%, 304/512), with no sentence-oriented features. This is the requested selectivity-aware directional result, while retaining the earlier caveat that it uses a small 30-sentence held-out evaluation set.
## Step 56 — ConCA comparison setup

What we did: prepared a separate run of the official ConCA BatchNorm architecture on frozen FLORES raw Gemma layer-8 activations and the frozen `z_C` block.

Why: this tests whether the frozen invariant block helps a non-Top-k concept-extraction method, without changing ConCA or retraining the partition.

What we found: the official repository distinguishes ConCA normalization architectures from SAE baselines. Its supplied ConCA runner uses `AutoEncodeBatchNorm` with `PAnnealSoftPlusTrainer`; the procedure is fixed before execution. FLORES uses language as S, so there are no held-out prompt-template families in this comparison.

## Step 57 — Frozen ConCA comparison completed

What we did: ran the fixed official ConCA procedure separately on raw Gemma layer-8 activations and frozen `z_C`. The partition was not retrained and ConCA was not jointly trained with it.

Why: this directly tests whether the invariant block is useful as preprocessing for an independent concept-component method.

What we found: on the 30 held-out FLORES sentence concepts evaluated from held-out Arabic/Chinese, `z_C` ConCA improved mean concept AUC from 0.965 to 0.982 and mean component stability from 0.918 to 0.954. It also reduced language leakage: language-selective components fell from 48.9% to 0%, and a language probe fell from 99.3% to 28.0%. The raw ConCA dictionary was mixed (41.0% sentence-selective, 48.9% language-selective); `z_C` ConCA was entirely sentence-selective by the pre-existing audit criterion. This is a FLORES language-held-out result; template-held-out generalization is not available in this dataset.

## Step 58 — WildJailbreak pairability audit

What we did: checked the public WildJailbreak metadata, the accompanying paper, and the workspace for the row-level fields needed to form honest same-intent/different-attack and different-intent/same-attack pairs.

Why: pair counts and held-out attack-family splits must be verified before any defensive-jailbreak model training.

What we found: the published data files are gated and no local copy exists. The public description confirms harmful/benign and vanilla/adversarial subsets, but does not permit counting intent groups, per-intent attack realizations, attack-family reuse, ambiguous changes of intent, or valid pairs. The audit therefore does not pass; no training was performed and no counts were invented.

## Step 59 — JailbreakBench pairability audit

What we did: inspected the public JBB-Behaviors schema and public attack artifacts, then counted only nonempty records for one fixed target model. This prevents target-model differences from being treated as attack-style differences.

Why: we need verified same-behavior/different-method positives and different-behavior/same-method negatives before safety robustness training.

What we found: the artifacts preserve canonical behavior and goal fields and expose attack-method provenance. The script reports the exact valid counts and full-method holdout candidates. It does not treat non-verbatim goal text in an adversarial prompt as an intent change; that requires a separate semantic audit before any training.

Result: the strict audit retained 94 of 100 source behavior IDs after rejecting six IDs whose canonical behavior/goal fields conflicted across artifacts. With a fixed Vicuna target model, five methods supplied 454 valid records: DSN=94, GCG=94, JBC=94, PAIR=78, and prompt-with-random-search=94. This gives 876 unordered same-behavior/different-method positives and 20,487 unordered different-behavior/same-method negatives. DSN, GCG, JBC, and prompt-with-random-search each cover all 94 retained behaviors and can be wholly held out. The source provides 100 benign control behaviors. No model training was performed.

## Step 60 — JailbreakBench blinded semantic-audit packet

What we did: prepared a deterministic 120-record packet containing only canonical goals and adversarial prompts. It samples 24 non-verbatim records from each of the five attack methods and hides method and behavior IDs in a separate key file.

Why: semantic preservation cannot be established by canonical metadata or string overlap. An independent reviewer must judge whether the prompt retains the specified intent.

What we found: the packet is ready for annotation with `SAME_INTENT`, `PARTIAL_DRIFT`, `CHANGED_INTENT`, or `AMBIGUOUS`. No labels have been inferred and no model training was performed.

Correction: the public artifacts do not support equal non-verbatim sampling across all methods. DSN and JBC have zero eligible records, and all four GCG candidates are removed by the strict canonical-conflict rule. The final 100-record packet uses 50 PAIR and 50 prompt-with-random-search examples. DSN, GCG, and JBC will be reported as not estimable from this non-verbatim audit.

Pre-registered rule: only `SAME_INTENT` records may form positive pairs. A sampled method is eligible for training or full holdout only if at least 95% of its blinded sample is `SAME_INTENT` and none is `CHANGED_INTENT`; `PARTIAL_DRIFT` and `AMBIGUOUS` records are rejected.

## Step 62 — PAIR semantic-review export

What we did: exported the 50 already sampled PAIR records into a standalone blinded CSV with only canonical goal, adversarial prompt, `YES`/`NO`/`AMBIGUOUS` label field, and a reviewer-note field.

Why: PAIR is the intended held-out transformation and needs semantic validation before it can be used as an unseen attack family.

What we found: the export contains exactly 50 rows and no attack-method metadata. No labels were added and no model training was performed.

## Step 61 — JailbreakBench mechanism audit

What we did: prepared a deterministic, method-stratified structural audit of 20 strict records per attack method. It measures goal copying, prefix/suffix additions, rewrites, role-play markers, and a conservative obfuscation heuristic without printing or redistributing attack prompts.

Why: attack algorithm names are not a valid nuisance definition. We need observed surface mechanisms to decide whether the dataset supports an unseen-transformation claim.

What we found: pending the terminal audit report. Structural labels are kept separate from independent intent-preservation labels.

Result: DSN and GCG both preserve the canonical goal verbatim and append a short suffix (18/20 and 20/20 samples respectively). JBC preserves the goal verbatim and adds a role-play prefix (20/20). Prompt-with-random-search preserves the goal verbatim and adds role-play prefix and suffix wrappers (20/20). PAIR is the only sampled method that substantially rewrites the request (19/20), usually with role-play markers (17/20). No sampled method triggered the conservative encoding/obfuscation heuristic. Therefore the five named algorithms reduce to three observed transformation mechanisms: suffix perturbation, wrapper/role-play, and rewritten role-play. JailbreakBench can support a limited downstream proof of concept but not a broad claim about diverse unseen jailbreak transformations.

Correction to Step 60: literal raw-string non-verbatim matching overstated transformation diversity. Under whitespace/case normalization, sampled prompt-with-random-search examples copy the canonical goal verbatim; its raw mismatch was formatting, not paraphrase.
