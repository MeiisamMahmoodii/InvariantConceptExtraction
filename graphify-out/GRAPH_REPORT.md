# Graph Report - InvariantConceptExtraction  (2026-08-25)

## Corpus Check
- Corpus is ~13,555 words - fits in a single context window. You may not need a graph.

## Summary
- 178 nodes · 242 edges · 20 communities (16 shown, 4 thin omitted)
- Extraction: 88% EXTRACTED · 11% INFERRED · 1% AMBIGUOUS · INFERRED: 26 edges (avg confidence: 0.82)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Dataset Construction Pipeline
- Surface Validation & Audits
- Representation Diagnostics
- Frozen Repr & Contrastive Design
- Relation Feasibility & Coverage
- Factual Matrix Building
- Contrastive Encoder Training
- Representation Audit Probes
- Iterative Linear Erasure
- Surface Generation Scripts
- Contrastive Encoder Analysis
- Relation Feasibility Script
- Build Factual Matrix Script
- CS Sandbox Generator
- Semantic Audit Script
- Activation Extraction
- Graphify Detect Output
- Factor SAE Package

## God Nodes (most connected - your core abstractions)
1. `Factual C-Matrix` - 14 edges
2. `Strict V1 Surface Dataset` - 9 edges
3. `Layer-8 Contrastive Encoder` - 9 edges
4. `Controlled Surface Dataset` - 8 edges
5. `main()` - 7 edges
6. `Analyze Representation Audit` - 7 edges
7. `Controlled C×S Dataset Validator` - 6 edges
8. `Generate Controlled Surface Dataset` - 6 edges
9. `S Surface Families` - 6 edges
10. `Representation Diagnostic Report` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Relation Coverage Acceptance` --conceptually_related_to--> `Factual C-Matrix`  [INFERRED]
  Report/relation_coverage_report.md → code/build_factual_matrix.py
- `Strict V1 Surface Dataset` --shares_data_with--> `Factual C-Matrix`  [INFERRED]
  Report/FULL_SURFACE_AUDIT.md → code/build_factual_matrix.py
- `Factual C-Matrix` --implements--> `V1 Six-Relation Set`  [INFERRED]
  code/build_factual_matrix.py → Report/RELATION_FEASIBILITY_AUDIT.md
- `Controlled Surface Dataset` --implements--> `Strict V1 Surface Dataset`  [INFERRED]
  code/generate_controlled_surface_dataset.py → Report/FULL_SURFACE_AUDIT.md
- `Controlled Surface Dataset Validator` --implements--> `S-Train vs S-Test Family Split`  [EXTRACTED]
  code/validate_controlled_surface_dataset.py → Report/DATASET_CARD.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Factual Dataset Construction Pipeline** — code_run_relation_feasibility_module, code_build_factual_matrix_module, code_generate_controlled_surface_dataset_module, code_extract_gemma2_activations_module [INFERRED 0.85]
- **Contrastive Train And Audit Flow** — code_train_contrastive_encoder_module, contrastive_encoder_layer8_checkpoint, code_analyze_contrastive_encoder_module, report_contrastive_representation_diagnostic_report [EXTRACTED 1.00]
- **Raw Activation Diagnostic Suite** — code_analyze_representation_audit_module, code_run_iterative_linear_erasure_module, diagnostic_linear_probes, gemma2_2b_activations [INFERRED 0.80]
- **Strict V1 Dataset Construction Pipeline** — concept_relation_coverage, factual_c_matrix, surface_template_candidates, concept_strict_v1 [INFERRED 0.85]
- **Frozen Representation Diagnostic Suite** — concept_gemma2_2b, concept_linear_probes, concept_subspace_removal, concept_linear_erasure [EXTRACTED 0.95]
- **C×S Factor Validation Pattern** — code_validate_dataset_main, code_validate_controlled_surface_dataset_main, code_validate_surface_candidates_main, concept_s_families [INFERRED 0.80]

## Communities (20 total, 4 thin omitted)

### Community 0 - "Dataset Construction Pipeline"
Cohesion: 0.11
Nodes (23): Create Final Semantic Audit, Generate Controlled Surface Dataset, Generate CS Dataset Sandbox, Render Surface Candidates, load(), Factual C-Matrix Validator, Fail if the factual C-matrix violates its approved structure., C Factors (domain, relation, subject, value) (+15 more)

### Community 1 - "Surface Validation & Audits"
Cohesion: 0.12
Nodes (22): Controlled Surface Dataset Validator, Fail if full controlled surface generation violates the approved constraints., Controlled C×S Dataset Validator, Validate the controlled C×S dataset; exit non-zero on a hard-constraint failure., Surface Template Candidates Validator, Invalid Unresolved-Label Dataset Archive, Proposition Preservation (relation, subject, value), S Surface Families (+14 more)

### Community 2 - "Representation Diagnostics"
Cohesion: 0.14
Nodes (21): Analyze Contrastive Encoder, Analyze Representation Audit, Extract Gemma2 Activations, Run Iterative Linear Erasure, Train Contrastive Encoder, InfoNCE Contrastive Encoder, Contrastive Encoder Layer-8 Checkpoint, Diagnostic Linear Probes (+13 more)

### Community 3 - "Frozen Repr & Contrastive Design"
Cohesion: 0.14
Nodes (20): Linear C/S Separability Finding, Fixed Diagnostic Layers 5/8/13/21, Frozen Gemma 2 2B Representations, InfoNCE Contrastive Objective, Layer-8 Contrastive Encoder, Iterative Linear Erasure Diagnostic, Diagnostic Linear Probes, Masked-Mean Pooling over Non-Padding Tokens (+12 more)

### Community 4 - "Relation Feasibility & Coverage"
Cohesion: 0.21
Nodes (12): Deferred Multi-Valued Relations, Geography Domain (148 subjects), Relation Coverage Acceptance, Royal Society of Chemistry Period Source, Science Domain (118 subjects), Single-Valued Relation Filter, V1 Six-Relation Set, Wikidata Structured Source (+4 more)

### Community 5 - "Factual Matrix Building"
Cohesion: 0.24
Nodes (10): C Domain and Relation Hierarchy, Build Factual Matrix, Run Relation Feasibility, Geography and Science Domains, Rationale: Period From RSC Not Atomic Number, Rationale: In-batch Negatives Preserve C Hierarchy, Relation Coverage Audit, Factual Matrix Report (+2 more)

### Community 6 - "Contrastive Encoder Training"
Cohesion: 0.27
Nodes (6): build_pairs(), Encoder, load_train_rows(), main(), Train a contrastive encoder on frozen layer-8 Gemma-2-2B activations. Train…, Pairs: (anchor_idx, positive_idx) — same fact, different S_train family.

### Community 7 - "Representation Audit Probes"
Cohesion: 0.42
Nodes (8): cosine(), fit_probe(), main(), probe_accuracy(), project_out(), Analyze frozen activations with pair distributions and diagnostic linear probes…, sample_pairs(), summary()

### Community 8 - "Iterative Linear Erasure"
Cohesion: 0.50
Nodes (8): accuracy(), fit_probe(), iterative_erasure(), learned_basis(), main(), metrics(), Iteratively erase linear S/C probe subspaces from frozen activations; no…, remove()

### Community 9 - "Surface Generation Scripts"
Cohesion: 0.38
Nodes (5): main(), Generate the full controlled surface dataset from the approved factual matrix., main(), Render a 90-row surface-template preflight; not the final text dataset., render()

### Community 10 - "Contrastive Encoder Analysis"
Cohesion: 0.53
Nodes (5): cosine_summary(), main(), pairs_for_test(), probe_accuracy(), Audit frozen contrastive embeddings on held-out subjects.

### Community 11 - "Relation Feasibility Script"
Cohesion: 0.60
Nodes (5): canonical_value_id(), main(), Fetch and audit six factual relations; this script creates no text dataset., rsc_period(), value()

### Community 12 - "Build Factual Matrix Script"
Cohesion: 0.60
Nodes (4): main(), Build the approved factual C-matrix only; no text or S realizations are created., read_relation(), split()

### Community 13 - "CS Sandbox Generator"
Cohesion: 0.60
Nodes (4): main(), Build the controlled C×S sandbox from a local Wikidata snapshot., text_for(), value()

## Ambiguous Edges - Review These
- `Controlled Surface Validation Report` → `CS Sandbox Validation Report`  [AMBIGUOUS]
  Report/validation_report.json · relation: conceptually_related_to
- `Layer-8 Contrastive Encoder` → `Partitioned SAE Baseline Goal`  [AMBIGUOUS]
  pyproject.toml · relation: conceptually_related_to

## Knowledge Gaps
- **26 isolated node(s):** `factor-sae`, `Create Final Semantic Audit`, `Subject-Disjoint C Splits`, `Geography and Science Domains`, `Iterative Linear Erasure Report` (+21 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Controlled Surface Validation Report` and `CS Sandbox Validation Report`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Layer-8 Contrastive Encoder` and `Partitioned SAE Baseline Goal`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `Factual C-Matrix` connect `Dataset Construction Pipeline` to `Surface Validation & Audits`, `Relation Feasibility & Coverage`, `Factual Matrix Building`?**
  _High betweenness centrality (0.182) - this node is a cross-community bridge._
- **Why does `Strict V1 Surface Dataset` connect `Surface Validation & Audits` to `Dataset Construction Pipeline`, `Frozen Repr & Contrastive Design`?**
  _High betweenness centrality (0.180) - this node is a cross-community bridge._
- **Why does `Controlled Surface Dataset` connect `Dataset Construction Pipeline` to `Surface Validation & Audits`, `Representation Diagnostics`?**
  _High betweenness centrality (0.159) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `Factual C-Matrix` (e.g. with `Relation Coverage Acceptance` and `Strict V1 Surface Dataset`) actually correct?**
  _`Factual C-Matrix` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `Strict V1 Surface Dataset` (e.g. with `Factual C-Matrix` and `Controlled C×S Capital Dataset (500 rows)`) actually correct?**
  _`Strict V1 Surface Dataset` has 4 INFERRED edges - model-reasoned connections that need verification._