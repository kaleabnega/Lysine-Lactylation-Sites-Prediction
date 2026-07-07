# Structure-Aware Kla Prediction Plan

This experiment branch is for a publishable architectural extension of
PCBert-Kla. The current best-performing model improves the baseline with
AdamW fine-tuning and seed ensembling, but its neural architecture is still the
same core PCBert-Kla design. A paper-level contribution should add a new model
component with biological motivation.

## Proposed Model

Working name:

```text
Struct-PCBert-Kla
```

Base PCBert-Kla uses:

```text
51-aa sequence window -> ProtBert CLS embedding -> 1024 dimensions
physicochemical descriptors -> 27 dimensions
concat(1024, 27) -> attention-based classifier
```

The structure-aware model adds a third branch:

```text
local 3D lysine microenvironment graph -> graph encoder -> structural embedding
```

The final architecture becomes:

```text
Sequence branch:
  51-aa sequence window
  -> ProtBert
  -> CLS sequence embedding

Physicochemical branch:
  molecular weight, pI, amino acid composition, secondary structure fraction,
  hydrophobicity, net charge

Structure branch:
  residues spatially near the central lysine
  -> residue contact graph
  -> GCN/GAT/Graph Transformer
  -> structure embedding

Fusion:
  concat(sequence embedding, physicochemical vector, structure embedding)
  -> gated/attention fusion classifier
  -> Kla probability
```

This is a real neural architecture change because the model receives a new
structured input and learns a structural representation before classification.

## Biological Justification

Lysine lactylation is a site-specific chemical modification. Sequence context
matters, but the real modification site exists in a folded 3D protein, where
nearby residues in space may not be adjacent in the sequence. A structure-aware
branch can encode the local chemical/spatial environment around lysine:

- solvent accessibility around the lysine
- nearby charged residues
- local residue contact pattern
- secondary-structure context
- confidence of predicted structure, such as pLDDT if using AlphaFold outputs

This gives a stronger publication argument than optimizer tuning or ensembling
alone.

## Current Data Blocker

The cloned PCBert-Kla benchmark currently contains only anonymized 51-residue
windows:

```text
Protein 0|26|1
KKDAEGKSTTNQEKSRKKNFMMTLGKAKSKQKRSLQHTRRVLKGHIDRTKR
```

It does not provide:

- original rice protein accessions
- full-length protein sequences
- original site positions in full proteins
- PDB/AlphaFold structure identifiers

That means we cannot yet build a high-quality full-protein structural graph.
Predicting a 3D structure for each isolated 51-aa peptide window is possible as
a prototype, but it is biologically weaker because the native full-protein
folding context is missing.

## Required Data For A Strong Version

Before implementing the real structure branch, recover or reconstruct:

```text
record_id
protein_accession
full_protein_sequence
full_protein_lysine_position
label
51-aa_window
```

Then obtain structures through one of:

```text
AlphaFold DB / rice proteome structures
ESMFold predicted full-length structures
ColabFold/AlphaFold for missing proteins
```

## Structure Feature Extraction

For each lactylation candidate site:

1. Load the full-protein structure.
2. Locate the central lysine residue.
3. Select residues within a spatial cutoff, e.g. 8-12 Angstroms, or the top-k
   nearest residues.
4. Build a residue graph:

```text
node = residue
edge = residues within distance cutoff
```

Candidate node features:

```text
amino acid one-hot or embedding
relative sequence position to lysine
distance to central lysine
secondary structure class
solvent accessibility
pLDDT / structure confidence
charge / polarity / hydrophobicity class
```

Candidate edge features:

```text
C-alpha distance
sequence separation
contact indicator
radial basis expansion of distance
```

## Model Variants To Compare

Minimum ablation table:

```text
1. PCBert-Kla replication
2. PCBert-Kla + AdamW
3. PCBert-Kla + AdamW + seed ensemble
4. Struct-PCBert-Kla without ensemble
5. Struct-PCBert-Kla + seed ensemble
6. Struct-PCBert-Kla without physicochemical features
7. Struct-PCBert-Kla without structure branch
```

The key publication claim should come from comparing variants 2 and 4, because
that isolates the architectural contribution.

## First Implementation Milestone

Do not start by predicting structures. First recover the original protein IDs
or full-length sequences from the benchmark source or associated papers.

Milestone checklist:

```text
[x] Inspect original DeepKla repository data files.
[x] Build a first exact window-to-proteome mapping script.
[x] Test exact mapping against UniProt japonica and Ensembl Plants rice FASTA.
[x] Recover the open-access DeepKla paper text and supplementary notes from PMC.
[x] Identify the experimental source papers for the training and independent data.
[x] Test exact mapping against NCBI RefSeq rice IRGSP-1.0 protein FASTA.
[x] Recover Meng et al. 2021 ACS Table S1 from the rice source paper.
[x] Build a rice source site table with protein accession and full-site position.
[ ] Audit why the public PCBert-Kla/DeepKla training windows do not match Meng
    et al. 2021 Table S1 by exact sequence.
[ ] Verify every 51-aa window maps uniquely to a full protein.
[ ] Download or predict full-length structures.
[ ] Create cached local graph files per site.
[ ] Add a structure graph encoder branch to the model.
[ ] Run ablations against the AdamW PCBert-Kla baseline.
```

## Exact Mapping Attempt

A reusable mapping script is available at:

```text
experiment-1/structure_aware_kla/scripts/map_windows_to_proteome.py
```

Example command:

```bash
python experiment-1/structure_aware_kla/scripts/map_windows_to_proteome.py \
  --proteome-fasta /path/to/rice_proteome.fa \
  --output-dir experiment-1/structure_aware_kla/results/window_mapping
```

For a single benchmark file:

```bash
python experiment-1/structure_aware_kla/scripts/map_windows_to_proteome.py \
  --windows baselines/PCBert-Kla-original/data/test.csv \
  --proteome-fasta /path/to/botrytis_uniprot.fasta \
  --output-dir experiment-1/structure_aware_kla/results/botrytis_test_mapping
```

Supplementary XLSX tables can be converted to CSV without extra Python
packages:

```bash
python experiment-1/structure_aware_kla/scripts/extract_xlsx_table.py \
  /path/to/Table_1.XLSX \
  --sheet 'Table S1' \
  --header-row-contains 'Protein accession' \
  --header-row-contains 'Position' \
  --header-row-contains 'Subcellular localization' \
  --output experiment-1/structure_aware_kla/results/botrytis_kla_sites.csv
```

For source tables with modified peptide strings, validate whether benchmark
windows align to known modified peptides at the central lysine:

```bash
python experiment-1/structure_aware_kla/scripts/validate_windows_against_modified_peptides.py \
  --windows baselines/PCBert-Kla-original/data/test.csv \
  --site-table-csv experiment-1/structure_aware_kla/results/botrytis_kla_sites.csv \
  --output experiment-1/structure_aware_kla/results/botrytis_window_peptide_validation.csv
```

The full provenance audit can be reproduced with:

```bash
python experiment-1/structure_aware_kla/scripts/audit_dataset_provenance.py \
  --pcbert-train baselines/PCBert-Kla-original/data/train.csv \
  --pcbert-test baselines/PCBert-Kla-original/data/test.csv \
  --deepkla-train /path/to/DeepKla/data/upTrain.fa \
  --deepkla-test /path/to/DeepKla/data/fungiForTest.fa \
  --meng-table-s1 experiment-1/structure_aware_kla/results/meng_2021_table_s1_rice_lactylated_sites.clean.csv \
  --rice-proteome-fasta /path/to/oryza_sativa_japonica_uniprot.fasta \
  --botrytis-proteome-fasta /path/to/botrytis_b0510_uniprot.fasta \
  --output-json experiment-1/structure_aware_kla/results/dataset_provenance_audit.json \
  --output-md experiment-1/structure_aware_kla/results/dataset_provenance_audit.md
```

Tested sources:

```text
1. Original DeepKla GitHub data:
   https://github.com/linDing-group/DeepKla

2. UniProt Oryza sativa subsp. japonica proteins:
   https://rest.uniprot.org/uniprotkb/stream?compressed=false&format=fasta&query=%28organism_id%3A39947%29

3. UniProt Oryza sativa species-level proteins:
   https://rest.uniprot.org/uniprotkb/stream?compressed=false&format=fasta&query=%28organism_id%3A4530%29

4. Ensembl Plants Oryza sativa IRGSP-1.0 peptide FASTA:
   https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/current/fasta/oryza_sativa/pep/Oryza_sativa.IRGSP-1.0.pep.all.fa.gz

5. NCBI RefSeq Oryza sativa IRGSP-1.0 protein FASTA:
   https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/001/433/935/GCF_001433935.1_IRGSP-1.0/GCF_001433935.1_IRGSP-1.0_protein.faa.gz
```

Observed result:

```text
Original DeepKla upTrain.fa and fungiForTest.fa contain only labels and 51-aa
windows, not protein accessions.

Exact mapping against UniProt japonica:
  Proteins searched: 61,200
  Unique-window status counts:
    multiple_matches: 1
    no_match: 1,936

Exact mapping against Ensembl Plants IRGSP-1.0:
  Proteins searched: 42,582
  Unique-window status counts:
    multiple_matches: 1
    no_match: 1,936

Exact mapping against NCBI RefSeq IRGSP-1.0:
  Proteins searched: 42,580
  Unique-window status counts:
    multiple_matches: 1
    no_match: 1,936

Exact mapping of independent-test windows against UniProt Botrytis cinerea
B05.10:
  Proteins searched: 12,998
  Unique-window status counts:
    multiple_matches: 35
    unique_match: 314
```

## Recovered Source Evidence

DeepKla is available in PMC:

```text
DeepKla: An attention mechanism-based deep neural network for protein lysine
lactylation site prediction
DOI: 10.1002/imt2.11
PMCID: PMC10989745
GitHub: https://github.com/linDing-group/DeepKla
```

The DeepKla paper states that:

```text
Training positives:
  Rice lactylation data collected from Meng et al. 2021.

Training negatives:
  Other lysine residues from the same proteins excluding annotated Kla sites.

Window length:
  51 amino acids, centered on the candidate lysine.

Redundancy filtering:
  CD-HIT at 30% sequence similarity.

Independent test:
  273 Kla data in Botrytis cinerea from Gao et al. 2020, reduced to
  177 positive and 177 negative windows after the same processing criteria.
```

The two source papers are:

```text
Rice training source:
  Meng X, Baine JM, Yan T, Wang S. 2021.
  Comprehensive Analysis of Lysine Lactylation in Rice (Oryza sativa) Grains.
  Journal of Agricultural and Food Chemistry 69:8287-8297.
  DOI: 10.1021/acs.jafc.1c00760
  PMID: 34264677
  Reported source scale: 638 Kla sites across 342 proteins.

Independent test source:
  Gao M, Zhang N, Liang W. 2020.
  Systematic Analysis of Lysine Lactylation in the Plant Fungal Pathogen
  Botrytis cinerea.
  Frontiers in Microbiology 11:2615.
  DOI: 10.3389/fmicb.2020.594743
  PMCID: PMC7649125
  PRIDE/ProteomeXchange: PXD020746
```

The DeepKla PMC supplementary files contain architecture/sequence-representation
notes only. They do not contain the original rice protein accessions or full
site table. The public DeepKla GitHub benchmark files also contain only labels
and 51-aa windows.

The Botrytis independent-test source is recoverable. Its PMC supplementary
Table 1 contains:

```text
Protein accession
Position
Amino acid
Protein description
Gene name
Localization probability
PEP
Score
Modified sequence
Charge
Mass error [ppm]
```

When the DeepKla independent-test windows are mapped against the UniProt
Botrytis cinerea B05.10 proteome, every unique 51-aa test window maps to at
least one protein location. Most map uniquely; the multiple matches are expected
for repetitive/conserved windows.

Peptide-level validation against the Botrytis supplementary Table 1 gives:

```text
windows: 354
source modified peptides parsed: 268
positive windows with central-K peptide evidence: 170
positive windows without central-K peptide evidence: 7
negative windows with central-K peptide evidence: 5
negative windows without central-K peptide evidence: 172
```

The exception rows should be kept for audit. They may reflect the DeepKla
authors' filtering and relabeling steps, repeated sequence contexts, or source
database/version differences.

The ACS supplementary XLSX for Meng et al. 2021 has now been recovered:

```text
Source file:
  experiment-1/structure_aware_kla/source_data/meng_2021_rice_lactylome/jf1c00760_si_001.xlsx

Extracted Table S1:
  experiment-1/structure_aware_kla/results/meng_2021_table_s1_rice_lactylated_sites.clean.csv

Observed Table S1 integrity:
  data rows: 638
  unique accession-position-amino-acid sites: 638
  unique protein accessions: 342
  amino acid at every reported site: K
```

This matches the Meng et al. paper's reported source scale: 638 Kla sites
across 342 proteins. Current UniProt Oryza sativa subsp. japonica resolves most
of these source accessions:

```text
UniProt japonica proteins searched: 61,200
Table S1 accessions present: 335 of 342
valid K sites by accession and position: 626 of 638
reconstructed 51-aa source windows: 529
overlap with unique PCBert-Kla training windows: 0
```

Direct peptide-level validation also shows that Meng Table S1 does not explain
the public PCBert-Kla training windows:

```text
training windows: 3,487
Meng Table S1 modified peptides parsed: 627
positive training windows with central-K peptide evidence: 10
positive training windows without central-K peptide evidence: 1,710
negative training windows with central-K peptide evidence: 1
negative training windows without central-K peptide evidence: 1,766
```

This means Meng Table S1 is a valid recovered source lactylome, but it is not a
direct exact-sequence reconstruction of the public PCBert-Kla/DeepKla training
file currently in this repository.

An additional audit result is important: both the public training and independent
test windows map strongly to the UniProt Botrytis cinerea B05.10 proteome:

```text
Combined train + test mapping against Botrytis B05.10:
  Proteins searched: 12,998
  Unique-window status counts:
    multiple_matches: 250
    unique_match: 1,687

Training split only:
  unique_match: 1,607
  multiple_matches: 232

Independent-test split only:
  unique_match: 314
  multiple_matches: 35
```

This is unexpected under the paper-level narrative that the training set is rice
and the independent test set is Botrytis. It should be treated as a provenance
audit issue before using these benchmark windows as the basis for a publishable
structure-aware model.

A reproducible audit report is available at:

```text
experiment-1/structure_aware_kla/results/dataset_provenance_audit.md
```

## Current Interpretation

Exact mapping failed against UniProt rice, Ensembl Plants IRGSP-1.0, and NCBI
RefSeq IRGSP-1.0. Meng et al. 2021 Table S1 has now been recovered and gives a
clean rice lactylome site table, but reconstructed windows from that table do
not overlap the public PCBert-Kla training windows. Conversely, the public
training and independent-test windows map well to Botrytis B05.10.

Current interpretation:

```text
The recovered Meng et al. rice lactylome is reliable source biology, but the
public PCBert-Kla/DeepKla benchmark files are not currently traceable to it by
exact sequence. Do not claim a structure-aware model on the public benchmark is
using recovered rice full-protein structure until this provenance mismatch is
resolved.
```

The next recovery step should audit the DeepKla repository history, paper
processing scripts, and any generated files to determine whether the public
benchmark was transformed, mislabeled, or drawn from a different FASTA/source
than the described rice Table S1.

## Practical Prototype If Full IDs Cannot Be Recovered

If full-length proteins cannot be recovered, a weaker prototype can predict
structures for the 51-aa windows and build a local graph around position 26.
This can test the code path, but should be described cautiously:

```text
prototype structure branch based on predicted peptide-window conformation,
not native full-protein structure
```

This prototype should not be the main publishable claim unless full-protein
structure recovery fails and the limitation is clearly stated.
