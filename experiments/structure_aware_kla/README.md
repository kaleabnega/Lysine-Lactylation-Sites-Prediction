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
[ ] Find original DeepKla/PCBert-Kla full protein records.
[ ] Build a site table with protein accession and full-site position.
[ ] Verify every 51-aa window maps uniquely to a full protein.
[ ] Download or predict full-length structures.
[ ] Create cached local graph files per site.
[ ] Add a structure graph encoder branch to the model.
[ ] Run ablations against the AdamW PCBert-Kla baseline.
```

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
