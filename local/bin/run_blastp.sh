#!/usr/bin/env bash
set -euo pipefail

outgroup="$1"          # e.g. cin, csa
query="$2"             # e.g. hsa, gga
threads="${3:-1}"      # optional
evalue="${4:-1e-5}"    # optional
max_targets="${5:-100}" # optional

db_fa="${outgroup}_peptides.protein_coding.fa"
q_fa="${query}_peptides.protein_coding.fa"
db_prefix="blastdb/${outgroup}"
out_tsv="blast_${outgroup}_${query}.raw.tsv"

mkdir -p blastdb blastdb_results

# Build DB only if missing (pin/phr/psq are created)
if [[ ! -s "${db_prefix}.pin" ]]; then
  echo "Building BLAST DB for ${db_fa}..."
  makeblastdb -in "${db_fa}" -dbtype prot -out "${db_prefix}" -parse_seqids -taxid 1
fi

# Header
printf "qseqid\tsseqid\tbitscore\tevalue\tpident\tlength\tqcovs\n" > "${out_tsv}"

# Run blastp and append
blastp \
  -query "${q_fa}" \
  -db "${db_prefix}" \
  -evalue "${evalue}" \
  -max_hsps 1 \
  -max_target_seqs "${max_targets}" \
  -num_threads "${threads}" \
  -outfmt "6 qseqid sseqid bitscore evalue pident length qcovs" >> "${out_tsv}"

# Optionally also copy/move to blastdb_results
# cp "${out_tsv}" blastdb_results/
