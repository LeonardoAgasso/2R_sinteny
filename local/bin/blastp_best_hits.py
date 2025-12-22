#!/usr/bin/env python3
"""Download proteomes and compute best BLASTp hits for a vertebrate→outgroup pair.

This script downloads the Ensembl peptide FASTA for the requested species,
reduces each proteome to the longest isoform per gene (sequence IDs are set to
gene IDs), builds a BLAST protein database for the outgroup, runs BLASTp, and
writes a TSV with the best outgroup hit per vertebrate gene.

Usage (defaults to the pair requested by the user):
    python local/bin/blastp_best_hits.py --vertebrate hsa --outgroup csa
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from collections import defaultdict
from typing import Dict, Iterable, Iterator, Tuple


DATASET_ROOT = "/home/lagasso/projects/sinteny/dataset"

# Minimal species metadata for now; extend if needed.
SPECIES = {
    "hsa": {
        "category": "vertebrata",
        "download_url": "https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz",
    },
    "csa": {
        "category": "outgroups",
        "download_url": "https://ftp.ensembl.org/pub/current_fasta/ciona_savignyi/pep/Ciona_savignyi.CSAV2.0.pep.all.fa.gz",
    },
}


def download_file(url: str, dest: str, force: bool = False) -> str:
    """Download url to dest if missing."""
    if os.path.exists(dest) and not force:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url} -> {dest}", file=sys.stderr)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def gunzip_file(src_gz: str, dest: str, force: bool = False) -> str:
    """Decompress gzip file."""
    if os.path.exists(dest) and not force:
        return dest
    print(f"Decompressing {src_gz} -> {dest}", file=sys.stderr)
    with gzip.open(src_gz, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return dest


def fasta_records(path: str) -> Iterator[Tuple[str, str]]:
    """Yield (header, sequence) tuples from FASTA."""
    header = None
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if header:
        yield header, "".join(chunks)


def gene_from_header(header: str) -> str | None:
    """Extract Ensembl gene ID from a FASTA header."""
    m = re.search(r"gene:([A-Za-z0-9._-]+)", header)
    if m:
        return m.group(1)
    return None


def longest_per_gene(in_fasta: str, out_fasta: str, force: bool = False) -> str:
    """Keep the longest isoform per gene, writing headers as gene IDs."""
    if os.path.exists(out_fasta) and not force:
        return out_fasta
    print(f"Selecting longest isoform per gene from {in_fasta}", file=sys.stderr)
    best: Dict[str, Tuple[str, str]] = {}
    for header, seq in fasta_records(in_fasta):
        gene_id = gene_from_header(header)
        if not gene_id:
            continue
        prev = best.get(gene_id)
        if prev is None or len(seq) > len(prev[1]):
            prot_id = header.split()[0]
            best[gene_id] = (prot_id, seq)
    os.makedirs(os.path.dirname(out_fasta), exist_ok=True)
    with open(out_fasta, "w") as f_out:
        for gene_id, (prot_id, seq) in sorted(best.items()):
            f_out.write(f">{gene_id} protein={prot_id}\n")
            f_out.write(f"{seq}\n")
    return out_fasta


def ensure_proteome(code: str, force: bool = False) -> Tuple[str, str]:
    """Download and prepare raw + longest-per-gene FASTA; return paths."""
    meta = SPECIES[code]
    category = meta["category"]
    base_dir = os.path.join(DATASET_ROOT, "genomes", category, code)
    os.makedirs(base_dir, exist_ok=True)
    raw_gz = os.path.join(base_dir, f"{code}_proteins.fa.gz")
    raw_fa = os.path.join(base_dir, f"{code}_proteins.fa")
    longest = os.path.join(base_dir, f"{code}_proteins_longest.fa")

    download_file(meta["download_url"], raw_gz, force=force)
    gunzip_file(raw_gz, raw_fa, force=force)
    longest_per_gene(raw_fa, longest, force=force)
    return raw_fa, longest


def make_blast_db(fasta: str, db_prefix: str, force: bool = False) -> str:
    """Create a protein BLAST database if missing."""
    pin = f"{db_prefix}.pin"
    if os.path.exists(pin) and not force:
        return db_prefix
    os.makedirs(os.path.dirname(db_prefix), exist_ok=True)
    print(f"Building BLAST database {db_prefix}", file=sys.stderr)
    subprocess.run(
        [
            "makeblastdb",
            "-in",
            fasta,
            "-dbtype",
            "prot",
            "-out",
            db_prefix,
            "-parse_seqids",
        ],
        check=True,
    )
    return db_prefix


def run_blastp(query_fa: str, db_prefix: str, out_path: str, threads: int, force: bool = False) -> str:
    """Run BLASTp and write raw tabular output."""
    if os.path.exists(out_path) and not force:
        return out_path
    print(f"Running BLASTp {query_fa} vs {db_prefix}", file=sys.stderr)
    cmd = [
        "blastp",
        "-query",
        query_fa,
        "-db",
        db_prefix,
        "-out",
        out_path,
        "-outfmt",
        "6 qseqid sseqid bitscore evalue pident length qcovs scovs",
        "-max_target_seqs",
        "20",
        "-max_hsps",
        "1",
        "-evalue",
        "1e-5",
        "-num_threads",
        str(threads),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def best_hits(raw_blast_path: str, out_tsv: str, force: bool = False) -> str:
    """Reduce BLAST output to the best hit per query gene by bitscore then evalue."""
    if os.path.exists(out_tsv) and not force:
        return out_tsv
    print(f"Selecting best hits per query from {raw_blast_path}", file=sys.stderr)
    best: Dict[str, Tuple[str, float, float, str]] = {}  # q -> (s, bitscore, evalue, rest)
    with open(raw_blast_path) as f:
        for line in f:
            if not line.strip():
                continue
            q, s, bitscore, evalue, pident, length, qcovs, scovs = line.rstrip("\n").split("\t")
            b = float(bitscore)
            e = float(evalue)
            prev = best.get(q)
            if prev is None or b > prev[1] or (b == prev[1] and e < prev[2]):
                best[q] = (s, b, e, "\t".join([pident, length, qcovs, scovs]))
    header = ["vertebrate_gene", "outgroup_gene", "bitscore", "evalue", "pident", "aln_len", "qcovs", "scovs"]
    with open(out_tsv, "w") as f_out:
        f_out.write("\t".join(header) + "\n")
        for q in sorted(best):
            s, b, e, rest = best[q]
            f_out.write("\t".join([q, s, f"{b}", f"{e}", rest]) + "\n")
    return out_tsv


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertebrate", default="hsa", choices=SPECIES.keys(), help="Vertebrate code (default: hsa)")
    parser.add_argument("--outgroup", default="csa", choices=SPECIES.keys(), help="Outgroup code (default: csa)")
    parser.add_argument("--threads", type=int, default=4, help="Number of BLAST threads")
    parser.add_argument("--force", action="store_true", help="Redownload/regenerate intermediate files")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if SPECIES[args.vertebrate]["category"] != "vertebrata":
        parser.error(f"{args.vertebrate} is not marked as a vertebrate in SPECIES")
    if SPECIES[args.outgroup]["category"] != "outgroups":
        parser.error(f"{args.outgroup} is not marked as an outgroup in SPECIES")

    _, vertebrate_longest = ensure_proteome(args.vertebrate, force=args.force)
    _, outgroup_longest = ensure_proteome(args.outgroup, force=args.force)

    db_prefix = os.path.join(DATASET_ROOT, "genomes", SPECIES[args.outgroup]["category"], args.outgroup, f"{args.outgroup}_blastdb", "proteins")
    make_blast_db(outgroup_longest, db_prefix, force=args.force)

    results_dir = os.path.join(DATASET_ROOT, "results", "blastp")
    os.makedirs(results_dir, exist_ok=True)

    raw_blast = os.path.join(results_dir, f"{args.vertebrate}_vs_{args.outgroup}_blastp_raw.tsv")
    best_tsv = os.path.join(results_dir, f"{args.vertebrate}_vs_{args.outgroup}_blastp_best.tsv")

    run_blastp(vertebrate_longest, db_prefix, raw_blast, threads=args.threads, force=args.force)
    best_hits(raw_blast, best_tsv, force=args.force)

    print(f"Best-hit table written to {best_tsv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

