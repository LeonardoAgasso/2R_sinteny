#!/usr/bin/env python3

import sys
import os
import math
from collections import defaultdict
sys.path.append("/home/lagasso/projects/sinteny/local/utils/")
from poibin import PoiBin
import numpy as np


"""
Compute synteny-based q-values for vertebrate genes using outgroup genomes.
"""



debug = False
verbose = True
WINDOW_SIZES = [100, 200, 300, 400, 500]
MIN_K = 1
OUTGROUPS = ["cin", "csa"]

v_err_file_path = "./err/v_excluded_genes.txt"
o_err_file_path = "./err/o_excluded_genes.txt"



# FLOAT FORMATTING
def fmt(x):
	return format(x, ".17g")

def warn(msg):
	print(msg, file=sys.stderr)


# GENOMES
def load_bed_genome(bed_path):
	"""
	Load a BED-like file and return:
		- chr_to_genes: {chrom: [gene1, gene2, ...]} ordered by genomic position
		- gene_to_pos: {gene: (chrom, index_in_that_chrom_list)}
	"""
	chr_to_genes = defaultdict(list)
	tmp_chr_to_list = defaultdict(list)
	with open(bed_path) as bed_file:
		for line in bed_file:
			if not line.strip() or line.startswith("#"):	# skip empty lines and comments
				continue
			fields = line.strip().split("\t")	# assumed tsv file
			if len(fields) < 4:
				continue	 # not a BED
			chrom = fields[0]
			start = int(fields[1])
			gene_id = fields[3]
			tmp_chr_to_list[chrom].append((start, gene_id))

	gene_to_pos = {}

	for chrom, lst in tmp_chr_to_list.items():
		# sort by genomic start
		lst.sort(key=lambda x: x[0])
		genes_ordered = [g for _, g in lst]
		chr_to_genes[chrom] = genes_ordered
		for idx, g in enumerate(genes_ordered):
			gene_to_pos[g] = (chrom, idx)

	return chr_to_genes, gene_to_pos		

def load_outgroup_genomes(folder_path):
	"""
	Visit all the folders inside folder_path whose name is in OUTGROUPS.
	Load their BED files (assumed to be named OUTGROUP_biomart_proteincoding.bed), in a dictionary:
		outgroup_name -> (chr_to_genes, gene_to_pos)
	"""
	if verbose:
		warn(f"Loading outgroup genomes from {folder_path}...")
	
	outgroup_genomes = {}
	for og in OUTGROUPS:
		bed_path = f"{folder_path}/{og}/{og}_biomart_all.bed"
		chr_to_genes, gene_to_pos = load_bed_genome(bed_path)
		outgroup_genomes[og] = (chr_to_genes, gene_to_pos)
	return outgroup_genomes


# HOMOLOGY
def load_ortholog_pairs(orthology_tsv, outgroup_col_idx_zero_based):
	"""
	Load orthology TSV.
	Assumes:
		- column 0: vertebrate gene ID
		- column outgroup_col_idx_zero_based: outgroup gene ID

	Returns:
		- ortholog_pairs: list of (V, O)
		- v_to_o: V -> set of O
		- o_to_v: O -> set of V
	"""
	if verbose:
		warn(f"Loading ortholog pairs from {orthology_tsv}...")
	
	ortholog_pairs = []
	v_to_o = defaultdict(set)
	o_to_v = defaultdict(set)

	with open(orthology_tsv) as ortho_file:
		header = ortho_file.readline() # consume header
		for line in ortho_file:
			if not line.strip():
				continue
			fields = line.strip().split("\t")
			if len(fields) <= outgroup_col_idx_zero_based:
				continue
			vertebrate_gene = fields[0]
			outgroup_gene = fields[outgroup_col_idx_zero_based]
			if not vertebrate_gene or not outgroup_gene:
				continue
			ortholog_pairs.append((vertebrate_gene, outgroup_gene))
			v_to_o[vertebrate_gene].add(outgroup_gene)
			o_to_v[outgroup_gene].add(vertebrate_gene)
	ortholog_pairs = list(set(ortholog_pairs))	# avoid duplicate pairs

	return ortholog_pairs, v_to_o, o_to_v

def load_all_orthology_pairs(orthology_root_dir, vert):
	"""
	For each outgroup in OUTGROUPS, load its orthology TSV.

	Assumes file structure:
		<orthology_root_dir>/<og>/<og>_orthology.tsv

	and columns:
		col 0: vertebrate gene
		col 1: outgroup gene

	Returns:
		orthology_data: dict[og] -> (ortholog_pairs, v_to_o, o_to_v)
	"""
	if verbose:
		warn(f"Loading orthology for outgroups from {orthology_root_dir}...")

	orthology_data = {}
	for og in OUTGROUPS:
		ortho_path = f"{orthology_root_dir}/{vert}_{og}_biomart_orthologs.tsv"
		# vertebrate in col 0, outgroup in col 1
		ortholog_pairs, v_to_o, o_to_v = load_ortholog_pairs(ortho_path, outgroup_col_idx_zero_based=1)
		orthology_data[og] = (ortholog_pairs, v_to_o, o_to_v)
	return orthology_data

def load_paralog_pairs(paralogy_tsv, self_col_idx_zero_based):
	"""
	Load paralogy TSV.
	Assumes:
		- column self_col_idx_zero_based: gene ID

	Returns:
		- paralog_pairs: list of (G1, G2)
		- g_to_g: G -> set of G (paralogs)
	"""
	if verbose:
		warn(f"Loading paralog pairs from {paralogy_tsv}...")
	
	paralog_pairs = []
	g_to_g = defaultdict(set)

	with open(paralogy_tsv) as para_file:
		header = para_file.readline() # consume header
		for line in para_file:
			if not line.strip():
				continue
			fields = line.strip().split("\t")
			if len(fields) <= self_col_idx_zero_based:
				continue
			gene1 = fields[self_col_idx_zero_based]
			gene2 = fields[self_col_idx_zero_based + 1]  # assume next column
			if not gene1 or not gene2:
				continue
			paralog_pairs.append((gene1, gene2))
			g_to_g[gene1].add(gene2)
			g_to_g[gene2].add(gene1)
	paralog_pairs = list(set(paralog_pairs))	# avoid duplicate pairs

	return paralog_pairs, g_to_g


# P-VALUE COMPUTATION
def get_window_indices(center_idx, n_genes, W):
	"""
	v1 convention: total window length is W+1 (includes the anchor).
	Fixed length at chromosome boundaries by shifting (asymmetric near ends).
	Returns half-open [start, end) indices.
	"""
	L = W + 1  # total length including anchor

	if n_genes <= L:
		return 0, n_genes

	half_left = W // 2
	start = center_idx - half_left
	end = start + L

	if start < 0:
		start = 0
		end = L
	if end > n_genes:
		end = n_genes
		start = n_genes - L

	return start, end

def total_windows_in_genome(chr_to_genes, W):
	"""Total number of full windows of length (W+1) across all chromosomes."""
	L = W + 1
	total = 0
	for genes in chr_to_genes.values():
		n = len(genes)
		if n >= L:
			total += (n - L + 1)
	return total

def compute_Pi_for_outgene(out_gene, o_to_v, chr_to_genes_V, gene_to_pos_V, total_windows, W):
	"""
	Compute P_i for a single outgroup gene: i.e. the probability that at least
	one of its vertebrate orthologs is found in a random window of size W on the 
	vertebrate genome.
	"""
	orthologs = o_to_v.get(out_gene, set())
	if not orthologs:
		return 0.0

	L = W + 1  # total length including anchor
	windows_without = 0

	# positions of orthologs per chromosome (indices, not genomic coords)
	positions_by_chr = defaultdict(list)
	for v_gene in orthologs:
		if v_gene not in gene_to_pos_V:
			continue
		chr_v, idx_v = gene_to_pos_V[v_gene]
		positions_by_chr[chr_v].append(idx_v)

	for chr_v, genes_on_chr in chr_to_genes_V.items():
		n = len(genes_on_chr)
		if n < L:
			continue

		pos_list = sorted(positions_by_chr.get(chr_v, []))
		if not pos_list:
			# no orthologs on this chromosome
			windows_without += (n - L + 1)
			continue

		# gaps between ortholog positions
		prev = -1
		for p in pos_list:
			seg_len = p - prev - 1
			if seg_len >= L:
				windows_without += (seg_len - L + 1)
			prev = p

		# tail from last ortholog to end
		seg_len = (n - 1) - prev
		if seg_len >= L:
			windows_without += (seg_len - L + 1)

	if total_windows == 0:
		return 0.0

	Pi = 1.0 - (windows_without/total_windows)
	
	return min(1.0, max(0.0, Pi))

def compute_all_Pi(o_to_v, chr_to_genes_V, gene_to_pos_V, W):
	"""
	Compute P_i for all outgroup genes. Return a dict: outgroup_gene -> Pi.
	Outgroup genes are identified 
	"""
	total_windows = total_windows_in_genome(chr_to_genes_V, W)
	Pi_dict = {}
	for out_gene in o_to_v.keys():
		Pi_dict[out_gene] = compute_Pi_for_outgene(out_gene, o_to_v, chr_to_genes_V, gene_to_pos_V, total_windows, W)
	return Pi_dict

def geometric_mean_P(Pi_dict, outgroup_genes):
	"""
	Compute the geometric mean of Pi values, but *only over Pi > 0*.
	That is:
		P_geom = exp( (1/k) * sum_{Pi > 0} log(Pi) )
	where k is the number of non-zero Pi values.

	If all Pi are zero or missing, the function returns 0.0.
	"""
	log_sum = 0.0
	k = 0  # count of Pi > 0

	for og in outgroup_genes:
		Pi = Pi_dict.get(og, 0.0)
		if Pi > 0.0:
			log_sum += math.log(Pi)
			k += 1

	if k == 0:
		# No positive Pi values – geometric mean not defined
		return None

	return math.exp(log_sum / k)

def binomial_tail(N0, k, P):
	"""
	Right-tail P[X >= k] for X ~ Binomial(N0, P). 
	"""
	if N0 == 0:
		return None
	if P <= 0.0:
		return 0.0 if k > 0 else 1.0
	if P >= 1.0:
		return 1.0

	p_tail = 0.0
	for j in range(k, N0 + 1):
		p_tail += math.comb(N0, j) * (P ** j) * ((1 - P) ** (N0 - j))

	return min(1.0, max(0.0, p_tail))

def poibin_tail(Pi_dict, genes, k: int):
    """Exact right-tail for Poisson-binomial sum of independent Bernoulli(p_i)."""
    probs = [Pi_dict.get(g, 0.0) for g in genes]
    
    if not probs:
        return None
    
    try:
        pb = PoiBin(probs)
        result = pb.pval(k)
        
        if np.isnan(result):
            return None
        
        return result
    
	except Exception as e:
        warn(f"PoiBin error: {e}")
        return None
	

def has_full_window(center_idx, n_genes, W):
	"""
	Return True if a full window of size W can be centered on center_idx
	without shrinking at chromosome edges.
	"""
	half = W // 2
	start = center_idx - half
	end = start + W + 1 
	return (start >= 0) and (end <= n_genes)

def pvalue_for_anchor(anchor_A, anchor_B, W, chr_to_genes_A, chr_to_genes_B, gene_to_pos_A, gene_to_pos_B, b_to_a, Pi_dict, p_method="poibin", err_tag=None, err_file_path=None):
	"""
	Compute N0, k, P_geom, P_value for one anchor (V,O) and window size W.
	"""	

	if anchor_A not in gene_to_pos_A:
		return None
	if anchor_B not in gene_to_pos_B:
		return None

	chr_a, idx_a = gene_to_pos_A[anchor_A]
	chr_b, idx_b = gene_to_pos_B[anchor_B]

	genes_on_chr_a = chr_to_genes_A[chr_a]
	genes_on_chr_b = chr_to_genes_B[chr_b]

	# Strict exclusion (your original design): require a symmetric full window
	if not has_full_window(idx_a, len(genes_on_chr_a), W):
		if err_tag and err_file_path:
			warn(f"WARNING ({err_tag}): {anchor_A} on chr {chr_a} (n={len(genes_on_chr_a)}) cannot fit window W={W}; excluded as anchor")
			with open(err_file_path, "a") as f:
				f.write(f"{anchor_A}\t{W}\t{chr_a}\t{len(genes_on_chr_a)}\n")
		return None

	if not has_full_window(idx_b, len(genes_on_chr_b), W):
		if err_tag and err_file_path:
			warn(f"WARNING ({err_tag}): {anchor_B} on chr {chr_b} (n={len(genes_on_chr_b)}) cannot fit window W={W}; excluded as anchor")
			with open(err_file_path, "a") as f:
				f.write(f"{anchor_B}\t{W}\t{chr_b}\t{len(genes_on_chr_b)}\n")
		return None

	# Windows
	start_a, end_a = get_window_indices(idx_a, len(genes_on_chr_a), W)
	start_b, end_b = get_window_indices(idx_b, len(genes_on_chr_b), W)

	window_a_genes = set(genes_on_chr_a[start_a:end_a])
	window_b_genes = set(genes_on_chr_b[start_b:end_b])

	# Exclude anchor outgroup gene from the outgroup window
	window_b_genes_no_anchor = window_b_genes - {anchor_B}

    # B_candidates: outgroup genes in window that have at least one vertebrate ortholog anywhere
	B_candidates = [og for og in window_b_genes_no_anchor if og in b_to_a]
	N0 = len(B_candidates)

	if N0 == 0:
		return None	# exclude anchors with no outgroup orthologs in window

	# k: outgroup genes that have ≥1 ortholog in the vertebrate window
	k = 0
	for og in B_candidates:
		orth_vs = b_to_a[og]  # set of vertebrate orthologs
		if orth_vs & window_a_genes:
			k += 1

	if k < MIN_K:
		return None # require at least 2 orthologous outgroup genes in the vertebrate window, excluding the anchor.

    # warn(f"Probabilities for B_candidates: {[Pi_dict.get(og, 0.0) for og in B_candidates]}")
	P_geom = geometric_mean_P(Pi_dict, B_candidates)

	if P_geom is None:
		return None  # no positive Pi values
    
	if p_method == "mean_field":
		p_val = binomial_tail(N0, k, P_geom)
	elif p_method == "poibin":
		p_val = poibin_tail(Pi_dict, B_candidates, k)  # NaN check
	else:
		raise ValueError(f"Unknown p-value computation method: {p_method}")
    
	return {"N0": N0, "k": k, "P_geom": P_geom, "p_value": p_val}


# Q-VALUE
def qval_vertebrate_genes_og(v_o_qscores, method="worse_p"):
	"""
	Given v_o_qscores: dict[(vertebrate_gene, outgroup_gene) -> q_score],
	find vertebrate genes sharing the same outgroup gene.

	For each outgroup gene O, consider all vertebrate genes V that form
	anchors (V, O). For each pair (V, V') with the same O, combine their
	q-scores (i.e. p-values) into a V-V' q-value:

		method == "worse_p":      q = max(q1, q2)
		method == "geometric_mean": q = sqrt(q1 * q2)
		method == "harmonic_mean":  q = 2*q1*q2 / (q1 + q2)

	Returns:
		vv_pvals: dict[(v1, v2) -> q_value]
	"""
	outgroup_to_vertebrates = defaultdict(list)
	for (v_gene, o_gene), q in v_o_qscores.items():
		outgroup_to_vertebrates[o_gene].append((v_gene, q))

	vv_pvals = {}
	for o_gene, v_list in outgroup_to_vertebrates.items():
		if len(v_list) < 2:
			continue  # need at least two vertebrates sharing the same outgroup

		for i in range(len(v_list)):
			for j in range(i + 1, len(v_list)):
				v1, q1 = v_list[i]
				v2, q2 = v_list[j]

				if method == "worse_p":
					q = max(q1, q2)
				elif method == "geometric_mean":
					q = math.sqrt(q1 * q2)
				elif method == "harmonic_mean":
					q = 0.0 if (q1 + q2) == 0 else (2.0 * q1 * q2 / (q1 + q2))
				else:
					raise ValueError(f"Unknown method for q-value computation: {method}")

				# canonicalize pair ordering to avoid duplicates (v1, v2) vs (v2, v1)
				if v1 > v2:
					v1, v2 = v2, v1

				# if multiple outgroups contribute the same V-V' pair,
				# you may want to combine them (e.g. multiply q-scores, like OHNOLOGS v1)
				if (v1, v2) in vv_pvals:
					# here I choose to keep the *minimum* q (most significant) across outgroups,
					# but you could also multiply or geometric-mean them.
					vv_pvals[(v1, v2)] = min(vv_pvals[(v1, v2)], q)
				else:
					vv_pvals[(v1, v2)] = q

	return vv_pvals


def qval_vertebrate_genes_self(v_qscores):
	"""
	v_qscores: dict[(v_anchor, v_partner) -> q_dir]
		Directional q-scores for self-comparison (anchor -> partner).

	For each unordered pair {v1, v2}, we may have:
		q12 = q(v1 -> v2)
		q21 = q(v2 -> v1)

	We combine them into a single self q-value q_self(v1, v2):

		method == "worse_p":        q = max(q12, q21)
		method == "geometric_mean": q = sqrt(q12 * q21)  (OHNOLOGS-like)
		method == "harmonic_mean":  q = 2*q12*q21 / (q12 + q21)

	If only one direction exists, we just use that single q.
	Returns:
		vv_qvals_self: dict[(v1, v2) -> q_self] with v1 < v2
	"""
	from collections import defaultdict

	pair_to_qs = defaultdict(list)  # key: (v1, v2) with v1 < v2 -> list of directional q's

	for (va, vb), q in v_qscores.items():
		if va == vb:
			continue  # ignore self-self
		v1, v2 = (va, vb) if va < vb else (vb, va)
		pair_to_qs[(v1, v2)].append(q)

	vv_qvals_self = {}
	for (v1, v2), qs in pair_to_qs.items():
		if not qs:
			continue
		log_sum = 0.0
		for x in qs:
			if x <= 0.0:
				x = 1e-300
			log_sum += math.log(x)

		vv_qvals_self[(v1, v2)] = math.exp(log_sum / len(qs))

	return vv_qvals_self



# MAIN
def main():
	
	if len(sys.argv) != 4:
		warn(f"Usage: {sys.argv[0]} <vertebrate_short_name> <pval_compute_method> <pval_choose_method>")
		warn(f"  where: <vertebrate_short_name> is one of: hsa, mmu, rno, gga, dre")
		warn(f"         <pval_compute_method> is one of: poibin, mean_field")
		warn(f"         <pval_choose_method> is one of: worse_p, geometric_mean, harmonic_mean")
		sys.exit(1)

	dataset_path = "/home/lagasso/projects/sinteny/dataset"
	
	vertebrate      = sys.argv[1]
	pval_compute_method = sys.argv[2]
	pval_choose_method = sys.argv[3]

	vertebrate_bed = f'{dataset_path}/genomes/vertebrata/{vertebrate}/{vertebrate}_biomart_all.bed'
	#vertebrate_bed = f'{dataset_path}/genomes/vertebrata/{vertebrate}/{vertebrate}_biomart_proteincoding.bed'
	outgroup_root_dir   = f'{dataset_path}/genomes/outgroups/'
	orthology_root_dir  = f'{dataset_path}/orthologs'
	paralogy_tsv        = f'{dataset_path}/paralogs/{vertebrate}_biomart_paralogs.tsv'
	
	self_col_1based     = 1
	self_col_idx        = self_col_1based - 1

	os.makedirs("./err", exist_ok=True)

	# reset error files
	if os.path.exists(v_err_file_path):
		os.remove(v_err_file_path)
	if os.path.exists(o_err_file_path):
		os.remove(o_err_file_path)

	excluded_genes_header = "gene_id\tW\tchr\tn_genes_on_chromosome\n"
	with open(v_err_file_path, "w") as f:
		f.write(excluded_genes_header)
	with open(o_err_file_path, "w") as f:
		f.write(excluded_genes_header)

	chr_to_genes_V, gene_to_pos_V = load_bed_genome(vertebrate_bed)
	outgroup_genomes = load_outgroup_genomes(outgroup_root_dir)
	orthology_data = load_all_orthology_pairs(orthology_root_dir, vertebrate)
	paralog_pairs, g_to_g = load_paralog_pairs(paralogy_tsv, self_col_idx)

	# Precompute Pi per window size for outgroup comparisons
	Pi_by_W_out = {og: {} for og in OUTGROUPS}
	for og in OUTGROUPS:
		_, _, o_to_v = orthology_data[og]  # note: load_all_orthology_pairs gives (ortholog_pairs, v_to_o, o_to_v)
		for W in WINDOW_SIZES:
			Pi_by_W_out[og][W] = compute_all_Pi(o_to_v, chr_to_genes_V, gene_to_pos_V, W)

    # Precompute Pi per window size for self comparisons (paralogs)
	Pi_by_W_self = {}
	for W in WINDOW_SIZES:
		Pi_by_W_self[W] = compute_all_Pi(g_to_g, chr_to_genes_V, gene_to_pos_V, W)

	# Global outgroup-combined q-values for V–V
	vv_qvals_og_combined = defaultdict(lambda: 1.0)  # start at 1.0 so we can multiply

	for og in OUTGROUPS:
		if verbose:
			warn(f"Processing outgroup {og}...")

		chr_to_genes_O, gene_to_pos_O = outgroup_genomes[og]
		ortholog_pairs, v_to_o, o_to_v = orthology_data[og]

		# For each anchor (V,O), store p-values across W
		v_o_pvals = defaultdict(list)

		if verbose:
			warn("V --> O p-value computations...")
		
		for vertebrate_gene, outgroup_gene in ortholog_pairs:
			if vertebrate_gene not in gene_to_pos_V:
				continue
			if outgroup_gene not in gene_to_pos_O:
				continue

			for W in WINDOW_SIZES:
				Pi_dict = Pi_by_W_out[og][W]
				res = pvalue_for_anchor(vertebrate_gene, outgroup_gene, W, chr_to_genes_V, chr_to_genes_O, gene_to_pos_V, gene_to_pos_O, o_to_v, Pi_dict, p_method=pval_compute_method, err_tag="V->O", err_file_path=v_err_file_path)
				if res is None:
					continue
				v_o_pvals[(vertebrate_gene, outgroup_gene)].append(res["p_value"])

		if verbose:
			warn("V --> O q-score computations...")
		
		v_og_qscores = {}
		for (v_gene, o_gene), pvals in v_o_pvals.items():
			positive_pvals = [p for p in pvals if p > 0.0]
			if not positive_pvals:
				q = 0.0
			else:
				log_sum = sum(math.log(p) for p in positive_pvals)
				q = math.exp(log_sum / len(positive_pvals))
			v_og_qscores[(v_gene, o_gene)] = q

		if verbose:
			warn("V --> V' q-value computations...")

		vv_qvals_og_this = qval_vertebrate_genes_og(v_og_qscores, method=pval_choose_method)

		if verbose:
			warn(f"Combining V–V' q-values from outgroup {og}...")
		
		for (v1, v2), q_og in vv_qvals_og_this.items():
			vv_qvals_og_combined[(v1, v2)] *= q_og

	# SELF COMPARISON
	if verbose:
		warn("Self-comparisons...")
	
	v_self_pvals = defaultdict(list)  # key: (anchor_gene, partner_gene)
	
	if verbose:
		warn("V --> V p-value computations (self)...")

	for g1, g2 in paralog_pairs:
		if g1 not in gene_to_pos_V or g2 not in gene_to_pos_V:
			continue

		for W in WINDOW_SIZES:
			Pi_dict_self = Pi_by_W_self[W]

			# g1 --> g2
			res12 = pvalue_for_anchor(g1, g2, W, chr_to_genes_V, chr_to_genes_V, gene_to_pos_V, gene_to_pos_V, g_to_g, Pi_dict_self, p_method=pval_compute_method, err_tag=None, err_file_path=None)
			if res12 is not None:
				v_self_pvals[(g1, g2)].append(res12["p_value"])

			# g2 --> g1
			res21 = pvalue_for_anchor(g2, g1, W, chr_to_genes_V, chr_to_genes_V, gene_to_pos_V, gene_to_pos_V, g_to_g, Pi_dict_self, p_method=pval_compute_method, err_tag=None, err_file_path=None)
			if res21 is not None:
				v_self_pvals[(g2, g1)].append(res21["p_value"])

	#
	v_self_qscores = {}
	for (v1, v2), pvals in v_self_pvals.items():
		positive_pvals = [p for p in pvals if p > 0.0]
		if not positive_pvals:
			q = 0.0
		else:
			log_sum = sum(math.log(p) for p in positive_pvals)
			q = math.exp(log_sum / len(positive_pvals))
		v_self_qscores[(v1, v2)] = q	

	vv_qvals_self = qval_vertebrate_genes_self(v_self_qscores)	# here we always use geometric mean

	out_f = sys.stdout
	header = ["v1", "v2", "q_outgroup", "q_self"]
	out_f.write("\t".join(header) + "\n")

	for (v1, v2), q_out in vv_qvals_og_combined.items():
		q_self = vv_qvals_self.get((v1, v2), 1.0)
		out_f.write("\t".join([v1, v2, fmt(q_out), fmt(q_self)]) + "\n")


if __name__ == "__main__":
	main()