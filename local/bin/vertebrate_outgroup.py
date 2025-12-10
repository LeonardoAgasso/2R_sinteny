#!/usr/bin/env python3

import sys
import math
import time
from collections import defaultdict
sys.path.append("/home/lagasso/projects/sinteny/local/utils/")
from poibin import PoiBin

debug = False
# Hardcoded window sizes
WINDOW_SIZES = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]


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


def get_window_indices(center_idx, n_genes, W):
    """
    n_genes is the total number of genes on the chromosome.
	If a gene is too close to the end/beginning of a chromosome,
	the window size is reduced.
    """
    half = W // 2
    start = max(0, center_idx - half)
    end = start + W
    if end > n_genes:
        end = n_genes
        start = max(0, end - W)
    return start, end


def total_windows_in_genome(chr_to_genes, W):
    """Total number of windows of length W across all chromosomes."""
    total = 0
    for chrom, genes in chr_to_genes.items():
        n = len(genes)
        if n >= W:
            total += (n - W + 1)
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
		if n < W:
			continue

		pos_list = sorted(positions_by_chr.get(chr_v, []))
		if not pos_list:
			# no orthologs on this chromosome
			windows_without += (n - W + 1)
			continue

		# gaps between ortholog positions
		prev = -1
		for p in pos_list:
			seg_len = p - prev - 1
			if seg_len >= W:
				windows_without += (seg_len - W + 1)
			prev = p

		# tail from last ortholog to end
		seg_len = (n - 1) - prev
		if seg_len >= W:
			windows_without += (seg_len - W + 1)

	if total_windows == 0:
		return 0.0

	Pi = 1.0 - (windows_without/total_windows)
	if Pi < 0.0:
		Pi = 0.0
	if Pi > 1.0:
		Pi = 1.0
	return Pi


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
		return 0.0

	return math.exp(log_sum / k)


def binomial_tail(N0, k, P):
	"""
	Right-tail P[X >= k] for X ~ Binomial(N0, P). 
	"""
	if N0 == 0:
		return 1.0
	if P <= 0.0:
		return 0.0 if k > 0 else 1.0
	if P >= 1.0:
		return 1.0

	p_tail = 0.0
	for j in range(k, N0 + 1):
		p_tail += math.comb(N0, j) * (P ** j) * ((1 - P) ** (N0 - j))
	if p_tail < 0.0:
		p_tail = 0.0
	if p_tail > 1.0:
		p_tail = 1.0
	return p_tail


def poibin_tail(Pi_dict, outgroup_genes, k):
	"""
	Exact right-tail P[X >= k] for X = sum of independent Bernoulli(p_i),
	using the Poisson–binomial distribution.
	"""
	# Build the list of probabilities in a fixed order over your trials
	probs = [Pi_dict.get(og, 0.0) for og in outgroup_genes]  # ignore missing genes (Pi=0)
	pb = PoiBin(probs)
	return pb.pval(k)   # this is P(X >= k)


def pvalue_for_anchor(vertebrate_gene, outgroup_gene, W, chr_to_genes_V, chr_to_genes_O, gene_to_pos_V, gene_to_pos_O, o_to_v, Pi_dict):
	"""
	Compute N0, k, P_geom, P_value for one anchor (V,O) and window size W.
	"""
	if vertebrate_gene not in gene_to_pos_V:
		return None
	if outgroup_gene not in gene_to_pos_O:
		return None

	chr_v, idx_v = gene_to_pos_V[vertebrate_gene]
	chr_o, idx_o = gene_to_pos_O[outgroup_gene]

	genes_on_chr_v = chr_to_genes_V[chr_v]
	genes_on_chr_o = chr_to_genes_O[chr_o]

    # Windows
	start_v, end_v = get_window_indices(idx_v, len(genes_on_chr_v), W)
	start_o, end_o = get_window_indices(idx_o, len(genes_on_chr_o), W)

	window_v_genes = set(genes_on_chr_v[start_v:end_v])
	window_o_genes = set(genes_on_chr_o[start_o:end_o])

	# Exclude anchor outgroup gene from the outgroup window
	window_o_genes_no_anchor = window_o_genes - {outgroup_gene}

    # O_candidates: outgroup genes in window that have at least one vertebrate ortholog anywhere
	O_candidates = [og for og in window_o_genes_no_anchor if og in o_to_v]
	N0 = len(O_candidates)

	if N0 == 0:
		return {"N0": 0, "k": 0, "P_geom": 0.0, "p_value_mf": 1.0, "p_value": 1.0,}

	# k: outgroup genes that have ≥1 ortholog in the vertebrate window
	k = 0
	for og in O_candidates:
		orth_vs = o_to_v[og]  # set of vertebrate orthologs
		if orth_vs & window_v_genes:
			k += 1

    # print(f"Probabilities for O_candidates: {[Pi_dict.get(og, 0.0) for og in O_candidates]}", file=sys.stderr)
	P_geom = geometric_mean_P(Pi_dict, O_candidates)
    
	p_val_mf = binomial_tail(N0, k, P_geom)
	p_val_pb = poibin_tail(Pi_dict, O_candidates, k)  # NaN check
    
	# whenever the two pvalues differ significantly, print on a file (test_pi_distr) the two distributions of pi (one per line).
	with open("test_pi_distr.tsv", "a") as test_f:
			test_f.write(vertebrate_gene + "," + outgroup_gene + "\t" + str(k) + "\t" + str(N0) + "\t" + "\t".join([str(Pi_dict.get(og, 0.0)) for og in O_candidates]) + "\n")

	return {"N0": N0, "k": k, "P_geom": P_geom, "p_value_mf": p_val_mf, "p_value": p_val_pb,}


def debug_res(res, out_f, chr_o, outgroup_gene, idx_o, chr_v, vertebrate_gene, idx_v, W):
	if res is None:
		return
	
	out_f.write(
	"\t".join([chr_o, outgroup_gene, str(idx_o), chr_v, vertebrate_gene, str(idx_v), str(W), str(res["N0"]), str(res["k"]), f"{res['P_geom']:.6g}", f"{res['p_value_mf']:.6g}", f"{res['p_value']:.6g}"]) + "\n")


def qval_verterbrate_genes(v_o_qscores, method="worse_p"):
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
    from collections import defaultdict

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
                    q = max(q1, q2)  # Singh & Isambert q-score for outgroup comparison:contentReference[oaicite:4]{index=4}
                elif method == "geometric_mean":
                    q = math.sqrt(q1 * q2)
                elif method == "harmonic_mean":
                    if q1 + q2 == 0:
                        q = 0.0
                    else:
                        q = 2 * (q1 * q2) / (q1 + q2)
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
            

def main():
	if len(sys.argv) != 5:
		print(f"Usage: {sys.argv[0]} <vertebrate_bed> <outgroup_bed> <orthology_tsv> <outgroup_column_1based> > <output_tsv>", file=sys.stderr)
		sys.exit(1)

	vertebrate_bed = sys.argv[1]
	outgroup_bed = sys.argv[2]
	orthology_tsv = sys.argv[3]
	outgroup_col_1based = int(sys.argv[4])

	outgroup_col_idx = outgroup_col_1based - 1

    # Load genomes
	chr_to_genes_V, gene_to_pos_V = load_bed_genome(vertebrate_bed)
	chr_to_genes_O, gene_to_pos_O = load_bed_genome(outgroup_bed)

    # Load orthology
	ortholog_pairs, v_to_o, o_to_v = load_ortholog_pairs(orthology_tsv, outgroup_col_idx)

    # Precompute Pi per window size
	Pi_by_W = {}
    
	for W in WINDOW_SIZES:
		Pi_by_W[W] = compute_all_Pi(o_to_v, chr_to_genes_V, gene_to_pos_V, W)

    # Compute p-values for each anchor and window size
	out_f = sys.stdout
	#header = ["out_chr", "out_gene", "out_idx", "vert_chr", "vert_gene", "vert_idx", "W", "N0", "k", "P_geom", "P_value_(mean_field)", "P_value_(PoiBin)"]
	header = ["v1", "v2", "q_value"]
	out_f.write("\t".join(header) + "\n")

	v_o_pvals = defaultdict(list)

	for vertebrate_gene, outgroup_gene in ortholog_pairs:

		if vertebrate_gene not in gene_to_pos_V:
			continue
		if outgroup_gene not in gene_to_pos_O:
			continue

		chr_v, idx_v = gene_to_pos_V[vertebrate_gene]
		chr_o, idx_o = gene_to_pos_O[outgroup_gene]

		for W in WINDOW_SIZES:
			Pi_dict = Pi_by_W[W]
			res = pvalue_for_anchor(vertebrate_gene, outgroup_gene, W, chr_to_genes_V, chr_to_genes_O, gene_to_pos_V, gene_to_pos_O, o_to_v, Pi_dict)
			if res is None:
				continue
			v_o_pvals[(vertebrate_gene, outgroup_gene)].append(res["p_value"])
			if debug:
				debug_res(res, out_f, chr_o, outgroup_gene, idx_o, chr_v, vertebrate_gene, idx_v, W)

	v_o_qscores = {}

	for (v_gene, o_gene), pvals in v_o_pvals.items():
		positive_pvals = [p for p in pvals if p > 0.0]
		if not positive_pvals:
			combined_q = 0.0
		else:
			log_sum = sum(math.log(p) for p in positive_pvals)
			q = math.exp(log_sum / len(positive_pvals))
		v_o_qscores[(v_gene, o_gene)] = q
            
	vv_qvals = qval_verterbrate_genes(v_o_qscores, method="worse_p")
	for (v1, v2), qval in vv_qvals.items():
		out_f.write(f"{v1}\t{v2}\t{qval:.6g}\n")
			


if __name__ == "__main__":
	main()