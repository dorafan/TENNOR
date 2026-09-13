#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""""

import argparse, os, json, random, re
from collections import Counter
from heapq import heappush, heappop

WIKI10_NTRAIN = 14146
WIKI10_L      = 30938



def parse_header_counts(line):
    """"""
    nums = re.findall(r"-?\d+", line)
    if len(nums) >= 3:
        try:
            a, b, c = int(nums[0]), int(nums[1]), int(nums[2])
            return a, b, c
        except Exception:
            return None
    return None

def split_labels_features(raw_line):
    """"""
    s = raw_line.rstrip("\n\r")
    if not s:
        return [], ""

    parts = s.split(None, 1)
    labels_part = parts[0]
    feats_part = parts[1] if len(parts) > 1 else ""

    labels = [int(t) for t in labels_part.split(",") if t != ""]
    return labels, feats_part

def read_bow(path):
    """"""
    docs = []
    feat_max = -1
    with open(path, "r", encoding="utf-8", newline="") as f:
        header = f.readline()
        header_counts = parse_header_counts(header) if header else None
        for line in f:
            if not line.strip():
                continue
            labels, feats = split_labels_features(line)
            docs.append((labels, feats))

            for tok in feats.split():
                if ":" in tok:
                    try:
                        idx = int(tok.split(":", 1)[0])
                        if idx > feat_max:
                            feat_max = idx
                    except Exception:
                        pass
    feat_dim = (feat_max + 1) if feat_max >= 0 else None
    return header, header_counts, docs, feat_dim




































































































def pick_docs_all_coverfill(train_docs, L_target, N_target, label_freq, warm_ratio=0.0,
                            head_bias=0.5, size_bias=0.5):
    """"""
    import math
    S = set()
    if warm_ratio > 0.0:
        k = max(1, int(L_target * warm_ratio))
        top = [lab for lab, _ in label_freq.most_common(k)]
        S.update(top)


    doc_labs = [(idx, set(labs)) for idx, (labs, _feats) in enumerate(train_docs) if labs]

    w = {l: math.log1p(label_freq[l]) for l in label_freq}


    from heapq import heappush, heappop
    heap = []
    for idx, s in doc_labs:
        new = s - S
        if not new:
            continue

        freq_score = sum(w.get(l, 0.0) for l in new)
        score = head_bias * freq_score + size_bias * len(new)
        heappush(heap, (-score, -len(new), idx))

    chosen, chosen_set = [], set()
    while heap and len(chosen) < N_target and len(S) < L_target:
        _negscore, _negk, idx = heappop(heap)
        if idx in chosen_set:
            continue
        s = doc_labs[idx][1]
        new = s - S
        if not new:
            continue
        if len(S) + len(new) <= L_target:
            S |= new
            chosen.append(idx)
            chosen_set.add(idx)


    if len(chosen) < N_target:
        for idx, s in doc_labs:
            if idx in chosen_set:
                continue
            if s.issubset(S):
                chosen.append(idx)
                chosen_set.add(idx)
                if len(chosen) >= N_target:
                    break

    chosen.sort()
    return chosen, S





def write_train_small(train_docs, chosen_idx_set, kept_labels_sorted, out_path, src_header, src_header_counts):
    """"""
    label_keep_map = {old: new for new, old in enumerate(kept_labels_sorted)}
    n_docs = 0
    used_labels_new = set()



    new_header = None
    if src_header_counts:

        orig_docs, orig_feats, _orig_labels = src_header_counts
        feats_dim = orig_feats

        new_header = f"{0} {feats_dim} {len(kept_labels_sorted)}\n"

    with open(out_path, "w", encoding="utf-8", newline="\n") as fo:
        if new_header:
            fo.write(new_header)
        else:

            fo.write(src_header if src_header else "\n")

        for idx, (labs, feats) in enumerate(train_docs):
            if idx not in chosen_idx_set:
                continue

            mapped = sorted({label_keep_map[l] for l in labs})
            used_labels_new.update(mapped)
            line = ",".join(str(x) for x in mapped)
            if feats:
                fo.write(f"{line} {feats.strip()}\n")
            else:
                fo.write(f"{line}\n")
            n_docs += 1


    if new_header:

        with open(out_path, "r+", encoding="utf-8", newline="\n") as fo:
            rest = fo.read()
            fo.seek(0)

            fo.write(f"{n_docs} {src_header_counts[1]} {len(kept_labels_sorted)}\n")
            fo.write(rest.split("\n", 1)[1] if "\n" in rest else "")

    return n_docs, len(used_labels_new), label_keep_map

def write_test_small(test_docs, kept_labels_sorted, out_path, src_header, src_header_counts):
    """"""
    keep = set(kept_labels_sorted)
    label_keep_map = {old: new for new, old in enumerate(kept_labels_sorted)}
    n_docs = 0

    new_header = None
    if src_header_counts:
        feats_dim = src_header_counts[1]
        new_header = f"{0} {feats_dim} {len(kept_labels_sorted)}\n"

    with open(out_path, "w", encoding="utf-8", newline="\n") as fo:
        if new_header:
            fo.write(new_header)
        else:
            fo.write(src_header if src_header else "\n")

        for labs, feats in test_docs:
            inter = sorted({label_keep_map[l] for l in labs if l in keep})
            if not inter:
                continue
            line = ",".join(str(x) for x in inter)
            if feats:
                fo.write(f"{line} {feats.strip()}\n")
            else:
                fo.write(f"{line}\n")
            n_docs += 1

    if new_header:
        with open(out_path, "r+", encoding="utf-8", newline="\n") as fo:
            rest = fo.read()
            fo.seek(0)
            fo.write(f"{n_docs} {src_header_counts[1]} {len(kept_labels_sorted)}\n")
            fo.write(rest.split("\n", 1)[1] if "\n" in rest else "")

    return n_docs



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train.txt (labels + features)")
    ap.add_argument("--test",  required=True, help="Path to test.txt  (labels + features)")
    ap.add_argument("--out",   required=True, help="Output directory")
    ap.add_argument("--multiplier", type=int, choices=[2,3], required=True,
                    help="2 or 3 times Wiki10-31K targets")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--warm_ratio", type=float, default=0.25,
                    help="Warm-start fraction of L_target by top-freq labels (0~0.5)")
    ap.add_argument("--test_ratio", type=float, default=1.0/3.0,
                help="Downsample test.small.txt to ≈ ratio × train docs (set <=0 to disable).")

    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)


    N_target = args.multiplier * WIKI10_NTRAIN
    L_target = args.multiplier * WIKI10_L
    print(f"[Target] Ntrain={N_target:,}, L={L_target:,}")


    print("[Read] train.txt ...")
    tr_header, tr_counts, train_docs, tr_featdim = read_bow(args.train)
    print("[Read] test.txt ...")
    te_header, te_counts, test_docs, te_featdim = read_bow(args.test)


    label_freq = Counter()
    for labs, _ in train_docs:
        for l in labs:
            label_freq[l] += 1
    uniq_labels = len(label_freq)
    print(f"[Stats] Train docs={len(train_docs):,}, uniq_labels={uniq_labels:,}, feat_dim≈{tr_featdim or (tr_counts[1] if tr_counts else 'NA')}")






    chosen_idx, kept_labels = pick_docs_all_coverfill(
        train_docs, L_target=L_target, N_target=N_target,
        label_freq=label_freq, warm_ratio=0.0
    )
    if len(chosen_idx) < N_target:
        print(f"[Warn] Could only pick {len(chosen_idx):,}/{N_target:,} docs under ALL + L_budget={L_target:,}. "
              f"Try --multiplier 3 or increase --warm_ratio.")

    kept_sorted = sorted(kept_labels, key=lambda l: (-label_freq[l], l))


    train_out = os.path.join(args.out, "train.small.txt")
    n_trn, l_trn, label_keep_map = write_train_small(
        train_docs, chosen_idx_set=set(chosen_idx),
        kept_labels_sorted=kept_sorted,
        out_path=train_out,
        src_header=tr_header, src_header_counts=tr_counts
    )
    print(f"[Write] train.small.txt -> docs={n_trn:,}, labels={l_trn:,}")


    with open(os.path.join(args.out, "kept_train_idx.txt"), "w", encoding="utf-8", newline="\n") as fo:
        for i in chosen_idx:
            fo.write(f"{i}\n")


    test_out = os.path.join(args.out, "test.small.txt")
    n_tst = write_test_small(
        test_docs, kept_labels_sorted=kept_sorted,
        out_path=test_out,
        src_header=te_header, src_header_counts=te_counts
    )
    print(f"[Write] test.small.txt  -> docs={n_tst:,}")







    if args.test_ratio > 0:

        train_out = os.path.join(args.out, "train.small.txt")
        with open(train_out, "r", encoding="utf-8", newline="\n") as f:
            tr_header_line = f.readline().strip()
        tr_counts = parse_header_counts(tr_header_line)
        train_docs_cnt = tr_counts[0] if tr_counts else None
        if train_docs_cnt is None:

            train_docs_cnt = n_trn if 'n_trn' in locals() else n_tst * 3

        target_test = int(round(train_docs_cnt * args.test_ratio))
        if n_tst > target_test:

            with open(test_out, "r", encoding="utf-8", newline="\n") as f:
                old_header = f.readline()
                lines = f.readlines()


            rnd = random.Random(args.seed)
            rnd.shuffle(lines)
            keep = lines[:target_test]


            old_counts = parse_header_counts(old_header)
            feat_dim = old_counts[1] if old_counts else (te_counts[1] if te_counts else 0)


            with open(test_out, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"{len(keep)} {feat_dim} {len(kept_sorted)}\n")
                f.writelines(keep)

            print(f"[Downsample] test.small.txt -> docs={len(keep):,} (from {n_tst:,}, target≈{target_test:,})")







    with open(os.path.join(args.out, "label_map.json"), "w", encoding="utf-8") as fo:
        json.dump({str(k): int(v) for k, v in label_keep_map.items()}, fo, ensure_ascii=False, indent=2)

    stats = {
        "multiplier": args.multiplier,
        "seed": args.seed,
        "target": {"Ntrain": N_target, "L": L_target},
        "actual": {"Ntrain": n_trn, "L": l_trn, "Ntest": n_tst},
        "source_counts": {
            "train_docs_src": len(train_docs),
            "labels_src": uniq_labels,
            "featdim_train_header": (tr_counts[1] if tr_counts else None),
            "featdim_train_guess": tr_featdim,
            "featdim_test_header": (te_counts[1] if te_counts else None),
            "featdim_test_guess": te_featdim
        },
        "notes": "Training uses ALL+budget (no partial deletion). Test uses ANY to same label space."
    }
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as fo:
        json.dump(stats, fo, ensure_ascii=False, indent=2)

    print("[Done] Output dir:", os.path.abspath(args.out))

if __name__ == "__main__":
    main()
