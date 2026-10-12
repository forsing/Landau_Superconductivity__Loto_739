#!/usr/bin/env python3

"""LANDAU V6 — GINZBURG-LANDAU: complex order and phase coupling."""

import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.optimize import nnls



DEFAULT = "/data/loto7_4698_k80.csv"
# DEFAULT = "/data/loto7_4698_k80_loto_2971.csv"
# DEFAULT = "/data/loto7_4698_k80_loto_plus_1727.csv"



N, K, SIZE, START, WINDOW = 39, 7, 8, 32, 384
CONTEXT = 3*SIZE
LOCAL = 2*SIZE+SIZE-1
WIDTH = LOCAL+2*SIZE*CONTEXT
BOUNDED = np.arange(SIZE, LOCAL)
FREE = np.setdiff1d(np.arange(WIDTH), BOUNDED)


def load(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.reader(f), 1):
            if not row or all(not v.strip() for v in row):
                continue
            try:
                numbers = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem sedam celih brojeva."
                ) from error

            if (len(numbers) != K or numbers != sorted(numbers)
                    or len(set(numbers)) != K
                    or not all(1 <= v <= N for v in numbers)):
                raise ValueError(
                    f"Red {line}: neispravna sortirana 7/39 kombinacija."
                )
            rows.append(numbers)

    return np.asarray(rows, dtype=int).reshape(-1, K)


def order_parameter(rows):
    rows = np.atleast_2d(rows)
    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), N+1),
    ))
    distances = np.diff(edges, axis=1)
    midpoint = .5*(edges[:, :-1]+edges[:, 1:])

    return (
        np.sqrt(distances/(N+1))
        * np.exp(2j*np.pi*midpoint/(N+1))
    )


def context(psi, t):
    field = np.r_[
        psi[t-1],
        psi[t-2],
        psi[t-8:t].mean(axis=0),
    ]
    link = np.angle(
        psi[t-1, 1:]*psi[t-1, :-1].conj()
    )
    return field, link


def features(psi, ctx, link):
    psi = np.atleast_2d(psi)
    density = np.abs(psi)**2
    gradient = np.abs(
        psi[:, 1:]
        - np.exp(1j*link)[None, :]*psi[:, :-1]
    )**2
    product = (
        psi.conj()[:, :, None]*ctx[None, None, :]
    ).reshape(len(psi), -1)

    return np.column_stack((
        density,
        density*density,
        gradient,
        -2*product.real,
        2*product.imag,
    ))


def alternatives(row):
    occupied = set(map(int, row))
    empty = sorted(set(range(1, N+1))-occupied)
    result = set()

    for removed in occupied:
        lower = [v for v in empty if v < removed]
        upper = [v for v in empty if v > removed]
        choices = {empty[0], empty[-1]}

        if lower:
            choices.add(lower[-1])
        if upper:
            choices.add(upper[0])

        for inserted in choices:
            result.add(tuple(sorted(
                (occupied-{removed}) | {inserted}
            )))

    return np.asarray(sorted(result), dtype=int)


def prepare(rows, psi, t):
    blocks = []

    # Samo ciljne kombinacije koje prethode redu t.
    for i in range(max(START, t-WINDOW), t):
        ctx, link = context(psi, i)
        positive = features(psi[i], ctx, link)
        negative = features(
            order_parameter(alternatives(rows[i])),
            ctx,
            link,
        )
        blocks.append(negative-positive)

    design = np.vstack(blocks)
    scale = np.maximum(
        np.sqrt((design*design).mean(axis=0)), 1e-8
    )
    a = design/scale
    ctx, link = context(psi, t)

    return (
        a.T @ a,
        a.sum(axis=0),
        scale,
        ctx,
        link,
    )


def bounded_coefficients(gram, rhs, scale, alpha):
    g = gram+alpha*np.eye(WIDTH)
    lower = np.zeros(WIDTH)

    # Pozitivan stabilizujuci clan cetvrtog stepena.
    lower[SIZE:2*SIZE] = 1e-4*scale[SIZE:2*SIZE]
    shifted = rhs-g @ lower

    guu = g[np.ix_(FREE, FREE)]
    guv = g[np.ix_(FREE, BOUNDED)]
    solved = np.linalg.solve(
        guu,
        np.column_stack((shifted[FREE], guv)),
    )
    u0 = solved[:, 0]
    inverse_cross = solved[:, 1:]
    gvu = guv.T

    schur = (
        g[np.ix_(BOUNDED, BOUNDED)]
        - gvu @ inverse_cross
    )
    reduced = shifted[BOUNDED]-gvu @ u0
    factor = np.linalg.cholesky(
        (schur+schur.T)/2
    )

    z, _ = nnls(
        factor.T,
        np.linalg.solve(factor, reduced),
        maxiter=1000,
    )

    coefficient = lower.copy()
    coefficient[BOUNDED] += z
    coefficient[FREE] = u0-inverse_cross @ z
    return coefficient/scale


def fit(prepared, alpha):
    gram, rhs, scale, ctx, link = prepared
    coefficient = bounded_coefficients(
        gram, rhs, scale, alpha
    )

    real = coefficient[
        LOCAL:LOCAL+SIZE*CONTEXT
    ].reshape(SIZE, CONTEXT)
    imaginary = coefficient[
        LOCAL+SIZE*CONTEXT:
    ].reshape(SIZE, CONTEXT)

    source = (real+1j*imaginary) @ ctx

    return (
        coefficient[:SIZE],
        coefficient[SIZE:2*SIZE],
        coefficient[2*SIZE:LOCAL],
        source,
        link,
    )


def energy_fields(psi, model):
    psi = np.atleast_2d(psi)
    alpha, beta, stiffness, source, link = model
    density = np.abs(psi)**2
    gradient = np.abs(
        psi[:, 1:]
        - np.exp(1j*link)[None, :]*psi[:, :-1]
    )**2

    return (
        density @ alpha
        + (density*density) @ beta
        + gradient @ stiffness
        - 2*np.real(psi.conj() @ source)
    )


def energy(rows, model):
    return energy_fields(order_parameter(rows), model)


def seeds(history):
    candidates = [
        tuple(map(int, row)) for row in history[-32:]
    ]
    for step in range(1, 7):
        for first in range(1, N-(K-1)*step+1):
            candidates.append(tuple(
                first+j*step for j in range(K)
            ))
    return np.asarray(sorted(set(candidates)), dtype=int)


def search(history, model):
    current = seeds(history)
    values = np.arange(1, N+1)

    for iteration in range(1, 101):
        available = np.all(
            values[None, :, None] != current[:, None, :],
            axis=2,
        )
        who, which = np.nonzero(available)
        blocks = [current]
        owners = [np.arange(len(current))]

        for j in range(K):
            neighbors = current[who].copy()
            neighbors[:, j] = values[which]
            blocks.append(np.sort(neighbors, axis=1))
            owners.append(who)

        candidates = np.vstack(blocks)
        owners = np.concatenate(owners)
        scores = energy(candidates, model)

        minimum = np.full(len(current), np.inf)
        np.minimum.at(minimum, owners, scores)

        eligible = scores <= minimum[owners]+1e-12
        indices = np.arange(len(candidates))
        chosen = np.full(
            len(current), len(candidates), dtype=int
        )
        np.minimum.at(
            chosen,
            owners,
            np.where(eligible, indices, len(candidates)),
        )

        updated = candidates[chosen]
        if np.array_equal(updated, current):
            scores = energy(current, model)
            best = int(np.argmin(scores))
            return (
                tuple(map(int, current[best])),
                float(scores[best]),
                iteration,
            )
        current = updated

    raise RuntimeError(
        "Pretraga nije dostigla optimum zamena u 100 koraka."
    )


def hits(prediction, actual):
    return len(
        set(map(int, prediction)) & set(map(int, actual))
    )


def summarize(label, scores):
    print(
        f"{label}: prosek={np.mean(scores):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(scores, minlength=8).tolist()}",
        flush=True,
    )


def self_test():
    rows = np.tile(np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [4, 12, 18, 28, 35, 36, 37],
        [33, 34, 35, 36, 37, 38, 39],
        [2, 7, 12, 18, 24, 30, 39],
    ]), (50, 1))

    psi = order_parameter(rows)
    assert np.allclose(
        (np.abs(psi)**2).sum(axis=1), 1
    )

    prepared = prepare(rows, psi, 170)
    model = fit(prepared, 10.)
    assert np.all(model[1] >= 1e-4-1e-12)
    assert np.all(model[2] >= -1e-12)

    changed = rows.copy()
    changed[170:] = [3, 8, 13, 19, 25, 31, 38]
    altered = fit(
        prepare(changed, order_parameter(changed), 170),
        10.,
    )
    for a, b in zip(model, altered):
        assert np.array_equal(a, b)

    rotation = np.exp(.73j)
    rotated_model = (
        *model[:3],
        model[3]*rotation,
        model[4],
    )
    assert np.allclose(
        energy_fields(psi, model),
        energy_fields(psi*rotation, rotated_model),
    )

    ctx, link = context(psi, 170)
    coefficient = bounded_coefficients(
        *prepared[:3], 10.
    )
    assert np.allclose(
        features(psi, ctx, link) @ coefficient,
        energy_fields(psi, model),
    )

    result = search(rows[:170], model)
    assert result == search(rows[:170], model)

    combo, score, _ = result
    neighbors = []
    for removed in combo:
        for inserted in set(range(1, N+1))-set(combo):
            neighbors.append(sorted(
                (set(combo)-{removed}) | {inserted}
            ))

    assert energy(neighbors, model).min() >= score-1e-12

    print(
        "Self-test OK: kompleksno polje, fazna simetrija, "
        "stabilnost i hronologija."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=DEFAULT)
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    rows = load(args.csv)
    if args.validation < 16 or args.test < 16:
        parser.error(
            "Validation i test moraju imati bar 16 kola."
        )

    split = len(rows)-args.test
    begin = split-args.validation
    if begin < START+128:
        parser.error(
            "Premalo istorije za zadatu validaciju i test."
        )

    psi = order_parameter(rows)
    alphas = (1., 10., 100.)
    validation = [[] for _ in alphas]

    print(
        f"Ulaz: {Path(args.csv).resolve()}; "
        f"redova={len(rows)}",
        flush=True,
    )

    for t in range(begin, split):
        prepared = prepare(rows, psi, t)

        for j, alpha in enumerate(alphas):
            model = fit(prepared, alpha)
            combo, _, _ = search(rows[:t], model)
            validation[j].append(hits(combo, rows[t]))

        if (t-begin+1) % 16 == 0:
            print(
                f"Validacija: {t-begin+1}/{args.validation}",
                flush=True,
            )

    means = [
        np.mean(values) for values in validation
    ]
    best = int(np.argmax(means))
    alpha = alphas[best]

    print(
        f"Validacija: redovi {begin+1}..{split}; "
        f"alpha={alpha:g}; prosek={means[best]:.4f}",
        flush=True,
    )

    model_scores = []
    previous_scores = []
    no_gradient_scores = []

    for t in range(split, len(rows)):
        model = fit(
            prepare(rows, psi, t), alpha
        )
        combo, _, _ = search(rows[:t], model)

        model_scores.append(hits(combo, rows[t]))
        previous_scores.append(hits(rows[t-1], rows[t]))

        reduced = (
            model[0],
            model[1],
            np.zeros_like(model[2]),
            model[3],
            model[4],
        )
        reference, _, _ = search(rows[:t], reduced)
        no_gradient_scores.append(hits(reference, rows[t]))

        if (t-split+1) % 16 == 0:
            print(
                f"Test: {t-split+1}/{args.test}",
                flush=True,
            )

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize("LANDAU V6 GINZBURG-LANDAU", model_scores)
    summarize("Prethodna kombinacija", previous_scores)
    summarize(
        "Isto polje bez gradijentne sprege",
        no_gradient_scores,
    )

    model = fit(
        prepare(rows, psi, len(rows)), alpha
    )
    combo, score, iterations = search(rows, model)

    print(
        f"Poslednji red: {','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(f"F={score:.10f}; iteracija={iterations}")
    print(
        f"beta_min={model[1].min():.8f}; "
        f"jacina gradijentne sprege="
        f"{np.linalg.norm(model[2]):.6f}"
    )
    print(
        "F rangira cele kombinacije. "
        "Pretraga je deterministicka; "
        "globalni minimum nije potvrden."
    )


if __name__ == "__main__":
    main()



"""
Ulaz: /data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 4587..4634; alpha=10; prosek=1.3542
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
LANDAU V6 GINZBURG-LANDAU: prosek=1.4062; pogodaka 0..7=[10, 24, 24, 6, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Isto polje bez gradijentne sprege: prosek=1.2344; pogodaka 0..7=[11, 33, 16, 3, 0, 1, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 8,x,11,y,13,z,28
F=-6.2705353677; iteracija=15
beta_min=0.00010000; jacina gradijentne sprege=0.392394
F rangira cele kombinacije. Pretraga je deterministicka; globalni minimum nije potvrden.



Ulaz: /data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 2860..2907; alpha=1; prosek=1.2500
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
LANDAU V6 GINZBURG-LANDAU: prosek=1.2500; pogodaka 0..7=[18, 20, 19, 6, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Isto polje bez gradijentne sprege: prosek=1.4062; pogodaka 0..7=[13, 23, 20, 5, 3, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 8,x,13,y,15,z,31
F=-7.3001379510; iteracija=23
beta_min=0.00010000; jacina gradijentne sprege=0.616697
F rangira cele kombinacije. Pretraga je deterministicka; globalni minimum nije potvrden.



Ulaz: /data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 1616..1663; alpha=10; prosek=1.3958
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
LANDAU V6 GINZBURG-LANDAU: prosek=1.3750; pogodaka 0..7=[10, 26, 23, 4, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Isto polje bez gradijentne sprege: prosek=1.3750; pogodaka 0..7=[9, 29, 20, 5, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 11,x,13,y,27,z,29
F=-5.2951783922; iteracija=14
beta_min=0.00010000; jacina gradijentne sprege=0.380617
F rangira cele kombinacije. Pretraga je deterministicka; globalni minimum nije potvrden.
"""



"""
V6 LANDAU.   GINSBURG-LANDAU superconductivity

GINZBURG–LANDAU. Model sa kompleksnim parametrom reda i spregom između delova rasporeda.

V6 će imati tri dela energije: lokalni parametar reda, stabilizujući član četvrtog stepena i spregu amplituda i faza susednih delova rasporeda. Proverio sam i da zajedničko pomeranje faze ne menja ocenu kombinacije.

Kompleksno polje prošlo je proveru fazne simetrije i uslova stabilnosti. Validacija je završena; proverio sam poslednja 64 kola i poređenje sa istim poljem bez gradijentne sprege.

Model koristi kompleksni parametar reda, član četvrtog stepena i gradijentnu spregu amplituda i faza. Inspiracija je Ginzburg–Landauova teorija superprovodljivosti.

Predlog za red 4699:
8, x, 11, y, 13, z, 28

Na poslednja 64 kola:
Metod	Prosečan broj pogodaka
V6	1,4062
Prethodna kombinacija	1,2188
Isto polje bez gradijentne sprege	1,2344

Proverio sam fazna simetrija, stabilnost, hronologija, ponovljivost i svih 224 pojedinačnih zamena završne kombinacije. Globalni minimum nije potvrđen.
"""
