#!/usr/bin/env python3

"""Landau V6: kompleksne deformacije, stabilna GL energija i cele kombinacije."""

import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.optimize import nnls

DEFAULTS = (
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv",
)

N, K, SIZE, WINDOW = 39, 7, 8, 64
LOCAL, WIDTH = 23, 55
BOUNDED = np.arange(SIZE, LOCAL)
FREE = np.setdiff1d(np.arange(WIDTH), BOUNDED)
ALPHAS = (.01, .1, 1.)


def load(path):
    rows = []

    with Path(path).open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue

            try:
                values = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem cele brojeve."
                ) from error

            if (
                len(values) != K
                or values != sorted(values)
                or len(set(values)) != K
                or not all(1 <= v <= N for v in values)
            ):
                raise ValueError(
                    f"Red {line}: neispravna sortirana "
                    "7/39 kombinacija."
                )

            rows.append(values)

    return np.asarray(rows, dtype=int).reshape(-1, K)


def order_parameter(rows):
    rows = np.atleast_2d(rows)

    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), 40)
    ))

    distances = np.diff(edges, axis=1)
    midpoint = .5*(edges[:, :-1]+edges[:, 1:])

    return (
        np.sqrt(distances/40.)
        * np.exp(2j*np.pi*midpoint/40.)
    )


def context(psi, t):
    ctx = np.column_stack((
        psi[t-1], psi[t-2]
    ))
    link = np.angle(
        psi[t-1, 1:]*psi[t-1, :-1].conj()
    )

    return ctx, link


def features(psi, reference, ctx, link):
    # GL funkcional deluje na deformaciju prema
    # stvarnom prethodnom redu.
    delta = np.atleast_2d(psi)-reference
    density = np.abs(delta)**2

    gradient = np.abs(
        delta[:, 1:]
        - np.exp(1j*link)[None, :]*delta[:, :-1]
    )**2

    product = (
        delta.conj()[:, :, None]*ctx[None, :, :]
    ).reshape(len(delta), 16)

    return np.column_stack((
        density,
        density*density,
        gradient,
        -2*product.real,
        2*product.imag
    ))


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (K*len(unused), 1))

    for j in range(K):
        result[j*len(unused):(j+1)*len(unused), j] = unused

    return np.sort(result, axis=1)


def prepare(history):
    if len(history) < WINDOW+2:
        raise ValueError(
            "Potrebno je najmanje 66 prethodnih redova."
        )

    history = history[-(WINDOW+2):]
    psi = order_parameter(history)

    gram = np.zeros((WIDTH, WIDTH))
    rhs = np.zeros(WIDTH)

    for t in range(2, len(history)):
        ctx, link = context(psi, t)

        actual = features(
            psi[t], psi[t-1], ctx, link
        )
        alternatives = features(
            order_parameter(replacements(history[t])),
            psi[t-1],
            ctx,
            link
        )

        contrasts = alternatives-actual

        gram += (
            contrasts.T @ contrasts/len(contrasts)
        )
        rhs += contrasts.sum(axis=0)/len(contrasts)

    gram /= WINDOW
    rhs /= WINDOW

    # Numericko skaliranje karakteristika.
    scale = np.maximum(
        np.sqrt(np.diag(gram)), 1e-8
    )
    scaled_gram = (
        gram/scale[:, None]/scale[None, :]
    )
    scaled_rhs = rhs/scale

    candidates = np.unique(
        np.vstack(
            [history[2:]]
            + [
                replacements(row)
                for row in history[2:]
            ]
        ),
        axis=0
    )

    ctx, link = context(psi, len(history))

    return (
        scaled_gram, scaled_rhs, scale,
        ctx, link, psi[-1], candidates
    )


def bounded_coefficients(prepared, alpha):
    gram, rhs, scale = prepared[:3]
    g = gram+alpha*np.eye(WIDTH)

    lower = np.zeros(WIDTH)

    # Pozitivna stabilizacija deformacije oko
    # stvarne kombinacije.
    lower[SIZE:2*SIZE] = (
        1e-4*scale[SIZE:2*SIZE]
    )
    shifted = rhs-g @ lower

    guu = g[np.ix_(FREE, FREE)]
    guv = g[np.ix_(FREE, BOUNDED)]

    solved = np.linalg.solve(
        guu,
        np.column_stack((
            shifted[FREE], guv
        ))
    )
    u0 = solved[:, 0]
    inverse_cross = solved[:, 1:]

    schur = (
        g[np.ix_(BOUNDED, BOUNDED)]
        - guv.T @ inverse_cross
    )
    reduced = (
        shifted[BOUNDED]-guv.T @ u0
    )

    factor = np.linalg.cholesky(
        (schur+schur.T)/2
    )

    z, _ = nnls(
        factor.T,
        np.linalg.solve(factor, reduced),
        maxiter=1000
    )

    coefficient = lower.copy()
    coefficient[BOUNDED] += z
    coefficient[FREE] = (
        u0-inverse_cross @ z
    )

    return coefficient/scale


def fit(prepared, alpha):
    coefficient = bounded_coefficients(
        prepared, alpha
    )
    ctx, link, reference = prepared[3:6]

    real = coefficient[
        LOCAL:LOCAL+16
    ].reshape(SIZE, 2)

    imaginary = coefficient[
        LOCAL+16:
    ].reshape(SIZE, 2)

    source = (
        (real+1j*imaginary)*ctx
    ).sum(axis=1)

    return (
        coefficient[:SIZE],
        coefficient[SIZE:2*SIZE],
        coefficient[2*SIZE:LOCAL],
        source,
        link,
        reference
    )


def energy_fields(psi, model):
    a, beta, stiffness, source, link, reference = model

    delta = np.atleast_2d(psi)-reference
    density = np.abs(delta)**2

    gradient = np.abs(
        delta[:, 1:]
        - np.exp(1j*link)[None, :]*delta[:, :-1]
    )**2

    return (
        density @ a
        + (density*density) @ beta
        + gradient @ stiffness
        - 2*np.real(delta.conj() @ source)
    )


def energy(rows, model):
    return energy_fields(
        order_parameter(rows), model
    )


def choose(prepared, model):
    candidates = prepared[6]
    values = energy(candidates, model)

    if not np.all(np.isfinite(values)):
        raise ArithmeticError(
            "Neispravna GL energija."
        )

    best = int(np.argmin(values))

    return (
        tuple(map(int, candidates[best])),
        float(values[best])
    )


def predict(prepared, alpha):
    return choose(prepared, fit(prepared, alpha))


def hits(prediction, actual):
    return len(
        set(map(int, prediction))
        & set(map(int, actual))
    )


def summarize(label, scores):
    print(
        f"{label}: prosek={np.mean(scores):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(scores, minlength=8).tolist()}",
        flush=True
    )


def run_file(path, validation_count=48, test_count=64):
    rows = load(path)
    split = len(rows)-test_count
    begin = split-validation_count

    if (
        validation_count < 16
        or test_count < 16
        or begin < WINDOW+2
    ):
        raise ValueError(
            "Premalo istorije ili manje od "
            "16 validacionih/test kola."
        )

    print(
        f"\nUlaz: {Path(path).resolve()}; "
        f"redova={len(rows)}",
        flush=True
    )

    validation = [[] for _ in ALPHAS]

    for t in range(begin, split):
        prepared = prepare(rows[:t])

        for j, alpha in enumerate(ALPHAS):
            combo, _ = predict(prepared, alpha)
            validation[j].append(
                hits(combo, rows[t])
            )

        if (t-begin+1) % 16 == 0:
            print(
                f"Validacija: "
                f"{t-begin+1}/{validation_count}",
                flush=True
            )

    means = [np.mean(s) for s in validation]
    alpha = ALPHAS[int(np.argmax(means))]

    print(
        f"Validacija: redovi {begin+1}..{split}; "
        f"alpha={alpha:g}; "
        f"prosek={max(means):.4f}",
        flush=True
    )

    scores, previous, no_gradient = [], [], []

    # Alpha ostaje fiksna tokom testa.
    # Predikcija koristi samo prethodne redove.
    for t in range(split, len(rows)):
        prepared = prepare(rows[:t])
        model = fit(prepared, alpha)
        combo, _ = choose(prepared, model)

        scores.append(hits(combo, rows[t]))
        previous.append(
            hits(rows[t-1], rows[t])
        )

        reduced = (
            model[0],
            model[1],
            np.zeros(7),
            *model[3:]
        )
        reference, _ = choose(prepared, reduced)

        no_gradient.append(
            hits(reference, rows[t])
        )

        if (t-split+1) % 16 == 0:
            print(
                f"Test: {t-split+1}/{test_count}",
                flush=True
            )

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize("LANDAU V6 GINZBURG-LANDAU", scores)
    summarize("Prethodna kombinacija", previous)
    summarize(
        "Isto polje bez gradijentne sprege",
        no_gradient
    )

    prepared = prepare(rows)
    model = fit(prepared, alpha)
    combo, value = choose(prepared, model)

    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"F={value:.10f}; "
        f"kandidata={len(prepared[6])}"
    )
    print(
        f"beta_min={model[1].min():.8f}; "
        f"jacina gradijentne sprege="
        f"{np.linalg.norm(model[2]):.6f}"
    )
    print(
        "Izbor: poslednje 64 kombinacije "
        "i sve njihove zamene jednog broja."
    )
    print(
        "F je strukturna GL energija za rangiranje; "
        "nije verovatnoca."
    )


def check_model(prepared, model, coefficient):
    candidates = prepared[6]
    psi = order_parameter(candidates)
    ctx, link, reference = prepared[3:6]

    assert np.all(model[1] >= 1e-4-1e-12)
    assert np.all(model[2] >= -1e-12)

    assert np.allclose(
        features(
            psi, reference, ctx, link
        ) @ coefficient,
        energy_fields(psi, model),
        atol=1e-9
    )

    rotation = np.exp(.73j)
    rotated = (
        *model[:3],
        model[3]*rotation,
        model[4],
        model[5]*rotation
    )

    assert np.allclose(
        energy_fields(psi, model),
        energy_fields(
            psi*rotation, rotated
        ),
        atol=1e-9
    )


def self_test():
    base = np.array([
        [1,2,3,4,5,6,7],
        [4,12,18,28,35,36,37],
        [33,34,35,36,37,38,39],
        [2,7,12,18,24,30,39]
    ])

    psi = order_parameter(base)
    distances = 40*np.abs(psi)**2

    assert np.allclose(
        distances.sum(axis=1), 40.
    )
    assert np.allclose(
        np.cumsum(
            distances[:, :-1], axis=1
        ),
        base
    )

    for row in base:
        r = replacements(row)

        assert (
            len(r)
            == len(np.unique(r, axis=0))
            == 224
        )
        assert all(
            hits(row, x) == 6 for x in r
        )

    history = np.tile(base, (20, 1))
    p = prepare(history)

    for alpha in ALPHAS:
        coefficient = bounded_coefficients(
            p, alpha
        )
        model = fit(p, alpha)
        check_model(p, model, coefficient)

        # KKT uslovi ogranicenog minimuma pri ucenju.
        scaled = coefficient*p[2]
        gradient = (
            (p[0]+alpha*np.eye(WIDTH)) @ scaled
            - p[1]
        )

        lower = np.zeros(WIDTH)
        lower[SIZE:2*SIZE] = (
            1e-4*p[2][SIZE:2*SIZE]
        )
        active = (
            scaled[BOUNDED]-lower[BOUNDED]
            <= 1e-9
        )

        assert (
            np.max(np.abs(gradient[FREE]))
            < 1e-7
        )
        assert np.all(
            gradient[BOUNDED][active]
            >= -1e-7
        )
        assert np.all(
            np.abs(
                gradient[BOUNDED][~active]
            ) < 1e-7
        )

        result = choose(p, model)

        assert result == predict(
            prepare(history), alpha
        )
        assert any(
            np.array_equal(result[0], x)
            for x in p[6]
        )

    altered = history.copy()
    altered[-1] = base[0]

    assert (
        predict(prepare(history[:-1]), .1)
        == predict(prepare(altered[:-1]), .1)
    )

    print(
        "Self-test OK: rekonstrukcija, fazna simetrija, "
        "stabilnost, KKT i hronologija."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*")
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    for path in args.csv or DEFAULTS:
        run_file(path, args.validation, args.test)


if __name__ == "__main__":
    main()



"""
Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 4587..4634; alpha=0.01; prosek=1.4583
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
LANDAU V6 GINZBURG-LANDAU: prosek=1.1719; pogodaka 0..7=[15, 25, 22, 2, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Isto polje bez gradijentne sprege: prosek=1.2344; pogodaka 0..7=[13, 27, 20, 4, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 3,5,6,7,11,19,25
F=-1.6690878857; kandidata=14396
beta_min=0.00010000; jacina gradijentne sprege=1.537949
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.
F je strukturna GL energija za rangiranje; nije verovatnoca.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 2860..2907; alpha=1; prosek=1.1875
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
LANDAU V6 GINZBURG-LANDAU: prosek=1.1719; pogodaka 0..7=[13, 32, 15, 3, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Isto polje bez gradijentne sprege: prosek=1.2500; pogodaka 0..7=[9, 35, 16, 3, 1, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 1,2,5,20,30,37,39
F=-0.7602217028; kandidata=14392
beta_min=0.00010000; jacina gradijentne sprege=0.252392
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.
F je strukturna GL energija za rangiranje; nije verovatnoca.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 1616..1663; alpha=0.1; prosek=1.4375
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
LANDAU V6 GINZBURG-LANDAU: prosek=1.2188; pogodaka 0..7=[13, 30, 17, 2, 2, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Isto polje bez gradijentne sprege: prosek=1.2500; pogodaka 0..7=[13, 27, 20, 3, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 1,2,4,7,11,16,17
F=-1.8330593686; kandidata=14388
beta_min=0.00010000; jacina gradijentne sprege=1.585976
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.
F je strukturna GL energija za rangiranje; nije verovatnoca.
"""

 

"""
Energija se računa za deformaciju prema prethodnoj stvarnoj kombinaciji. 
Uklonjeni su prosečno kompleksno polje i aritmetički početni nizovi.

Fajl	                Sledeća kombinacija	V6 — prosek na 64 kola	Prethodna kombinacija	Bez gradijentne sprege
Objedinjeni, red 4699	3,5,6,7,11,19,25	1,1719	                1,2188	                1,2344
Loto, red 2972	        1,2,5,20,30,37,39	1,1719	                1,2969	                1,2500
Loto Plus, red 1728	    1,2,4,7,11,16,17	1,2188	                1,2969	                1,2500

Na ovom testu nije pokazala prednost. 
Provereni su rekonstrukcija, fazna simetrija, stabilnost, uslovi ograničenog minimuma pri učenju i hronologija. 
Izbor obuhvata poslednje 64 kombinacije i sve njihove zamene jednog broja.
"""
