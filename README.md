
  # Problem trgovačkog putnika — TSP Solver (Python + PyQt)

  Ovo je edukativna Python aplikacija za rješavanje Problema trgovačkog putnika (TSP) s jednostavnim PyQt GUI-jem za unos/vizualizaciju gradova i rješenja.

  Glavne značajke

  - Interaktivni GUI (see `main.py`) za dodavanje gradova klikom, generiranje slučajnih skupova i animiranu vizualizaciju rute.
  - Implementirani algoritmi u `tsp.py`: precizni Held–Karp (DP) i heuristike: nearest neighbour + 2-opt, simulated annealing.
  - Pomoćne funkcije u `utils.py`: dekoratori (`timeit`, `log_calls`), closure factory za SA i helperi s doctestom.
  - Pozadinsko izvršavanje solvera koristeći `QThread` (`SolverThread`) kako bi GUI ostao responzivan.
  - Jednostavni testovi: doctest + `unittest` (`tests/test_tsp.py`) i pokretač `run_tests.py`.

  Ograničenja

  - Held–Karp je egzaktan, ali eksponencijalan pa je stavljen maksimum na 14 gradova.

  Minimalna podržana verzija Pythona: 3.8

  Preporučena verzija za razvoj i testiranje: 3.13.5

  Struktura repozitorija

  - `main.py` — PyQt aplikacija i GUI logika
  - `tsp.py` — implementacije algoritama i pomoćne funkcije
  - `utils.py` — dekoratori i helperi
  - `tests/` — jedinični testovi
  - `run_tests.py` — pokretač testova
  - `requirements.txt` — ovisnosti

  Brzi početak (Windows)

  1. Kreiraj virtualno okruženje i aktiviraj:

  ```powershell
  python -m venv .venv
  # PowerShell
  .\.venv\Scripts\Activate.ps1

  # cmd
  \.venv\Scripts\activate.bat
  ```

  2. Instaliraj ovisnosti:

  ```bash
  pip install -r requirements.txt
  ```

  3. Pokreni GUI:

  ```bash
  python main.py
  ```

  4. Pokreni testove (doctest + unittest):

  ```bash
  python run_tests.py
  ```

  Razvojne naredbe

  ```bash
  # Formatiranje koda
  black .

  # Staticka provjera tipova (mypy)
  python -m mypy --ignore-missing-imports .

  # Lokalni sanity-check
  python scripts/check_style.py

  # Pokreni testove
  python run_tests.py
  ```

  Kratke napomene za developera

  - Dekoratori: `utils.timeit` zapisuje trajanje u `.last_duration`, `utils.log_calls` drži zapise poziva u `.call_log`.
  - SA closure: `utils.make_simulated_annealing` vraća konfigurirani solver;
  - Tipovi: datoteke sadrže osnovne PEP484 anotacije.
  
  ---

  # TSP Solver — English quick summary

  This repository contains a small Python project implementing exact and heuristic TSP solvers with a PyQt GUI for interactive experiments and visualization.

  Quick start

  ```bash
  pip install -r requirements.txt
  python main.py
  ```

  Notes

  - Exact Held–Karp is included but limited to 14 cities in practice.
  - Use `python run_tests.py` to run doctests and unit tests.

  Files

  - `tsp.py` — solvers and helpers
  - `main.py` — GUI and wiring


# Docker

## Izrada Docker slike

```bash
docker build -t tsp-solver .
```

## Pokretanje Docker kontejnera

```bash
docker run --rm -it -e DISPLAY=host.docker.internal:0 tsp-solver
```

> **Napomena:** Budući da aplikacija koristi PyQt grafičko sučelje, za prikaz GUI-ja iz Docker kontejnera potrebno je imati odgovarajuće podešen X11/display server na host računalu.

# Snap

## Izrada Snap paketa

```bash
sudo /snap/bin/snapcraft pack --destructive-mode
```

Nakon završetka procesa generira se datoteka:

```text
tsp-solver_1.0_amd64.snap
```

## Instalacija Snap paketa

```bash
sudo snap install tsp-solver_1.0_amd64.snap --dangerous --devmode
```

## Pokretanje aplikacije

```bash
tsp-solver
```

## Uklanjanje Snap aplikacije

```bash
sudo snap remove tsp-solver
```
