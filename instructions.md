## Uruchomienie skryptu generującego dane

#### Wygeneruj tylko brakujące dane (criteria 4 i 5)
python generate_monte_carlo.py

#### Wygeneruj konkretne rozmiary
python generate_monte_carlo.py --criteria 4 5

#### Wymuś regenerację wszystkiego
python generate_monte_carlo.py --force

#### Wymuś regenerację wybranych
python generate_monte_carlo.py --criteria 3 4 5 --force

# Ustaw alpha=3.0 dla wszystkich rozmiarów
python generate_monte_carlo.py --alpha 3.0

# Jeden sigma_low dla wszystkich n
python generate_monte_carlo.py --sigma-low 0.7 --sigma-high 1.1

# Per-criteria: różne sigma_low dla n=3 i n=4, reszta z domyślnych
python generate_monte_carlo.py --criteria 3 4 5 --sigma-low 3:0.85 4:0.75

# Pełny override
python generate_monte_carlo.py --sigma-low 3:0.85 4:0.75 5:0.70 --sigma-high 3:1.2 4:1.05



## Wyświetlenie danych

#### 3 macierze extreme z testu, criteria=4
python load_monte_carlo.py -n 4 -p test -s extreme -c 3

#### 10 macierzy high consistency z treningu, criteria=5
python load_monte_carlo.py --criteria 5 --phase train --stratum high --count 10

#### Wszystkie strata (all), 5 macierzy z validacji
python load_monte_carlo.py -n 4 -p valid

#### Tylko statystyki - ile macierzy mamy w każdym stratum i fazie
python load_monte_carlo.py -n 4 --info

#### Tylko wagi dla n=3 i n=4, stratum extreme
python load_monte_carlo.py -n 3 4 --weights -s extreme -c 5

#### Wagi z testu
python load_monte_carlo.py -n 4 -p test --weights -c 10

#### Wczytaj tylko pliki wygenerowane z alpha=3.0
python load_monte_carlo.py -n 5 --alpha 3.0 --weights

#### Wczytaj macierze wygenerowane z sigma=0.75
python load_monte_carlo.py -n 5 --sigma 0.75 -s high

#### Filtruj po obu
python load_monte_carlo.py -n 4 --alpha 2.0 --sigma 0.90 -s extreme -c 5
