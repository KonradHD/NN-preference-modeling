# Konstrukcja funkcji

Problem zaszumienia danych jest często spotykany w środowisku rzeczywistym, w kontakcie z niedoskonałą, ludzką percepcją. W problemach z dziedziny uczenia maszynowego, nie zawsze architektura modelu jest aspektem, który wymaga dostosowania. Nierzadko kluczowe znaczenie ma dobór odpowiedniej, odpornej na szum, funkcji straty \cite{Ghosh2017}. Na podstawie badań na temat odpornych na szum funkcji celu, należy wyróżnić fakt, że najbardziej obiecujące wyniki dają transformacje ograniczone. Eksperymenty zostały przeprowadzone dla problemu klasyfikacji i szumu w etykietach, ale posiadają mocną analogie w kierunku wyznaczania wag priorytetów. Jeżeli ekspert popełni jeden, znaczny błąd, nieograniczona funkcja straty zostanie przez niego zdominowana. Gdy ogromny błąd zostanie jednak po części zignorowany, sieć będzie mogła lepiej dopasować wektor wag do pozostałych, spójnych elementów macierzy.\newline

Warto również zastanowić się nad celem, który został postawiony. Wynikowy wektor wag priorytetów powinien być zarówno stabilny - odporny na niespójności w odpowiedziach ekspertów, dopasowany do danych wejściowych oraz zachowywać warunek COP. Nie sposób spełnić tych trzech warunków za pomocą uniwersalnego, jednolitego przekształcenia. Z tego powodu w realizowanych eksperymentach funkcja straty $L$ przyjmie postać:  

\begin{equation}
    L = \lambda_1 L_{fit} + \lambda_2 L_{COP} + \lambda_3 L_{stability}
    \label{eq: funkcja_straty}
\end{equation}

Gdzie parametry $\lambda_1$, $\lambda_2$, $\lambda_3$ będą oznaczały poszczególne wagi członów funkcji odpowiedzialnych za kolejno dopasowanie do macierzy, COP oraz stabilność. Ułatwia to manipulację istotnością poszczególnych elementów i maksymalizowanie wydajności sieci.   

## Kryterium stabilności

Kryterium stabilności jest dosyć mało intuicyjnych, aczkolwiek stosowanym zabiegiem regularyzującym. Zmusza on sieć, by w obliczu sprzecznych informacji spowodowanych szumem w macierzy porównań nie stawała się nadmiernie "pewna siebie" i nie faworyzowała skrajnie jednego kryterium, a dążyła do zrównoważonego wektora wag. Ponadto w praktyce, przypadek, w którym jedno kryterium bezwzględnie deklasowałoby pozostałe występuje niezwykle rzadko. Zasadą działania tego członu jest zbliżona do regularyzacji entropijnej. Mianowicie, błąd modelu jest zwiększany podczas estymacji skrajnie wysokich lub niskich wartości wag \cite{Pereyra2017}. Zabiega to znacznemu wpływowi niejednoznacznych danych wejściowych, które bez analizowanego członu, spowodowałyby bezpodstawne zmiany w rozkładzie wartości wag. 

### Zastosowanie dywergencji Kullbacka-Leiblera

Odpowiedzią na ten problem jest dywergencja Kullbacka-Leiblera oznaczana jako $D_{KL}$ i wykorzystywana jest do określenie odległości pomiędzy dwoma rozkładami prawdopodobieństwa. Jej działanie jest zbliżone do popularnej techniki wygładzania etykiet w problemach klasyfikacji \cite{Pereyra2017}. Co więcej, wygładzanie etykiet względem rozkładu jednostajnego $u$ jest równoważne z wykorzystaniem w funkcji celu $D_{KL}$ wyliczanej od rozkładu jednostajnego do rozkładu wyestymowanego przez sieć $w$. Natomiast odwrócenie kierunku dywergencji, czyli zdefiniowanie jej jako $D_{KL}(w\vert{}\vert{} u)$ pozwala uzyskać matematyczny mechanizm karania za nadmierną pewność \cite{Pereyra2017}. Z tego powodu jedną z propozycji jest przedstawienie członu odpowiedzialnego za stabilność jako: 

\begin{equation}
    L_{stability} = D_{KL}(w\vert{}\vert{} u) = \sum_{i=1}^{n} w_i \ln(w_i \cdot n)
\end{equation}

Dywergencja obliczana jest względem rozkładu jednostajnego, ponieważ w tym rozkładzie każde z $n$ kryteriów posiada identyczną wagę równą $\frac{1}{n}$. Sieć traktuje ten rozkład jako bezpieczny punkt odniesienia w skrajnych warunkach. 


% dodać inne sposoby w postaci entropii wymienionych w prezentacji 

### Wprowadzenie progu aktywacji kary
Wykorzystanie tego członu wiąże się również z pewnymi niebezpieczeństwami. Z pewnością, zjawiskiem niepożądanym jest nadmierne wypłaszczanie wartości wag priorytetów i sztuczne upodobnianie go do rozkładu normalnego. Z tego powodu wartość wagi $\lambda_3$ nie powinna być wysoka w porównaniu z pozostałymi wagami. Elementem, który również należałoby wprowadzić, w celu bardziej efektywnej pierwszej fazy nauki, w której sieć uczy się podstawowych zależności wynikających z macierzy porównań - jest próg aktywacji. Zastosowanie sztywnej kary od samego początku uczenia mogłoby doprowadzić do zdecydowanego spowolnienia tego procesu, albo nawet zakończenia nauki bardzo blisko rozkładu normalnego. Aby wykluczyć te sytuacje kara za dywergencję nie będzie naliczana od początku przetwarzania danych przez sieć, a dopiero w momencie, gdy entropia wygenerowanego rozkładu przekroczy wcześniej ustalony próg $\Gamma$. Jako punkt odniesienia zostanie wykorzystana entropia Shannona przyjmującą postać \cite{Pereyra2017}: 

\begin{equation}
    H(w(y\vert{}x)) = -\sum_{i} w(y_i\vert{}x) \ln(w(y_i\vert{}x))
\end{equation}

gdzie $w(y_i\vert{}x)$ to wartość $i$-tej wagi wygenerowanej przez sieć dla danego wejścia.
Natomiast wspomniany próg $\Gamma$ zostanie wyznaczony empirycznie, na podstawie szeregu eksperymentów oraz analizy wyników końcowych, a także tempa nauki. 


## Człon zachowania porządku
\label{subsec: cop}

Kryterium, którego spełnienie będzie wysuwało się na pierwszy plan w analizie wyników jest wcześniej wspomniane COP. Temat istotności tego warunku został przybliżony w podrozdziale \eqref{subsec: metoda_EVM}, a jego zachowanie staje się priorytetem w ocenach modelu. To m.in. na podstawie COP, zostanie oceniona skuteczność zaprezentowanej metody. Chociaż człon stabilności zapobiega generowaniu skrajnych wartości wag, a kryterium rekonstrukcji odpowiada za dopasowanie wyników do macierzy porównań, to żaden z nich nie zabezpieczenia wektora przed błędami logicznymi.\newline

% Troche AI 
### Mechanizm kary z marginesem separacji
Metryką wykorzystywaną do określenia jakości spełniania warunku COP jest współczynnik korelacji rangowej $\tau$ Kendalla, który wnikliwiej został przedstawiony w sekcji \eqref{sec: metryki}. Jak wykazał Thorsten Joachims w swojej pracy obejmującej uczenie funkcji rankingujących na podstawie  preferencji użytkowników \cite{Joachims2002}, minimalizacja liczby niezgodnych par, w których wyliczona waga kryterium nie zgadza się ze współczynnikami porównań, jest problemem NP-trudnym. Co więcej, zwykłe zliczanie liczby błędów predykcji sieci (albo jest błąd = 1, albo go nie ma = 0), jest funkcją w głównej mierze płaską, co oznacza, że jej pochodna w prawie każdym punkcie wynosi $0$. Sieć ucząca się za pomocą metody spadku gradientu, nie uzyskuje informacji, czy drobna zmiana parametrów perceptronów prowadzi do poprawy wyniku. W celu rozwiązania zaistniałego problemu wprowadza się nieujemne zmienne dopełniające $\xi_{i,j} \ge 0$ i minimalizuje ich sumę $\sum \xi_{i,j}$ \cite{Joachims2002}. W takim układzie, warunek poprawnego uporządkowania pary kryteriów przyjmuje postać nierówność, a przekładając tę koncepcję architekturę sieci neuronowych, dla każdej pary kryteriów, gdzie ekspert stwierdził, że kryterium $i$ jest ważniejsze od kryterium $j$ ($a_{ij} > 1$), człon $L_{COP}$ ma postać: 

\begin{equation}
    L_{COP} = \sum_{i,j: a_{ij} > 1} \max(0, -(s_i - s_j) + m)
\end{equation}

gdzie $s_i$ oraz $s_j$ oznaczają wartości logitów poszczególnych kryteriów.
Parametr $m > 0$ staje się marginesem mówiącym jaka jest akceptowalna różnica pomiędzy wyestymowanymi logitami $s_i$ oraz $s_j$. Nie dopuszcza on możliwość, w której kryteria są sobie równe, a jego wartość jest dobierana empirycznie na podstawie wyników modelu.\newline

Mechanizm funkcjonowania tak dobranego cżłonu jest intuicyjny i dopuszcza dwie możliwości:
\begin{enumerate}
    \item Brak kary - w przypadku, kiedy sieć poprawnie przewidzi relacje pomiędzy parami logitów, ale również ich różnica będzie większa od dobranego marginesu, wyrażenie $-(s_i - s_j) + m$ staje się niedodatnie, więc człon zostanie wyzerowany.
    \item Aktywacja kary - w sytuacji, gdy sieć przekłamie kolejność, lub różnica pomiędzy logitami będzie mniejsza od dopuszczalnego marginesu, wartość kary rośnie liniowo wraz ze skalą popełnionego błędu.
\end{enumerate}

### Zamiana na model Bradley'a-Terry'ego
Alternatywną opcją jest wykorzystanie omawianego w podrozdziale \eqref{subsec: bradley} modelu Bradley'a-Terry'ego, który pozwala wyeliminować potrzebę wyznaczania marginesu separacji $m$. Mianowicie, można zastosować probabilistyczną funkcję kosztu opartą na entropii krzyżowej, która zestawia ze sobą przewidywane przez sieć prawdopodobieństwo $P_{ij}$ z prawdopodobieństwem docelowym wyznaczonym przez ekspertów $\bar{P}_{ij}$. 
\begin{itemize}
    \item $\bar{P}_{ij} = 1$, gdy decyzją eksperta $a_{ij} > 1$
    \item $\bar{P}_{ij} = 0.5$, dla $a_{ij} = 1$ 
    \item $\bar{P}_{ij} = 0$, kiedy kryterium $j$ zostało ocenione na mniej istotne niż kryterium $i$ - $a_{ij} < 1$ 
\end{itemize}
Po podstawieniu postaci funkcji logistycznej \eqref{eq:funkcja_logistyczna}, koszt przypisany pojedynczej parze porównywanych elementów przyjmuje postać \cite{Burges2005}:

\begin{equation}
    C_{ij} = -\bar{P}_{ij}o_{ij} + \log(1 + e^{o_{ij}})
    \label{eq: koszt_bradley}
\end{equation}

\noindent gdzie $o_{ij} = s_i - s_j$ oznacza różnicę logitów wygenerowanych przez sieć dla kryteriów $i$ oraz $j$. Adoptując wyrażenie \eqref{eq: koszt_bradley} do koncepcji warunku zachowania porządku dla całej macierzy porównań, sumowany będzie koszt tylko dla relacji $a_{ij} > 1$, dla której przyjmujemy $\bar{P}_{ij} = 1$.

\begin{equation}
    L_{COP} = \sum_{i,j: a_{ij} > 1} \left( -o_{ij} + \log(1 + e^{o_{ij}}) \right) = \sum_{i,j: a_{ij} > 1} \log(1 + e^{-(s_i - s_j)})
\end{equation}

Zastosowanie takiego członu zachowania porządku posiada zarówno swoje zauważalne wady jak i zalety. Przede wszystkim jest funkcją gładką w całej dziedzinie, a więc jej gradient maleje płynnie w miarę zbliżania się do poprawnego wyniku. Jednakże, koszt \eqref{eq: koszt_bradley} będzie malał do zera tylko w sytuacji, kiedy różnica logitów będzie rosła do nieskończoności, co w praktyce oznacza, że zaprezentowany człon COP powoduje stałe rozsuwanie się wartości wag.   


## Moduł dopasowania do danych
\label{subsec: fit}

Kryterium najbardziej klasycznym dla uczenia maszynowego z wykorzystaniem sieci neuronowych jest moduł dopasowania do danych wejściowych. Pomimo posiadania członów $L_{stab}$ oraz $L_{COP}$, które głównie odpowiadają za stabilność wyników oraz eliminacje błędów logicznych w danych, to same nie zapewnią wyestymowania poprawnego wektora. Nie biorą bowiem pod uwagę stosunków, które powinny spełniać poprawne wartości wag priorytetów. Zgodnie z założeniem metody AHP wartość stosunku wagi odpowiadającej $i$-temu kryterium do wagi $j$-tego kryterium powinien odpowiadać ocenie eksperta: $\frac{w_i}{w_j} \approx a_{ij}$ \cite{Saaty1980}. Z tego wynika, że najbardziej intuicyjnym sposobem obliczenia błędu rekonstrukcji byłoby bezpośrednie przyrównanie wynikowych $\frac{w_i}{w_j}$ do $a_{ij}$. W takim przypadku należy jednak wyróżnić dwa problemy wynikające z charakteru danych.
\begin{enumerate}
    \item Asymetria skali błędu, o którym mowa była w podrozdziale \eqref{subsec: koncept_modelu}, która wiąże się z faktem, że wartość błędu wygenerowanego dla kryteriów dominujących [2, 9] będzie większy niż w przypadku mniej ważnych czynników $[\frac{1}{9}, \frac{1}{2}]$.
    \item Gwałtowny wzrost gradientu spowodowany dzieleniem $\frac{w_i}{w_j}$ w przypadku, kiedy mianownik zbliża się do zera.
\end{enumerate}

Operacją, która eliminuje obydwa problemy jest sprowadzenie błędu do przestrzeni logarytmicznej nawiązującej do metody logarytmicznych najmniejszych kwadratów zaprezentowanej przez Crawforda i Williamsa \cite{Crawford1985}.\newline

Wyjściowe wartości logitów zostaną poddane działaniu funkcji Softmax \eqref{eq:softmax}, więc logarytm naturalny ze stosunku dwóch wag upraszcza się bezpośrednio do różnicy odpowiadających im logitów:

\begin{equation}
    \ln\left(\frac{w_i}{w_j}\right) = \ln\left(\frac{\exp(s_i)}{\exp(s_j)}\right) = s_i - s_j
\end{equation}

Dzięki użyciu powyższego przekształcenia, relacja została sprowadzona do stabilnego numerycznie wyrażenia $s_i - s_j \approx \ln(a_{ij})$. Wykorzystując dodatkowo analizę odporności funkcji straty na szum informacyjny \cite{Ghosh2017}, moduł $L_{fit}$ można zdefiniować jako średni błąd kwadratowy lub bezwzględny wyliczany po wszystkich parach porównań z pominięciem głównej przekątnej macierzy:

\begin{equation}
    L_{fit} = \frac{2}{n(n-1)} \sum_{i < j} \left| (s_i - s_j) - \ln(a_{ij}) \right|^p
\end{equation}

gdzie n oznacza liczbę kryteriów, a parametr $p \in \{1, 2\}$ określa charakter kary. W zależności od wartości parametru $p$, rozgraniczone zostaną dwie wersję modułu dopasowania do danych. Wybór $p = 2$ będzie odpowiadało minimalizowaniu błędu średniokwadratowego w przestrzeni logarytmicznej, co dla spójnych macierzy reprezentuje podejście metody średniej geometrycznej \cite{Crawford1985}. Natomiast, przyjęcie $p = 1$ reprezentowałoby błąd bezwzględny MAE, który zapewnia odporność na wysoce błędne oceny eksperta, ograniczające siłę ich gradientu \cite{Ghosh2017}.