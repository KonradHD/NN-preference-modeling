# Architektura modelu

Rozwój dziedziny nauki związanej z budową, zastosowaniem oraz optymalizacją sztucznych sieci neuronowych postępuje w astronomicznych tempie, dlatego co chwilę pojawiają się nowe, ulepszone rozwiązania poszczególnych problemów. Wybór architektur jest naprawdę bogaty, aczkolwiek analiza zostanie zawężona do trzech typów sieci, które najbardziej pasują do opisywanego zagadnienia. Najbardziej obiecującym rozwiązaniem wydaje się być perceptron wielowarstwowy, nie podważając również licznych zalet sieci syjamskich oraz grafowych.

## Perceptron wielowarstwowy
\label{subsec: MLP}

### Mechanizm działania
W dalszych rozważaniach dotyczących Perceptron wielowarstwowy wykorzystywany będzie skrót MLP pochodzący od angielskiego tłumaczenia - Multi-Layer Perceptron. Jest on najbardziej powszechną oraz uniwersalną architekturą, co wiąże się z jego użytecznością. Warto się jednak początkowo skupić na mechanice działania sieci zaczynając od opisu fundamentalnych jej elementów - sztucznych neuronów. Przyjmuje on sygnały wejściowe od innych, połączonych z nim neuronów i oblicza ich kombinację liniową. Dla danego wektora sygnałów wejściowych $\mathbf{x} = [x_1, x_2, \dots, x_k]^T$, wektora wag połączeń $\mathbf{w} = [w_1, w_2, \dots, w_k]^T$ oraz wyrazu wolnego $b$, wartość pobudzenia neuronu $z$ wyraża się wzorem \cite{Bishop2006}:

\begin{equation}
    z = \sum_{i=1}^{k} w_i x_i + b = {w}^T {x} + b
    \label{eq: pobudzenie_neuronu}
\end{equation}

Jako, że jest to liniowa kombinacja, a sieć potrzebuje być w stanie odnaleźć nieliniowe zależności, każda wartość $z$ poddawana jest translacji przez tzw. funkcję aktywacji $\sigma$. Zatem ostateczny sygnał wyjściowy neuronu wynosi $a = \sigma(z)$. To właśnie obecność tej nieliniowości pozwala sieciom MLP aproksymować dowolnie złożone funkcje ciągłe \cite{Goodfellow2016}. \newline

W architekturze MLP neurony w warstwach są gęsto połączone, to znaczy, że odbierają sygnały od wszystkich neuronów z poprzedniej warstwy. Uogólniając równanie \eqref{eq: pobudzenie_neuronu} do postaci macierzowej, dla $l$-tej warstwy, wyjściowy wektor aktywacji wynosi:

\begin{equation}
    {h}^{(l)} = \sigma\left( {W}^{(l)} {h}^{(l-1)} + {b}^{(l)} \right)
\end{equation}

gdzie ${W}^{(l)}$ to macierz wag, ${b}^{(l)}$ to wektor przesunięć warstwy. Powołując się na wcześniejsze rozważania, wektor wejściowy do sieci ${h}^{(0)}$ (dla macierzy obejmującej $3$ kryteria) został przedstawiony w równaniu \eqref{eq:wektor_wejsciowy}. \newline

### Propagacja wsteczna
Sam proces uczenia rozpoczyna się po wykonaniu pierwszych predykcji. Do wyznaczania kierunku poprawy wag sieci wykorzystywana jest funkcja straty \eqref{eq: funkcja_straty}, która zwraca wartość błędu obecnych predykcji, stosowany do aktualizacji wag poszczególnych warstw $\mathbf{W}$, a także ich wektorów aktywacji $\mathbf{b}$ w taki sposób aby zminimalizować końcową wartość straty. Cały proces jest przeprowadzany z wykorzystaniem algorytmu propagacji wstecznej \cite{Goodfellow2016}, który oparty jest na regule łańcuchowej liczenia pochodnych funkcji złożonych, propaguje informację o błędzie od warstwy wyjściowej aż do warstwy wejściowej \cite{wiki:propagacja_wsteczna}. Proces umożliwia obliczenie gradientu dla każdego parametru sieci. W następnej kolejności, algorytm optymalizatora aktualizuje wagi sieci proporcjonalnie do wielkości i kierunku wyznaczonego gradientu, dążąc do lokalnego (lub w najlepszym przypadku globalnego) minimum funkcji celu.\newline

### Uwarunkowania architektury MLP
Struktura danych wejściowych jest reprezentowana w postaci wektora \eqref{eq:wektor_wejsciowy} o długości:
\begin{equation}
    K = \frac{n(n-1)}{2}
    \label{eq:dl_wektora_wejsciowego}
\end{equation}

gdzie $n$ jest liczbą kryteriów. Stanowi on naturalną reprezentacje danych dla środowiska gęsto połączonych warstw sieci. Fundamentalną zaletą perceptronu wielowarstwowego jest globalna analiza całego problemu decyzyjnego. W procesie przepływu sygnału sieć analizuje wszystkie oceny ekspertów jednocześnie, co pozwala ukrytym warstwom sieci na analizę całego kontekstu i wychwytywanie złożonych, wielowymiarowych zależności. Co więcej, MLP charakteryzuje się prostotą wdrożenia oraz implementacji, a także są stosunkowo szybkie w trenowaniu \cite{Bishop2006}.\newline

Trzeba mieć jednak na uwadze kilka ograniczeń m.in. brak inwariantności sieci na permutacje, co objawia się faktem, że przetasowanie kolejności kryteriów w macierzy porównań, wpłynie na ich końcowe wartości priorytetów. Jest to zjawisko niedopuszczalne w kontekście ostatecznych rezultatów, ponieważ kolejność zapisu kryteriów nie jest w żaden sposób wiążąca. Kolejnym niepożądanym zjawiskiem, które wymaga rozwiązania, jest z góry zdefiniowana liczba wejść oraz wyjść z sieci. Jedną z metod umożliwiającą predykcję różnych wielkości macierzy jest ustalenie maksymalnego rozmiaru wejścia i zastosowaniu maskowania polegającego na wypełnieniu brakujących elementów zerami. Takie podejście rodzi jednak bardzo duży problem, ponieważ w przestrzeni logarytmicznej wartość $0$ oznacza zupełną równoważność dwóch kryteriów: $a_{ij} = 1 \implies \ln(1) = 0$, co w konsekwencji jednoznacznie dyskredytuje tę metodę. Wprowadzenie innej wartości maskującej - bardzo niewielką liczbę ujemną doprowadziłoby z kolei do destabilizacji uczenia i zakłamania wyników. Jedynym rozwiązaniem pozostaje trenowanie oddzielnych sieci dostosowanych do konkretnych rozmiarów macierzy AHP.   


Oznacza to, że bez zastosowania specjalistycznych zabiegów, sieć jest w stanie przyjąć tylko jedną, wcześniej zdefiniowaną liczbę kryteriów $n$. 



# Specyfikacja architektury MLP
Biorąc pod uwagę zalety oraz wady przedstawionych rozwiązań architektonicznych, a także aspekt dopasowania do struktury danych wejściowych, w dalszych rozważaniach zostanie wykorzystany model MLP posiadający właściwości pozwalające na uzyskanie obiecujących rezultatów końcowych. Warto zaznaczyć, że odrzucone architektury również wykazują duży potencjał badawczy, aczkolwiek ze względu na limitowane zasoby oraz przede wszystkim cel pracy, który koncentruje się na ewaluacji opracowanej funkcji straty, nie zostaną one poddane dalszej analizie. 



### Struktura warstw ukrytych
Priorytetem warstw ukrytym w zaproponowanej architekturze jest wychwycenie nieliniowych zależności, często sprzecznych relacji panujących w osądach ekspertów. Jako, że sieć przyjmuje wektor zlogarytmowanych ocen, którego długość przedstawiona jest wzorem \eqref{eq:dl_wektora_wejsciowego}, pierwsza warstwa ukryta odpowiedzialna jest za rozszerzenie wymiarowości problemu do $2K$ lub $4K$ neuronów. Celem jest ułatwienie identyfikacji nieliniowych wzorców oraz rozpoznawania niespójności w ocenach. Dochodząc w ten sposób do największej warstwy mającej x neuronów, architektura przyjmuje kształt lejka, ponieważ następne warstwy stopniowo zmniejszają wymiarowość, filtrując tym samym szum i ekstraktując jedynie najbardziej istotne cechy, aż do osiągnięcia wymiaru $n$ w warstwie wyjściowej odpowiadającego liczbie kryteriów \cite{Bishop2006}. 
% Tutaj można rozbudowywać model, dodawać jakieś wymyślne warstwy
% dodać kod

### Funkcja aktywacji
Umożliwienie sieci znajdowanie nieliniowych zależności w danych odbywa się poprzez wykorzystanie odpowiedniej funkcji aktywacji w warstwach ukrytych \cite{Goodfellow2016}. Bez tego, zastosowanie nawet bardzo rozbudowanego perceptronu wielowarstwowego aproksymowałoby jedynie liniowe funkcje złożone. Obecnie standardem stała się funkcja ReLu pochodząca od angielskiego \textit{Rectified Linear Unit}, która ma postać: 

\begin{equation}
    f(x) = \max(0, x)
\end{equation}

Jak łatwo zauważyć, dla każdej ujemnej wartości aktywacji sumarycznej $x$, odpowiedź neuronu zostanie wyzerowana, co w przypadku pojawienia się ujemnych wartości, może prowadzić do zjawiska tzw. martwych neuronów \cite{Goodfellow2016}. Polega ono na stałym wygaszeniu neuronów i braku reakcji na nowe dane, gdyż jego wartość aktywacji niezmiennie wynosi $0$. W przypadku analizowanego zagadnienia, zwróciwszy uwagę na dobrane człony funkcji straty $L_{fit}$ \eqref{subsec: fit} oraz $L_{COP}$ \eqref{subsec: cop}, pojawiają się różnice logitów $s_i - s_j$, co oznacza generowanie ujemnych przepływów gradientu w trakcie procesu propagacji wstecznej. W naturalny sposób nasuwa się wniosek, że wykorzystanie funkcji ReLu może prowadzić do spowolnienia procesu nauki i otrzymanie mniej dokładnych rezultatów.\newline

Alternatywą, pochodzącą z tej samej rodziny funkcji, która w tym przypadku sprawdzi się znacznie lepiej jest funkcja LeakyReLU w postaci: 

\begin{equation}
    f(x) = \max(\alpha x, x)
\end{equation}

gdzie $\alpha$ jest niewielkim współczynnikiem, którego wartość oscyluje początkowo zostaje ustalona na $0{,}01$. Takie rozwiązanie umożliwia przepływ kontrolowanego, ujemnego gradientu, co zapewnia nieustaną wymianę informacji pomiędzy wszystkimi warstwami sieci. Warto podkreślić, że zastosowanie funkcji aktywacji dotyczy jedynie warstw ukrytych, a warstwa wyjściowa będzie generowała surowe logity, które w następnym etapie przekształcane są na rozkład prawdopodobieństwa za pomocą funkcji Softmax \eqref{subsec: softmax}.  

\paragraph{Mechanizmy regularyzacji}
Pomimo posiadania ogromnego zbiory danych uczących, którego wielkość ograniczają jedynie zasoby czasowe oraz pamięciowe, to zastosowanie mechanizmu regularyzacji jest kluczowe, aby uniknąć zjawiska przeuczenia, zapamiętywania przez sieć konkretnych wzorców, a nie generalizowania. Bowiem sieć może nauczyć się specyfiki rozkładu nakładanego na macierze szumu. Skuteczną techniką regularyzacji jest mechanizm losowego usypiania neuronów, szerzej znany pod angielską nazwą - Dropout. Bazuje na wyłączaniu określonego odsetku losowo wybranych neuronów w warstwach ukrytych. Wywiera tym samym potrzebę tworzenia rozproszonych oraz redundantnych reprezentacji wiedzy, co skutecznie przeciwdziała wnioskowaniu opartemu na jednej, lokalnej relacji \cite{Srivastava2014}. Zamiast tego, wymusza na sieci spojrzenie globalne, co dodatkowo ma pozytywny wpływ na stabilność rezultatów.  
% dodać może Weight Decay 


### Optymalizator sieci
Istotnym elementem każdej sieci, odpowiedzialnym za aktualizacje wag połączeń pomiędzy neuronami, jest algorytm optymalizacyjny dążący do zminimalizowania całkowitego błędu \cite{Goodfellow2016}. W rozwiązaniach klasycznych, których głównym przedstawicielem jest stochastyczny spadek gradientu, wszystkie wagi uaktualniane są z identyczną intensywnością wykorzystując stały krok uczenia. Jednakże, w przypadku wieloczłonowej funkcji straty \eqref{eq: funkcja_straty}, takie podejście będzie prowadziło do destabilizacji uczenia. Wynika to z faktu, że każdy człon funkcji posiadają swoją dynamikę, a także skalę obliczanych błędów. Przykładowo, gdy sieć odwróci ranking kryteriów, moduł $L_{COP}$ wygeneruje dosyć duży gradient, podczas gdy człon $L_{stab}$ stale wysyła bardzo niewielkie korekty. Odpowiedzią na ten problem jest algorytm Adam, który działa w sposób zbliżony do mechanizmu adaptacyjnego. Przyjmuje on indywidualne spojrzenie na każdy parametr sieci i na bieżąco dostosowuje tempo jego zmian na podstawie wcześniejszych korekt. Opiera swoje działania na dwóch mechanizmach: 
\begin{itemize}
    \item Momentum $\hat{m}_t$ - odpowiada za śledzenie poprzednich zmian wartości wag w postaci średniej ruchomej samych gradientów. W praktyce oznacza to, że jeżeli gradient nieustannie zmierza w tym samym kierunku, to $m_t$ zwiększa swoją wartość umożliwiając szybsze pokonanie płaskich odcinków funkcji straty. 
    \item Algorytm RMSProp $\hat{v}_t$ - odpowiada za analizę zmienności gradientów, obliczając średnią ruchomą z ich kwadratów \cite{Kingma2014}. Dla parametrów o stałych, niewielkich gradientach RMSProp automatycznie powiększa krok uczenia, natomiast dla zmiennych, chaotycznych gradientów krok jest zmniejszany, co powoduje tłumienie niekorzystnych oscylacji. Krótko mówiąc, pełni role stabilizatora. 
\end{itemize}


Wykorzystując te mechanizmy aktualizowany parametr $\theta$ w kroku $t$ wynosi: 

\begin{equation}
    \theta_t=\theta_{t-1}-\frac{\alpha}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t
\end{equation}

gdzie $\alpha$ jest bazowym krokiem uczenia, a $\epsilon$ jest niewielką stałą rzędu $10^{-8}$ stosowanym jako bezpiecznik przed dzieleniem przez zero. Dzięki takim mechanizmom, model potrafi autonomicznie analizować zróżnicowane sygnały pochodzące z trójczłonowej funkcji straty.

### Inicjalizacja wag
Często pomijanym aspektem w procesie uczenia sztucznych sieci neuronowych jest ustalenie początkowej wartości wag, co może przyspieszyć cały proces i zwiększyć jego stabilność. Ze względu na zastosowanie LeakyReLu jako funkcji aktywacji w warstwach ukrytych, wykorzystano inicjalizację He Kaminga \cite{he2015delving}. Nie zakłada ona symetrii funkcji aktywacji, a właściwie dostosowuje wariancję startowych wag, kompensując specyfikę funkcji LeakyReLu (tłumi wartości ujemne), co jest jednym z elementów walki z zanikającym gradientem. 