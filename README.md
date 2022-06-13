# PROJEKT SW
Wczytane zdjęcie jest skalowane i tworzone są maski dla każdego koloru przy pomocy funkcji inRange.  

Następnie korzystając z funkcji zaczerpniętej z *https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv* usuwane są cienie.  

Korzystając z adaptiveThreshold, algorytmu Cannyego oraz maski hsv (bez białego koloru) tworzony jest obraz binarny.  
Na obrazie binarnym znajdowane są kontury przy pomocy findContours, a następnie odrzucane jeśli ich wilekość jest poza określonymi wartościami.  
Jeśli wielkość kontura nie spełnia wymogów ale jego convexHull je spełnia to jest on dodawany zamiast kontura.  
Na bazie konturów wyznaczane są bounding boxy, które użyte będą do wyznaczania kolorów.  

Wykorzystując wcześniej utworzone maski dla każdego typu klocków, kontury są porównywane z konturami modelowymi w funckcji match_contours.  
Funkcja match_contours zwraca listę obiektów zaklasyfikowanych jako dany typ klocka. Ta lista wykorzystywana jest w funkcji choose_lower do wybrania najmniejszej wartości klasyfikacji dla danego typu klocka.  
Funckja check_convex rozwiązuje problem mylenia niektórych typów obiektów. Wykorzystuje ona convexHull kontura jak i sam kontur do obliczenia różnicy między nimi.  

Do wykrywania kolorów wycinany jest bounding box zawierający obiekt, a następnie aplikowana jest maska każdego koloru funckją bitwise_and. Jeśli suma z obrazu wynikowego nie jest równa 0 to w obrazie znajduje się obiekt danego koloru.
