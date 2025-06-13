# TODO

## 1. Skończenie pierwszego etapu:
        - dla wielowymiarowych można drzewka dodać do sprawozdania
- dokończenie sprawozdania i poprawienie go
- tytuły pod obrazkami

## 2. Przywrócenie funkcjonalności dla drugiego etapu:
- ranking metod określania stabilności

## 3. Trzeci etap (póki co minimalne wymagania):

- **Dane:**
    - przeanalizować excela, wygenerować csv
    - dodać metodę temperaturową z temperatury pokojowej do naszych metod

- **Model:**
    - Na podstawie próbek i składu procentowego budujemy regresor
    - mając dwie próbki ropy, zmieszane w danych proporcjach, rozpoznane jako lekka/średnia/ciężka, model mówi nam jaki może być index stabilności, Ta       procedura powtórzona cztery razy dla każdej metody badania stabilności robimy osobny model
    - alternative: Zamiast klasyfikacje lekka średnia ciężka możemy podawać np ciężar i mamy informacje jak procentowo możemy dobrac żeby je zmieszać ze sobą i mieć index stabilności po yxej pewnego punktu określającego czy jest stabilna czy niestabilna
  - **ostatecznie potencjalnie ustalenia na początek:** mamy gęstość i skład chemiczny probek, przeprowadzamy klasyfikacje, oceniamy stabilność na regresorze stabilności --> wynik: Czy dane próbki można ze sobą zmieszać?

- **GUI:**
    - funkcjonalność trzeciego etapu najważniejsza
    - reszta funkcjonalności może być wstawiona jako oddzielna rzecz do włączenia czy coś, czyli pokazanie wykresów wszystkich dla podanych danych


- **Sprawozdanie:**
    - trzeba napisać
