# NTO-IRS-2025
Репозиторий решения заданий финала НТО ИРС 2025 команды "План Б -которого нет-"
Repository of solutions to the tasks of the final stage of the final stage of NTO IRS 2025 of the "План Б -которого нет-" team

**<h1>Математика робота</h1>**
Одной из главных задач для решения всех 3 этапов является управление роботом. Для управления роботом были использовали следующие методы: feed forward, двойной П-регулятор. 
Была измерена длина окружности колеса двумя различными методами: через формулу окружности, через измерение мерной лентой. В обоих случаях были получены равные значения — 14,13 см. 

> [!TIP]
> Длина окружности вычисляется по формуле: $S=2πR$.

$S=4,5 × 3,14=14,13 см$.
   
   ![5352653806854334945](https://github.com/user-attachments/assets/537c0e97-4390-45a2-9f40-221018b8a53c)
   *Рис. 1. Диаметр колеса*


Длина окружности может быть вычислена также при помощи мерной ленты.

$S=14,13 см$
   
   ![5352653806854334942](https://github.com/user-attachments/assets/22316651-d011-4050-a6a4-fdfb0f4bfb09)
   *Рис. 2. Длина окружности колеса*


Для каждого значения ШИМ (100, 150, 200) было измерено время, за которое робот проезжал 14,13 см для левого и правого колёс. Это было сделано экспериментально, используя секундомер.

| ШИМ | Левое колесо, с. | Правое колесо, с. |
| ------------- | ------------- | ------------- |
| 100 | 1,836  | 1,630  |
| 150 | 1,536  | 1,333  |
| 200 | 1,446  | 1,136  |

*Таблица 1. За какое время робот проедет 14,13 см.*

Средние значения времени в секундах были получены эмпирическим методом при помощи засекания времени трех попыток проезда робота.

Левое колесо:

| ШИМ | Первая попытка | Вторая попытка | Третья попытка |
| ------------- | ------------- | ------------- | ------------- |
| 100 | 1,90 | 1,93  | 1,68  |
| 150 | 1,66  | 1,52  | 1,43  |
| 200 | 1,71  | 1,15  | 1,48  |

*Таблица 2. Время проката левого колеса*

Правое колесо:

| ШИМ | Первая попытка | Вторая попытка | Третья попытка |
| ------------- | ------------- | ------------- | ------------- |
| 100 | 1,87 | 1,42  | 1,60  |
| 150 | 1,40  | 1,05  | 1,55  |
| 200 | 1,02  | 1,18  | 1,21  |

*Таблица 3. Время проката правого колеса*


Полученные данные были использованы для построения графика зависимости времени от ШИМ для каждого колеса. Это поможет визуально оценить, как изменяется время проезда при изменении ШИМ.
Был построен график и определены коэффициенты для двух колёс по отдельности, с помощью двух систем уравнений.
Уравнение прямой: $y=kx+b$.

<details>

<summary>Была получена формула прямой для правого колеса: $y = 0,00044x + 0,0426$, где $k=0,00044$ и $b=0,0426$</summary>

### Правое колесо:

| Дано: | Решение: |
| ------------- | ------------- |
| $S=14,13см$, $t1=1,63с$, $t2=1,3с$| $V=S/t1=0,1413/1,63=0,0866 м/с$, $V=S/t2=0,1413/1,3=0,1086 м/с$ | 


1) Координаты: (100; 0,0866)
$0,0866 = 100k + b$
3) Координаты: (150; 0,1086)
$0,1086 = 150k + b$

</details>

![image](https://github.com/user-attachments/assets/0947e333-36de-45e8-b944-cf582ed763ed)
*Рис. 3. График функции зависимости скорости от ШИМ правого колеса*


<details>

<summary>Была получена формула прямой для левого: $y = 0,00030x + 0,0470$, где $k=0,0003$ и $b=0,0470$</summary>

### Левое колесо:

| Дано: | Решение: |
| ------------- | ------------- |
| $S=14,13см$, $t1=1,836с$, $t2=1,536с$| $V=S/t1=0,1413/1,836=0,07696 м/с$, $V=S/t2=0,1413/1,536=0,0979 м/с$ | 

1) Координаты: (100; 0,77)
$0,77 = 100k + b$
3) Координаты: (150; 0,92)
$0,92 = 150k + b$

</details>

![image](https://github.com/user-attachments/assets/b100042d-54e5-421d-81f0-257caa357ca6)
*Рис. 3. График функции зависимости скорости от ШИМ левого колеса*

| Колесо | Коэффициент |
| --- | --- |
| Левое колесо | 0,00030 |
| Правое колесо | 0,00044 |

*Таблица 4. Коэффициенты колёс*


![image](https://github.com/user-attachments/assets/3faa370e-38a6-4ca1-b4cf-3d692259cdd3)
*Рис. 5. Схема коммуникации между агентами*


![image](https://github.com/user-attachments/assets/427d3946-0980-4e5c-ac82-72490a4048ae)
*Рис. 6. П-регулятор*

![image](https://github.com/user-attachments/assets/31e8a5ae-e07e-4364-bcf6-5872769b1e83)
*Рис. 7. Регулятор функции Ляпунова*

![image](https://github.com/user-attachments/assets/21592372-d9f4-4a50-96da-e374663f6229)
*Рис. 8.*

![image](https://github.com/user-attachments/assets/6e8463fa-b53d-4dc7-bfa7-f5a9311a1516)

*Рис. 9.*

![image](https://github.com/user-attachments/assets/d4508280-f7a1-4794-8627-53c6efef484f)

*Рис. 10.*

P.S. @koipaqwe :+1: крутой md-файл! - я считаю, что слей :shipit:
