# MST CUDA

A parallel and distributed system: Implement Minimum Spanning Tree using CUDA.

Kelompok 35:<br>
- 13518044 Jun Ho Choi Hedyatmo
- 13518149 Stefanus Gusega Gunawan

## Cara Kerja Algoritma CUDA untuk MST
Versi MST yang digunakan adalah Algoritma Prim dengan menggunakan dense graph (versi O(v^2)), jika melihat referensi di cp-algorithms.com. Intinya, untuk tiap vertex yang belum dikunjungi akan diambil minimum weighted edge (direpresentasikan pada array of pair bernama min_edge, yg menyimpan weight dan vertex yang akan dituju). Lalu, hal yang diparalelisasi adalah proses dalam mencari minimum weighted edge (prosedur reduceMin). Dimulai dari dengan inisialisasi idxmin dan minval, lalu dipassing ke reduceMin dengan ukuran block sebesar 256 thread per block. Lalu, reduceMin dieksekusi di masing-masing thread secara paralel. Lalu, dalam pencarian nilai minimum dan indeks yang menyebabkan nilai minimum, dilakukan iterasi salah satu thread dari masing-masing block (ditentukan menggunakan variable index pada prosedur reduceMin). Dan, ketika local minimum index dan local minimum value sudah ditemukan, maka diperlakukan operasi atomik minimum untuk menghindari race condition dengan thread lainnya. Lalu, nilai minimum index secara global akan diupdate jika memang global minimum value yang dipassing ke prosedur sama dengan local minimum value. Hal ini bisa diparalelisasi, karena ketika vertex-vertex dibagi untuk dicek, bisa dicari minimum weight edge nya secara independen dengan vertex lain (di thread berbeda). Dan, untuk menghindari race condition juga bisa diatasi dengan prosedur atomic.

## Analisis kerja program
Grafik analisis dapat dilihat pada <https://docs.google.com/spreadsheets/d/1_byySQDTO7kYNruibIfGasLZ0Q71-8OyUwPMGdoEdQM/edit?usp=sharing>. Pada grafik ditunjukkan bahwa kinerja dari algoritma ini mendekati linear, yaitu dengan O(n). Di mana jauh lebih cepat daripada eksekusi serialnya, yang memiliki kompleksitas O(n^2).
