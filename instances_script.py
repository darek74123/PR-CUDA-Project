from os import listdir, path


BASE_DIR_PATH = path.dirname(path.abspath("__file__"))
INSTANCES_PATH = path.join(BASE_DIR_PATH, "instancje")

instances_filenames = [path.join(INSTANCES_PATH, f) for f in listdir(INSTANCES_PATH) if path.isfile(path.join(INSTANCES_PATH, f))]
result_filename = path.join(BASE_DIR_PATH, "all_instances.csv")


keys = [i.split(" ") for i in instances_filenames]
keys = [int(k[0][k[0].rfind('/') + 1:] + k[1].zfill(2) + k[2] + k[3][:-4].zfill(2)) for k in keys]
after_sort = list(zip(*sorted(zip(keys,instances_filenames), key = lambda t: t[0])))[1]

with open(result_filename, 'w') as result_file:    
    for filename in after_sort:
        with open(filename, 'r') as file:
            a, s, d, f = filename.split(" ")
            a = a[a.rfind('/') + 1:]
            f = f[:-4]
#             print("Rozmiar macierzy: " + a + ", Rozmiar bloku: " + s + ", Liczba obliczanych macierzy: " + f + ", Wersja kodu: " + d + ",,,,,,,,,,,,,,,,,,,,,,,,,,,\n")
            result_file.write("Rozmiar macierzy: " + a + ", Rozmiar bloku: " + s + ", Liczba obliczanych macierzy: " + f + ", Wersja kodu: " + d + ",,,,,,,,,,,,,,,,,,,,,,,,,,,\n")
            result_file.write(file.read())
            result_file.write(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\n")
            result_file.write(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\n")            
            
