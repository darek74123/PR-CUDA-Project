from os import listdir, path


BASE_DIR_PATH = path.dirname(path.abspath("__file__"))
INSTANCES_PATH = path.join(BASE_DIR_PATH, "instancje")

instances_filenames = [path.join(INSTANCES_PATH, f) for f in listdir(INSTANCES_PATH) if path.isfile(path.join(INSTANCES_PATH, f))]
result_filename = path.join(BASE_DIR_PATH, "all_instances.csv")


with open(result_filename, 'w') as result_file:    
    for filename in instances_filenames:
        with open(filename, 'r') as file:
            a, s, d, f = filename.split(" ")
            a = a[a.rfind('/') + 1:]
            f = f[:-4]
            result_file.write("Rozmiar macierzy: " + a + ", Rozmiar bloku: " + s + ", Liczba obliczanych macierzy: " + f + ", Wersja kodu: " + d + ",,,,,,,,,,,,,,,,,,,,,,,,,,\n")
            result_file.write(file.read())
            result_file.write("\n\n")
