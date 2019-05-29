import os

class Result:
    def __init__(self, n, b_size, version, n_mat, time):
        self.n = n
        self.b_size = b_size
        self.version = version
        self.n_mat = n_mat
        self.time = time
    def __str__(self):
        return "%s %s %s %s %s" % (self.n, self.b_size, self.version, self.n_mat, self.time)

def read_file(filename):
    with open(filename) as file:
        lines = file.readlines()
    return lines

def extract_lines(all_lines):
    proper_lines = []
    for i in range(0, len(all_lines), 11):
        proper_lines.append(all_lines[i+1][13:-1])
        proper_lines.append(all_lines[i+2][12:-1])
        proper_lines.append(all_lines[i+3][14:-1])
        proper_lines.append(all_lines[i+4][11:-1])
        proper_lines.append(all_lines[i+6][6:-6])
    return proper_lines

def build_results_list(proper_lines):
    results = []
    for i in range(0, len(proper_lines), 5):
        n = proper_lines[i]
        b = proper_lines[i+1]
        v = proper_lines[i+2]
        n_mat = proper_lines[i+3]
        t = proper_lines[i+4]
        results.append(Result(n,b,v,n_mat,t))
    return results

def print_results(results):
    for r in results:
        print(r)
        
def process_file(filename):
    print_results(build_results_list(extract_lines(read_file(filename))))

def main():
    for _, _, files in os.walk("txt/"):
            for filename in files:
                process_file("txt/"+filename)

if __name__ == "__main__":
    main()
