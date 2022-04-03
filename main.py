import numpy as np


PAM250 = {
    'A': {'A': 2, 'C': -2, 'D': 0, 'E': 0, 'F': -3, 'G': 1, 'H': -1, 'I': -1, 'K': -1, 'L': -2, 'M': -1, 'N': 0, 'P': 1,
          'Q': 0, 'R': -2, 'S': 1, 'T': 1, 'V': 0, 'W': -6, 'Y': -3},
    'C': {'A': -2, 'C': 12, 'D': -5, 'E': -5, 'F': -4, 'G': -3, 'H': -3, 'I': -2, 'K': -5, 'L': -6, 'M': -5, 'N': -4,
          'P': -3, 'Q': -5, 'R': -4, 'S': 0, 'T': -2, 'V': -2, 'W': -8, 'Y': 0},
    'D': {'A': 0, 'C': -5, 'D': 4, 'E': 3, 'F': -6, 'G': 1, 'H': 1, 'I': -2, 'K': 0, 'L': -4, 'M': -3, 'N': 2, 'P': -1,
          'Q': 2, 'R': -1, 'S': 0, 'T': 0, 'V': -2, 'W': -7, 'Y': -4},
    'E': {'A': 0, 'C': -5, 'D': 3, 'E': 4, 'F': -5, 'G': 0, 'H': 1, 'I': -2, 'K': 0, 'L': -3, 'M': -2, 'N': 1, 'P': -1,
          'Q': 2, 'R': -1, 'S': 0, 'T': 0, 'V': -2, 'W': -7, 'Y': -4},
    'F': {'A': -3, 'C': -4, 'D': -6, 'E': -5, 'F': 9, 'G': -5, 'H': -2, 'I': 1, 'K': -5, 'L': 2, 'M': 0, 'N': -3,
          'P': -5, 'Q': -5, 'R': -4, 'S': -3, 'T': -3, 'V': -1, 'W': 0, 'Y': 7},
    'G': {'A': 1, 'C': -3, 'D': 1, 'E': 0, 'F': -5, 'G': 5, 'H': -2, 'I': -3, 'K': -2, 'L': -4, 'M': -3, 'N': 0, 'P': 0,
          'Q': -1, 'R': -3, 'S': 1, 'T': 0, 'V': -1, 'W': -7, 'Y': -5},
    'H': {'A': -1, 'C': -3, 'D': 1, 'E': 1, 'F': -2, 'G': -2, 'H': 6, 'I': -2, 'K': 0, 'L': -2, 'M': -2, 'N': 2, 'P': 0,
          'Q': 3, 'R': 2, 'S': -1, 'T': -1, 'V': -2, 'W': -3, 'Y': 0},
    'I': {'A': -1, 'C': -2, 'D': -2, 'E': -2, 'F': 1, 'G': -3, 'H': -2, 'I': 5, 'K': -2, 'L': 2, 'M': 2, 'N': -2,
          'P': -2, 'Q': -2, 'R': -2, 'S': -1, 'T': 0, 'V': 4, 'W': -5, 'Y': -1},
    'K': {'A': -1, 'C': -5, 'D': 0, 'E': 0, 'F': -5, 'G': -2, 'H': 0, 'I': -2, 'K': 5, 'L': -3, 'M': 0, 'N': 1, 'P': -1,
          'Q': 1, 'R': 3, 'S': 0, 'T': 0, 'V': -2, 'W': -3, 'Y': -4},
    'L': {'A': -2, 'C': -6, 'D': -4, 'E': -3, 'F': 2, 'G': -4, 'H': -2, 'I': 2, 'K': -3, 'L': 6, 'M': 4, 'N': -3,
          'P': -3, 'Q': -2, 'R': -3, 'S': -3, 'T': -2, 'V': 2, 'W': -2, 'Y': -1},
    'M': {'A': -1, 'C': -5, 'D': -3, 'E': -2, 'F': 0, 'G': -3, 'H': -2, 'I': 2, 'K': 0, 'L': 4, 'M': 6, 'N': -2,
          'P': -2, 'Q': -1, 'R': 0, 'S': -2, 'T': -1, 'V': 2, 'W': -4, 'Y': -2},
    'N': {'A': 0, 'C': -4, 'D': 2, 'E': 1, 'F': -3, 'G': 0, 'H': 2, 'I': -2, 'K': 1, 'L': -3, 'M': -2, 'N': 2, 'P': 0,
          'Q': 1, 'R': 0, 'S': 1, 'T': 0, 'V': -2, 'W': -4, 'Y': -2},
    'P': {'A': 1, 'C': -3, 'D': -1, 'E': -1, 'F': -5, 'G': 0, 'H': 0, 'I': -2, 'K': -1, 'L': -3, 'M': -2, 'N': 0,
          'P': 6, 'Q': 0, 'R': 0, 'S': 1, 'T': 0, 'V': -1, 'W': -6, 'Y': -5},
    'Q': {'A': 0, 'C': -5, 'D': 2, 'E': 2, 'F': -5, 'G': -1, 'H': 3, 'I': -2, 'K': 1, 'L': -2, 'M': -1, 'N': 1, 'P': 0,
          'Q': 4, 'R': 1, 'S': -1, 'T': -1, 'V': -2, 'W': -5, 'Y': -4},
    'R': {'A': -2, 'C': -4, 'D': -1, 'E': -1, 'F': -4, 'G': -3, 'H': 2, 'I': -2, 'K': 3, 'L': -3, 'M': 0, 'N': 0,
          'P': 0, 'Q': 1, 'R': 6, 'S': 0, 'T': -1, 'V': -2, 'W': 2, 'Y': -4},
    'S': {'A': 1, 'C': 0, 'D': 0, 'E': 0, 'F': -3, 'G': 1, 'H': -1, 'I': -1, 'K': 0, 'L': -3, 'M': -2, 'N': 1, 'P': 1,
          'Q': -1, 'R': 0, 'S': 2, 'T': 1, 'V': -1, 'W': -2, 'Y': -3},
    'T': {'A': 1, 'C': -2, 'D': 0, 'E': 0, 'F': -3, 'G': 0, 'H': -1, 'I': 0, 'K': 0, 'L': -2, 'M': -1, 'N': 0, 'P': 0,
          'Q': -1, 'R': -1, 'S': 1, 'T': 3, 'V': 0, 'W': -5, 'Y': -3},
    'V': {'A': 0, 'C': -2, 'D': -2, 'E': -2, 'F': -1, 'G': -1, 'H': -2, 'I': 4, 'K': -2, 'L': 2, 'M': 2, 'N': -2,
          'P': -1, 'Q': -2, 'R': -2, 'S': -1, 'T': 0, 'V': 4, 'W': -6, 'Y': -2},
    'W': {'A': -6, 'C': -8, 'D': -7, 'E': -7, 'F': 0, 'G': -7, 'H': -3, 'I': -5, 'K': -3, 'L': -2, 'M': -4, 'N': -4,
          'P': -6, 'Q': -5, 'R': 2, 'S': -2, 'T': -5, 'V': -6, 'W': 17, 'Y': 0},
    'Y': {'A': -3, 'C': 0, 'D': -4, 'E': -4, 'F': 7, 'G': -5, 'H': 0, 'I': -1, 'K': -4, 'L': -1, 'M': -2, 'N': -2,
          'P': -5, 'Q': -4, 'R': -4, 'S': -3, 'T': -3, 'V': -2, 'W': 0, 'Y': 10}
}


def create_semiglobal(matrix, first_seq, second_seq):
    traceBack = {}
    for i in range(1, len(second_seq) + 1):
        for j in range(1, len(first_seq) + 1):
            matrix[i][j] = max(matrix[i - 1][j - 1] + PAM250.get(second_seq[i - 1]).get(first_seq[j - 1]),
                               matrix[i][j - 1] - 9, matrix[i - 1][j] - 9)

            if matrix[i][j] == matrix[i - 1][j - 1] + PAM250.get(second_seq[i - 1]).get(first_seq[j - 1]):
                if (i, j) not in traceBack:
                    traceBack[(i, j)] = []
                traceBack[(i, j)].append((i - 1, j - 1))
            if matrix[i][j] == matrix[i][j - 1] - 9:
                if (i, j) not in traceBack:
                    traceBack[(i, j)] = []
                traceBack[(i, j)].append((i, j - 1))
            if matrix[i][j] == matrix[i - 1][j] - 9:
                if (i, j) not in traceBack:
                    traceBack[(i, j)] = []
                traceBack[(i, j)].append((i - 1, j))
    return traceBack


def find_score(matrix):
    max_row = np.amax(matrix[-1, :])
    max_col = np.amax(matrix[:, -1])
    score = max(max_row, max_col)
    return score


def find_max(matrix):
    xsize, ysize = matrix.shape
    max_row = np.amax(matrix[-1, :])
    max_col = np.amax(matrix[:, -1])
    score = max(max_row, max_col)
    ind_list = []
    for ind in np.argwhere(matrix == score):
        if ind[0] == xsize - 1 or ind[1] == ysize - 1:
            ind_list.append(tuple(ind))
    return ind_list


def trace_backward(tb, ind, res1, res2, matrix, save_last, hamechi, kolln):
    back_ind_list = tb[ind]

    for back_ind in back_ind_list:
        if ind[0] == back_ind[0]:
            new_res2 = "-" + res2
            new_res1 = first_seq[ind[1] - 1] + res1

        if ind[1] == back_ind[1]:
            new_res1 = "-" + res1
            new_res2 = second_seq[ind[0] - 1] + res2

        if ind[0] != back_ind[0] and ind[1] != back_ind[1]:
            new_res1 = first_seq[ind[1] - 1] + res1
            new_res2 = second_seq[ind[0] - 1] + res2

        if matrix[back_ind] == 0 and (back_ind[0] == 0 or back_ind[1] == 0):
            new_res1, new_res2 = gap_first(back_ind[0], back_ind[1], new_res1, new_res2)
            new_res1, new_res2 = gap_last(save_last[0], save_last[1], new_res1, new_res2, matrix)
            hamechi.append((new_res1, new_res2))
            if len(hamechi) == kolln:
                sort_list(hamechi)
            return
        trace_backward(tb, back_ind, new_res1, new_res2, matrix, save_last, hamechi, kolln)


def gap_first(ind_x, ind_y, res1, res2):
    for i in range(ind_x, 0, -1):
        res1 = "-" + res1
        res2 = second_seq[i - 1] + res2
    for j in range(ind_y, 0, -1):
        res2 = "-" + res2
        res1 = first_seq[j - 1] + res1
    return res1, res2


def gap_last(ind_x, ind_y, res1, res2, matrix):
    xsize, ysize = matrix.shape
    # for i in range(xsize - ind_x - 1):
    for i in range(ind_x, xsize-1):
        res1 = res1 + "-"
        res2 = res2 + second_seq[i]
    # for j in range(ysize - ind_y - 1):
    for j in range(ind_y, ysize-1):
        res2 = res2 + "-"
        res1 = res1 + first_seq[j]
    return res1, res2


def sort_list(seq):
    sortedSeq = [i[0] + i[1] for i in seq]
    sortedSeq.sort()
    for i in sortedSeq:
        print(i[0:int(len(i) / 2)])
        print(i[int(len(i) / 2):])


if __name__ == '__main__':
    # first_seq = input()
    # second_seq = input()
    first_seq = 'HEAGAWGHE'
    second_seq = 'PAWHEA'
    mat = np.zeros((len(second_seq) + 1, len(first_seq) + 1))
    local = create_semiglobal(mat, first_seq, second_seq)
    score = find_score(mat)
    print(int(score))
    print(mat)

    if score == 0:
        exit(0)

    index = find_max(mat)
    hamechi = []
    for ind in index:
        trace_backward(local, ind, "", "", mat, ind, hamechi, len(index))
