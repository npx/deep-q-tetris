from board import make_board, can_place
from pieces import make_piece, get_rotation

if __name__ == "__main__":
    board = make_board()
    t = make_piece("t")
    t_right = get_rotation(t, 1)
    t_right2 = get_rotation(t, 1)
    print(can_place(board, t_right, (0, -1)))
