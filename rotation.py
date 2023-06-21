import pieces
from board import can_place, make_board, print_board, place_piece

# TODO 180s
NOT_I_KICKS = {
    (0, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
    (0, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],

    (1, 0): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
    (1, 2): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],

    (2, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
    (2, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],

    (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
}

I_KICKS = {
    (0, 1):	[(0, 0), (-2, 0), (+1, 0), (-2, -1), (+1, +2)],
    (0, 3):	[(0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)],

    (1, 0):	[(0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)],
    (1, 2):	[(0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)],

    (2, 1):	[(0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)],
    (2, 3):	[(0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)],

    (3, 2):	[(0, 0), (-2, 0), (+1, 0), (-2, -1), (+1, +2)],
    (3, 0):	[(0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)],
}


def get_kick(board, piece, direction, pos):
    name, fr = piece

    if name == "o":
        return pos

    rotated = pieces.get_rotation(piece, direction)
    _, to = rotated

    kicks = NOT_I_KICKS[(fr, to)] if name != "i" else I_KICKS

    for kick in kicks:
        row, col = pos
        x, y = kick
        kicked_position = (row - y, col + x)
        if can_place(board, rotated, kicked_position):
            return kicked_position

    return None


if __name__ == "__main__":
    tst_board = """
    xx........
    x.........
    x.xxxxxxxx
    x..xxxxxxx
    x.xxxxxxxx
    """

    piece = pieces.make_piece("t")
    pos = (0, 1)

    board = make_board(tst_board)
    print_board(board)

    placed = place_piece(board, piece, pos)
    print_board(placed)

    kick = get_kick(board, piece, 1, pos)
    if kick:
        rotated = pieces.get_rotation(piece, 1)
        placed = place_piece(board, rotated, kick)
    print_board(placed)
