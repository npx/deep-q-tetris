import pieces
from board import can_place, Board, Position

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


def get_kick(board: Board, piece: pieces.Piece, direction: int, pos: Position) -> Position | None:
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
