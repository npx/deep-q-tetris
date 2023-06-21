from i_piece import i_piece
from j_piece import j_piece
from l_piece import l_piece
from o_piece import o_piece
from s_piece import s_piece
from t_piece import t_piece
from z_piece import z_piece

PIECES = {
    "i": i_piece,
    "j": j_piece,
    "l": l_piece,
    "o": o_piece,
    "s": s_piece,
    "t": t_piece,
    "z": z_piece,
}


def make_piece(name):
    return (name, 0)


def get_rotation(piece, direction):
    name, rotation = piece
    return (name, (rotation+direction) % 4)


def get_dim(piece):
    name, rotation = piece
    piece = get_blocks(piece)
    return (len(piece), len(piece[0]))


def get_blocks(piece):
    name, rotation = piece
    return PIECES[name][rotation]
